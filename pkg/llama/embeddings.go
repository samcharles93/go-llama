package llama

/*
#include "llama.h"
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"math"
	"unsafe"
)

// Embed returns the embedding vector for a single text.
//
// The context must have been created with ContextOptions.Embeddings set. For
// pooled models (PoolingMean, PoolingCLS, PoolingLast) the vector is the pooled
// representation of the whole input; with PoolingNone it is the final token's
// hidden state, which is what decoder-only models produce.
//
// The vector is returned unnormalised: pass it to Normalize for unit length, or
// compare two of them with CosineSimilarity.
//
// Each call clears the KV cache first, so the same context can be reused for
// unrelated documents, but it cannot be used for chat at the same time.
//
// A pooled vector covers one llama.cpp micro-batch, which is 512 tokens by
// default, so a text longer than that is an error rather than a vector that
// covers only the tail of it.
func (c *Context) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := c.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	if len(vecs) != 1 {
		return nil, errorf(ErrEmbedding, "llama.cpp returned no embedding for the input")
	}
	return vecs[0], nil
}

// EmbedBatch embeds several texts. With pooling enabled the texts are decoded
// together, one sequence per text, so a batch of short texts costs one pass
// through the model instead of one pass each. The context must be created with
// ContextOptions.Sequences at least as large as len(texts).
//
// With PoolingNone there is no per-sequence vector to read, so the texts are
// decoded one at a time instead.
//
// Like Embed, each call clears the KV cache, so the context cannot serve chat at
// the same time. The context lock is held for the whole call, so a batch is not
// cancellable inside a decode: cancellation is checked between texts and between
// groups of texts.
func (c *Context) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if c == nil || c.closed.Load() {
		return nil, ErrClosed
	}
	if !c.embeddings {
		return nil, errorf(ErrEmbedding, "context was created without Embeddings enabled; set ContextOptions.Embeddings")
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if c.model == nil {
		return nil, errorf(ErrEmbedding, "context is not attached to a model")
	}
	nEmbd := c.model.nEmbdOut
	if nEmbd <= 0 {
		return nil, errorf(ErrEmbedding, "model %s has no embedding output", c.model.path)
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ptr == nil {
		return nil, ErrClosed
	}
	if len(texts) == 0 {
		return nil, nil
	}
	if c.poolingType() == PoolingNone {
		return c.embedSequential(ctx, texts, nEmbd)
	}
	return c.embedPooled(ctx, texts, nEmbd)
}

// poolingType reports the pooling the context actually uses. ContextOptions
// leaves it unspecified by default, and llama.cpp then takes the model's own
// pooling_type, so ContextOptions.Pooling is not the answer.
func (c *Context) poolingType() PoolingType {
	return PoolingType(int32(C.llama_pooling_type(c.ptr)))
}

// embedPooled decodes every text as its own sequence and reads back the pooled
// vector of each. Callers hold c.mu.
//
// llama.cpp pools a sequence inside one micro-batch and keeps only the micro-batch
// that last touched the sequence, so a sequence split across micro-batches would
// be pooled from its tail alone. The texts are therefore grouped so that a group
// fits the micro-batch, and each group is decoded in one call.
func (c *Context) embedPooled(ctx context.Context, texts []string, nEmbd int) ([][]float32, error) {
	nSeqMax := int(C.llama_n_seq_max(c.ptr))
	if len(texts) > nSeqMax {
		return nil, errorf(ErrEmbedding, "cannot embed %d texts in one batch: the context holds %d sequences, so ContextOptions.Sequences must be at least %d", len(texts), nSeqMax, len(texts))
	}
	limit := min(int(C.llama_n_ubatch(c.ptr)), c.nBatch)

	tokens := make([][]Token, len(texts))
	for i, text := range texts {
		if err := ctx.Err(); err != nil {
			return nil, wrap(ErrEmbedding, "embed", err)
		}
		seq, err := c.model.vocab.Tokenize(text, true)
		if err != nil {
			return nil, err
		}
		if len(seq) == 0 {
			return nil, errorf(ErrEmbedding, "text %d produced no tokens", i)
		}
		if len(seq) > limit {
			return nil, errorf(ErrEmbedding, "text %d is %d tokens, but pooled embeddings need the whole text in one llama.cpp micro-batch of %d tokens (512 by default)", i, len(seq), limit)
		}
		tokens[i] = seq
	}

	batch, err := NewBatch(int32(limit), int32(len(texts)))
	if err != nil {
		return nil, err
	}
	defer batch.Close()

	out := make([][]float32, len(texts))
	for start := 0; start < len(texts); {
		end, nTokens := start, 0
		for end < len(texts) && nTokens+len(tokens[end]) <= limit {
			nTokens += len(tokens[end])
			end++
		}
		if err := ctx.Err(); err != nil {
			return nil, wrap(ErrEmbedding, "embed", err)
		}

		batch.Reset()
		for seq := start; seq < end; seq++ {
			for pos, tok := range tokens[seq] {
				if err := batch.Add(tok, int32(pos), []SeqID{int32(seq - start)}, true); err != nil {
					return nil, err
				}
			}
		}
		C.llama_memory_clear(C.llama_get_memory(c.ptr), true)
		c.history = c.history[:0]
		if err := c.decode(batch); err != nil {
			return nil, err
		}
		for seq := start; seq < end; seq++ {
			ptr := C.llama_get_embeddings_seq(c.ptr, C.llama_seq_id(seq-start))
			if ptr == nil {
				return nil, errorf(ErrEmbedding, "llama.cpp returned no embedding for sequence %d", seq-start)
			}
			out[seq] = copyEmbedding(ptr, nEmbd)
		}
		start = end
	}
	return out, nil
}

// embedSequential decodes one text at a time, which is the only way to read
// per-token embeddings. Callers hold c.mu.
func (c *Context) embedSequential(ctx context.Context, texts []string, nEmbd int) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for _, text := range texts {
		if err := ctx.Err(); err != nil {
			return out, wrap(ErrEmbedding, "embed", err)
		}
		vec, err := c.embedOne(text, nEmbd)
		if err != nil {
			return out, err
		}
		out = append(out, vec)
	}
	return out, nil
}

// embedOne embeds one text with pooling disabled and returns the hidden state of
// its final token. Callers hold c.mu.
func (c *Context) embedOne(text string, nEmbd int) ([]float32, error) {
	tokens, err := c.model.vocab.Tokenize(text, true)
	if err != nil {
		return nil, err
	}
	if len(tokens) == 0 {
		return nil, errorf(ErrEmbedding, "input produced no tokens")
	}
	if len(tokens) > c.nCtx {
		return nil, errorf(ErrEmbedding, "input is %d tokens, context holds %d", len(tokens), c.nCtx)
	}

	C.llama_memory_clear(C.llama_get_memory(c.ptr), true)
	c.history = c.history[:0]

	// Embeddings need output on every token so every position is available,
	// unlike generation which only needs the last one.
	for start := 0; start < len(tokens); start += c.nBatch {
		end := start + c.nBatch
		if end > len(tokens) {
			end = len(tokens)
		}
		if err := c.decodeRange(tokens[start:end], int32(start), false, true); err != nil {
			return nil, err
		}
	}

	ptr := C.llama_get_embeddings_ith(c.ptr, C.int32_t(len(tokens)-1))
	if ptr == nil {
		return nil, errorf(ErrEmbedding, "llama.cpp returned no token embeddings (pooling none)")
	}
	return copyEmbedding(ptr, nEmbd), nil
}

// copyEmbedding takes a copy of llama.cpp's buffer, which the next decode
// overwrites.
func copyEmbedding(ptr *C.float, nEmbd int) []float32 {
	raw := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
	vec := make([]float32, nEmbd)
	copy(vec, raw)
	return vec
}

// Normalize scales vec to unit length in place and returns it, so
// CosineSimilarity and dot products rank the same way on every vector.
//
// An empty vector, an all-zero vector and a vector holding a non-finite value
// are left untouched: there is no scale that would make them unit length, and
// dividing by a zero norm would fill them with NaN or Inf.
func Normalize(vec []float32) []float32 {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	norm := math.Sqrt(sum)
	if norm == 0 || math.IsInf(norm, 0) || math.IsNaN(norm) {
		return vec
	}
	for i, v := range vec {
		vec[i] = float32(float64(v) / norm)
	}
	return vec
}

// CosineSimilarity returns the cosine of the angle between a and b, in the range
// [-1, 1]. The vectors need not be normalised.
//
// The result is NaN when the angle is undefined: when either vector is empty,
// when the lengths differ, or when either vector has zero magnitude. Test with
// math.IsNaN rather than comparing against a sentinel, so a genuine 0 similarity
// stays distinguishable from an unanswerable one.
func CosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return math.NaN()
	}
	var dot, normA, normB float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		normA += x * x
		normB += y * y
	}
	if normA == 0 || normB == 0 {
		return math.NaN()
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
