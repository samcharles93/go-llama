package llama

/*
#include "llama.h"
#include <stdlib.h>
*/
import "C"

import (
	"context"
	"errors"
	"fmt"
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

// EmbedBatch embeds several texts, one decode per text, reusing the same KV
// cache.
//
// The texts are not submitted as one multi-sequence batch. llama.cpp pools a
// sequence's embedding per micro-batch, and when a decode carries sequences of
// different lengths the sequences grouped with a shorter one come back pooled
// with the wrong token set: measured on a Qwen2 0.5B model, sequences 0 and 2 of
// lengths 3 and 3 matched their single-text vector exactly while the middle
// sequence of length 4 was off by a cosine of 0.56. A wrong vector is worse than
// a fast one, so each text gets its own decode.
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

	out := make([][]float32, 0, len(texts))
	for i, text := range texts {
		if err := ctx.Err(); err != nil {
			return out, wrap(ErrEmbedding, "embed", err)
		}
		vec, err := c.embedOne(text, nEmbd)
		if err != nil {
			if errors.Is(err, ErrEmbedding) {
				return out, fmt.Errorf("%w (text %d)", err, i)
			}
			return out, err
		}
		out = append(out, vec)
	}
	return out, nil
}

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
	// Pooling happens per micro-batch, so a text that spans more than one would
	// come back as the pooled tail rather than the whole input.
	if c.nUBatch > 0 && len(tokens) > c.nUBatch {
		return nil, errorf(ErrEmbedding, "input is %d tokens but pooled embeddings need the whole text in one micro-batch of %d; raise ContextOptions.UBatchSize", len(tokens), c.nUBatch)
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ptr == nil {
		return nil, ErrClosed
	}
	C.llama_memory_clear(C.llama_get_memory(c.ptr), true)
	c.history = c.history[:0]

	// Embeddings need output on every token so the pooling stage can see the
	// whole sequence, unlike generation which only needs the last position.
	if err := c.decodeRange(tokens, 0, false, true); err != nil {
		return nil, err
	}

	// The vector has to come from the same place the batched path reads it, or
	// Embed and EmbedBatch answer with different things: a pooled model returns
	// one vector per sequence, and only with pooling disabled is the last token's
	// hidden state the answer.
	var ptr *C.float
	if c.poolingType() == PoolingNone {
		ptr = C.llama_get_embeddings_ith(c.ptr, C.int32_t(len(tokens)-1))
	} else {
		ptr = C.llama_get_embeddings_seq(c.ptr, 0)
	}
	if ptr == nil {
		return nil, errorf(ErrEmbedding, "llama.cpp returned no embeddings for a single text (pooling %s)", c.poolingType())
	}
	return copyEmbedding(ptr, nEmbd), nil
}

// poolingType reports the pooling mode llama.cpp resolved for this context, which
// can differ from ContextOptions.Pooling when the model states its own default.
func (c *Context) poolingType() PoolingType {
	return PoolingType(int32(C.llama_pooling_type(c.ptr)))
}

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
