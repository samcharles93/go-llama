package llama

/*
#include "llama.h"
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"unsafe"
)

// Embed returns the embedding vector for text.
//
// The context must have been created with ContextOptions.Embeddings set. For
// pooled models (PoolingMean, PoolingCLS, PoolingLast) the vector is the pooled
// representation of the whole input; with PoolingNone it is the final token's
// hidden state, which is what decoder-only models produce.
//
// The vector is returned unnormalised: apply L2 normalisation in the caller if
// your index compares with cosine similarity.
//
// Each call clears the KV cache first, so the same context can be reused for
// unrelated documents, but it cannot be used for chat at the same time.
func (c *Context) Embed(ctx context.Context, text string) ([]float32, error) {
	vecs, err := c.EmbedBatch(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedBatch embeds several texts, one at a time, reusing the same KV cache.
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
	nEmbd := c.model.nEmbdOut
	if nEmbd <= 0 {
		return nil, errorf(ErrEmbedding, "model %s has no embedding output", c.model.path)
	}

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

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ptr == nil {
		return nil, ErrClosed
	}
	C.llama_memory_clear(C.llama_get_memory(c.ptr), true)
	c.history = c.history[:0]

	// Embeddings need output on every token so the pooling stage can see the
	// whole sequence, unlike generation which only needs the last position.
	for start := 0; start < len(tokens); start += c.nBatch {
		end := start + c.nBatch
		if end > len(tokens) {
			end = len(tokens)
		}
		if err := c.decodeRange(tokens[start:end], int32(start), false, true); err != nil {
			return nil, err
		}
	}

	var ptr *C.float
	switch c.pooling {
	case PoolingNone, PoolingUnspecified:
		ptr = C.llama_get_embeddings_ith(c.ptr, C.int32_t(len(tokens)-1))
	default:
		ptr = C.llama_get_embeddings_seq(c.ptr, 0)
	}
	if ptr == nil {
		return nil, errorf(ErrEmbedding, "llama.cpp returned no embeddings (pooling %s)", c.pooling)
	}

	raw := unsafe.Slice((*float32)(unsafe.Pointer(ptr)), nEmbd)
	vec := make([]float32, nEmbd)
	copy(vec, raw)
	return vec, nil
}
