//go:build integration

package llama

import (
	"errors"
	"math"
	"os"
	"strings"
	"testing"
)

// These tests need a real embedding model, so they run under the integration tag
// and skip unless GO_LLAMA_TEST_EMBED_MODEL (and, for the causal comparison,
// GO_LLAMA_TEST_MODEL) is set.

func embedModel(t *testing.T) *Model {
	t.Helper()
	path := os.Getenv("GO_LLAMA_TEST_EMBED_MODEL")
	if path == "" {
		t.Skip("GO_LLAMA_TEST_EMBED_MODEL is not set")
	}
	// GPULayers 0 keeps the test on the CPU so parallel lanes do not contend for
	// the card.
	m, err := LoadModel(path, ModelParams{GPULayers: 0})
	if err != nil {
		t.Fatalf("LoadModel(%s): %v", path, err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

func embedContext(t *testing.T, m *Model, opts ContextOptions) *Context {
	t.Helper()
	c, err := m.NewContext(opts)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	t.Cleanup(func() { _ = c.Close() })
	return c
}

func TestEmbedVectorMatchesModelSize(t *testing.T) {
	m := embedModel(t)
	c := embedContext(t, m, ContextOptions{Embeddings: true, Pooling: PoolingMean, Sequences: 1})

	vec, err := c.Embed(t.Context(), "the cat sat on the mat")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(vec) != m.EmbeddingSize() {
		t.Fatalf("embedding has %d values, model reports %d", len(vec), m.EmbeddingSize())
	}
	unit := Normalize(append([]float32(nil), vec...))
	if n := vecLength(unit); math.Abs(n-1) > 1e-4 {
		t.Fatalf("normalised embedding has length %v, want 1", n)
	}
}

func TestEmbedBatchAgreesWithSingleEmbed(t *testing.T) {
	m := embedModel(t)
	texts := []string{
		"the cat sat on the mat",
		"a dog barked at the postman",
		"quantum chromodynamics describes the strong interaction",
		"lentil soup with rosemary and lemon",
		"the train to Bristol was late again",
	}
	c := embedContext(t, m, ContextOptions{Embeddings: true, Pooling: PoolingMean, Sequences: uint32(len(texts))})

	batch, err := c.EmbedBatch(t.Context(), texts)
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if len(batch) != len(texts) {
		t.Fatalf("EmbedBatch returned %d vectors for %d texts", len(batch), len(texts))
	}
	for i, text := range texts {
		if len(batch[i]) != m.EmbeddingSize() {
			t.Fatalf("text %d: vector has %d values, model reports %d", i, len(batch[i]), m.EmbeddingSize())
		}
		single, err := c.Embed(t.Context(), text)
		if err != nil {
			t.Fatalf("Embed(%q): %v", text, err)
		}
		// llama.cpp runs a bidirectional encoder through a different graph when
		// the micro-batch holds more than one sequence, so the same text is not
		// bit-identical across batch shapes. The vectors still have to point the
		// same way; TestEmbedBatchMatchesSingleEmbedExactly pins the bit-exact
		// case on a causal model.
		if sim := CosineSimilarity(single, batch[i]); sim < 0.9995 {
			t.Fatalf("text %d: batched and single embeddings have cosine %v, want at least 0.9995", i, sim)
		}
	}
}

func TestEmbedBatchMatchesSingleEmbedExactly(t *testing.T) {
	path := os.Getenv("GO_LLAMA_TEST_MODEL")
	if path == "" {
		t.Skip("GO_LLAMA_TEST_MODEL is not set")
	}
	// GPULayers 0 keeps the test on the CPU so parallel lanes do not contend for
	// the card.
	m, err := LoadModel(path, ModelParams{GPULayers: 0})
	if err != nil {
		t.Fatalf("LoadModel(%s): %v", path, err)
	}
	t.Cleanup(func() { _ = m.Close() })

	texts := []string{"the cat sat on the mat", "a dog barked at the postman"}
	c := embedContext(t, m, ContextOptions{Embeddings: true, Pooling: PoolingMean, Sequences: uint32(len(texts))})

	batch, err := c.EmbedBatch(t.Context(), texts)
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	for i, text := range texts {
		single, err := c.Embed(t.Context(), text)
		if err != nil {
			t.Fatalf("Embed(%q): %v", text, err)
		}
		if diff := maxAbsDiff(single, batch[i]); diff > 1e-4 {
			t.Fatalf("text %d: batched and single embeddings differ by %v", i, diff)
		}
	}
}

func TestEmbedBatchNamesTheSequenceLimit(t *testing.T) {
	m := embedModel(t)
	c := embedContext(t, m, ContextOptions{Embeddings: true, Pooling: PoolingMean, Sequences: 2})

	_, err := c.EmbedBatch(t.Context(), []string{"one", "two", "three"})
	if !errors.Is(err, ErrEmbedding) {
		t.Fatalf("EmbedBatch past the sequence limit = %v, want ErrEmbedding", err)
	}
	if !strings.Contains(err.Error(), "Sequences") {
		t.Fatalf("error %q does not name the sequence limit", err)
	}
}

func TestEmbedBatchWithPoolingNone(t *testing.T) {
	m := embedModel(t)
	c := embedContext(t, m, ContextOptions{Embeddings: true, Pooling: PoolingNone, Sequences: 1})

	vecs, err := c.EmbedBatch(t.Context(), []string{"first text", "second text"})
	if err != nil {
		t.Fatalf("EmbedBatch with PoolingNone: %v", err)
	}
	if len(vecs) != 2 {
		t.Fatalf("EmbedBatch returned %d vectors, want 2", len(vecs))
	}
	for i, vec := range vecs {
		if len(vec) != m.EmbeddingSize() {
			t.Fatalf("vector %d has %d values, model reports %d", i, len(vec), m.EmbeddingSize())
		}
	}
}

func TestEmbedRejectsEmptyText(t *testing.T) {
	m := embedModel(t)
	c := embedContext(t, m, ContextOptions{Embeddings: true, Pooling: PoolingMean})

	if _, err := c.Embed(t.Context(), ""); !errors.Is(err, ErrEmbedding) {
		t.Fatalf("Embed of empty text = %v, want ErrEmbedding", err)
	}
	vecs, err := c.EmbedBatch(t.Context(), nil)
	if err != nil || len(vecs) != 0 {
		t.Fatalf("EmbedBatch of no texts = %v, %v; want no vectors and no error", vecs, err)
	}
}

func TestEmbedRejectsTextAboveTheMicroBatch(t *testing.T) {
	m := embedModel(t)
	c := embedContext(t, m, ContextOptions{Embeddings: true, Pooling: PoolingMean})

	// Long enough to need several micro-batches, which cannot be pooled into one
	// vector without silently covering only the tail.
	long := strings.Repeat("the river runs past the mill ", 200)
	_, err := c.Embed(t.Context(), long)
	if !errors.Is(err, ErrEmbedding) {
		t.Fatalf("Embed of a very long text = %v, want ErrEmbedding", err)
	}
	if !strings.Contains(err.Error(), "micro-batch") {
		t.Fatalf("error %q does not name the micro-batch limit", err)
	}
}

func TestParaphraseIsCloserThanUnrelatedText(t *testing.T) {
	m := embedModel(t)
	texts := []string{
		"the cat is sitting on the mat",
		"a cat sits on the mat",
		"quantum chromodynamics describes the strong nuclear force",
	}
	c := embedContext(t, m, ContextOptions{Embeddings: true, Pooling: PoolingMean, Sequences: uint32(len(texts))})

	vecs, err := c.EmbedBatch(t.Context(), texts)
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	paraphrase := CosineSimilarity(vecs[0], vecs[1])
	unrelated := CosineSimilarity(vecs[0], vecs[2])
	if math.IsNaN(paraphrase) || math.IsNaN(unrelated) {
		t.Fatalf("cosine similarities are NaN: paraphrase %v, unrelated %v", paraphrase, unrelated)
	}
	if paraphrase <= unrelated {
		t.Fatalf("paraphrase similarity %v does not beat unrelated similarity %v", paraphrase, unrelated)
	}
}

func maxAbsDiff(a, b []float32) float64 {
	if len(a) != len(b) {
		return math.Inf(1)
	}
	var worst float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > worst {
			worst = d
		}
	}
	return worst
}
