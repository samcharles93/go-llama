//go:build integration

package llama

import (
	"os"
	"testing"
)

func TestReproEmbedSeqs(t *testing.T) {
	m, err := LoadModel(os.Getenv("GO_LLAMA_TEST_EMBED_MODEL"), ModelParams{GPULayers: 0})
	if err != nil {
		t.Fatal(err)
	}
	defer m.Close()
	for _, cfg := range []ContextOptions{
		{Embeddings: true, Pooling: PoolingMean, Sequences: 1},
		{Embeddings: true, Pooling: PoolingMean, Sequences: 2},
	} {
		t.Logf("sequences=%d", cfg.Sequences)
		c, err := m.NewContext(cfg)
		if err != nil {
			t.Fatal(err)
		}
		for _, s := range []string{"the cat sat on the mat", "a dog barked at the postman"} {
			v, err := c.Embed(t.Context(), s)
			t.Logf("  %q -> %d %v", s, len(v), err)
		}
		c.Close()
	}
}
