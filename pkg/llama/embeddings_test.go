package llama

import (
	"errors"
	"math"
	"testing"
)

// These tests need no model file: they cover the vector helpers and the checks
// that run before llama.cpp is touched. Model-backed cases live in
// embeddings_integration_test.go.

func TestNormalize(t *testing.T) {
	t.Run("scales to unit length", func(t *testing.T) {
		got := Normalize([]float32{3, 4})
		want := []float32{0.6, 0.8}
		for i := range want {
			if math.Abs(float64(got[i]-want[i])) > 1e-6 {
				t.Fatalf("Normalize({3,4}) = %v, want %v", got, want)
			}
		}
		if n := vecLength(got); math.Abs(n-1) > 1e-6 {
			t.Fatalf("length after Normalize = %v, want 1", n)
		}
	})

	t.Run("leaves an empty vector alone", func(t *testing.T) {
		if got := Normalize(nil); len(got) != 0 {
			t.Fatalf("Normalize(nil) = %v, want empty", got)
		}
		if got := Normalize([]float32{}); len(got) != 0 {
			t.Fatalf("Normalize(empty) = %v, want empty", got)
		}
	})

	t.Run("leaves a zero vector alone", func(t *testing.T) {
		vec := []float32{0, 0, 0}
		got := Normalize(vec)
		for i, v := range got {
			if v != 0 {
				t.Fatalf("Normalize of a zero vector changed index %d to %v", i, v)
			}
		}
	})

	t.Run("does not manufacture values from NaN", func(t *testing.T) {
		vec := []float32{float32(math.NaN()), 1}
		got := Normalize(vec)
		if !math.IsNaN(float64(got[0])) || got[1] != 1 {
			t.Fatalf("Normalize of a NaN vector = %v, want the input unchanged", got)
		}
	})
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float64
	}{
		{name: "identical", a: []float32{1, 2, 3}, b: []float32{1, 2, 3}, want: 1},
		{name: "orthogonal", a: []float32{1, 0}, b: []float32{0, 1}, want: 0},
		{name: "opposite", a: []float32{1, 2}, b: []float32{-1, -2}, want: -1},
		{name: "scale invariant", a: []float32{3, 4}, b: []float32{30, 40}, want: 1},
		{name: "empty", a: nil, b: nil, want: math.NaN()},
		{name: "one empty", a: []float32{1}, b: nil, want: math.NaN()},
		{name: "length mismatch", a: []float32{1, 2}, b: []float32{1, 2, 3}, want: math.NaN()},
		{name: "zero vector", a: []float32{0, 0}, b: []float32{1, 1}, want: math.NaN()},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := CosineSimilarity(tc.a, tc.b)
			if math.IsNaN(tc.want) {
				if !math.IsNaN(got) {
					t.Fatalf("CosineSimilarity(%v, %v) = %v, want NaN", tc.a, tc.b, got)
				}
				return
			}
			if math.Abs(got-tc.want) > 1e-6 {
				t.Fatalf("CosineSimilarity(%v, %v) = %v, want %v", tc.a, tc.b, got, tc.want)
			}
		})
	}
}

func TestEmbedBatchRequiresEmbeddings(t *testing.T) {
	c := &Context{}
	if _, err := c.EmbedBatch(t.Context(), []string{"hi"}); !errors.Is(err, ErrEmbedding) {
		t.Fatalf("EmbedBatch without Embeddings = %v, want ErrEmbedding", err)
	}
	if _, err := c.Embed(t.Context(), "hi"); !errors.Is(err, ErrEmbedding) {
		t.Fatalf("Embed without Embeddings = %v, want ErrEmbedding", err)
	}
}

func TestEmbedBatchOnClosedContext(t *testing.T) {
	c := &Context{}
	c.closed.Store(true)
	if _, err := c.EmbedBatch(t.Context(), nil); !errors.Is(err, ErrClosed) {
		t.Fatalf("EmbedBatch on a closed context = %v, want ErrClosed", err)
	}
}

func vecLength(vec []float32) float64 {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	return math.Sqrt(sum)
}
