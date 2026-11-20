package llama

import (
	"context"
	"errors"
	"math"
	"os"
	"sync"
	"testing"
	"time"
)

// Environment variables the model-backed tests and benchmarks read. The
// integration suite documents the same two names in AGENTS.md.
const (
	envTestModel      = "GO_LLAMA_TEST_MODEL"
	envTestEmbedModel = "GO_LLAMA_TEST_EMBED_MODEL"
	envTestGPU        = "GO_LLAMA_TEST_GPU"
)

// cpuModelParams keeps the default suite on the CPU so a run without a GPU
// produces the same numbers everywhere. One integration test exercises the
// GPU path instead.
func cpuModelParams() ModelParams { return ModelParams{GPULayers: 0, UseMmap: true} }

// requireModel loads the GGUF file named by GO_LLAMA_TEST_MODEL and closes it
// when the test or benchmark ends. It skips when the variable is unset, which
// keeps `go test ./pkg/llama/` green on a machine with no model.
func requireModel(tb testing.TB) *Model {
	tb.Helper()
	path := os.Getenv(envTestModel)
	if path == "" {
		tb.Skipf("%s is unset; run scripts/fetch-test-models.sh and export it", envTestModel)
	}
	m, err := LoadModel(path, cpuModelParams())
	if err != nil {
		tb.Fatalf("LoadModel(%s): %v", path, err)
	}
	tb.Cleanup(func() {
		if err := m.Close(); err != nil {
			tb.Errorf("Close(%s): %v", path, err)
		}
	})
	return m
}

// requireEmbedModel loads the embedding model named by
// GO_LLAMA_TEST_EMBED_MODEL and skips when the variable is unset.
func requireEmbedModel(tb testing.TB) *Model {
	tb.Helper()
	path := os.Getenv(envTestEmbedModel)
	if path == "" {
		tb.Skipf("%s is unset; run scripts/fetch-test-models.sh and export it", envTestEmbedModel)
	}
	m, err := LoadModel(path, cpuModelParams())
	if err != nil {
		tb.Fatalf("LoadModel(%s): %v", path, err)
	}
	tb.Cleanup(func() {
		if err := m.Close(); err != nil {
			tb.Errorf("Close(%s): %v", path, err)
		}
	})
	return m
}

// chatPrompt renders one user turn through the model's own chat template. An
// instruct model given a raw prompt frequently answers with an immediate
// end-of-generation token, which makes a generation test measure nothing.
func chatPrompt(tb testing.TB, m *Model, user string) string {
	tb.Helper()
	prompt, err := m.ApplyChatTemplate([]ChatMessage{
		SystemMessage("You are a terse assistant. Answer in one short sentence."),
		UserMessage(user),
	}, true)
	if err != nil {
		tb.Fatalf("ApplyChatTemplate: %v", err)
	}
	return prompt
}

// newTestContext creates a context bound to the test's lifetime.
func newTestContext(tb testing.TB, m *Model, opts ...ContextOptions) *Context {
	tb.Helper()
	c, err := m.NewContext(opts...)
	if err != nil {
		tb.Fatalf("NewContext: %v", err)
	}
	tb.Cleanup(func() {
		if err := c.Close(); err != nil {
			tb.Errorf("Close context: %v", err)
		}
	})
	return c
}

// TestConcurrentGenerateSharesOneModel runs several decodes at once on separate
// contexts that share one loaded model. Under -race this is the check that the
// Model is read-only during generation and that no context touches another's
// KV cache.
func TestConcurrentGenerateSharesOneModel(t *testing.T) {
	m := requireModel(t)

	const workers = 4
	prompt := "Name one colour, then stop."
	opts := GreedyCompletionOptions(8)

	contexts := make([]*Context, workers)
	for i := range contexts {
		contexts[i] = newTestContext(t, m)
	}

	var wg sync.WaitGroup
	results := make([]Result, workers)
	errs := make([]error, workers)
	for i := range workers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			results[i], errs[i] = contexts[i].Generate(context.Background(), prompt, opts)
		}()
	}
	wg.Wait()

	for i := range workers {
		if errs[i] != nil {
			t.Fatalf("worker %d: Generate: %v", i, errs[i])
		}
		if len(results[i].Tokens) == 0 {
			t.Fatalf("worker %d: generated no tokens", i)
		}
		if results[i].PromptTokens == 0 {
			t.Fatalf("worker %d: PromptTokens = 0", i)
		}
	}
	// Greedy decoding of the same prompt on separate caches must agree, so a
	// difference means the workers interfered with each other.
	first := results[0].Tokens
	for i := 1; i < workers; i++ {
		if got := results[i].Tokens; !equalTokens(got, first) {
			t.Errorf("worker %d tokens = %v, want %v (same as worker 0)", i, got, first)
		}
	}
}

// TestMetricsConcurrentWithGeneration hammers Metrics and ResetMetrics from
// another goroutine while a generation holds the context lock. Both take the
// lock, so the race detector should stay quiet and generation should still
// finish normally.
func TestMetricsConcurrentWithGeneration(t *testing.T) {
	m := requireModel(t)
	c := newTestContext(t, m)

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stop:
				return
			default:
				_ = c.Metrics()
				c.ResetMetrics()
				time.Sleep(time.Millisecond)
			}
		}
	}()

	res, err := c.Generate(context.Background(), "Say hello.", GreedyCompletionOptions(16))
	close(stop)
	wg.Wait()

	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Tokens) == 0 {
		t.Fatal("generated no tokens")
	}
}

func equalTokens(a, b []Token) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// TestZeroContextRejectsUse covers the zero value: a Context that never came
// from NewContext has no handle, and every call must report ErrClosed instead
// of dereferencing it.
func TestZeroContextRejectsUse(t *testing.T) {
	c := &Context{}

	if _, err := c.Generate(context.Background(), "hi", GreedyCompletionOptions(4)); !errors.Is(err, ErrClosed) {
		t.Fatalf("Generate on zero Context = %v, want ErrClosed", err)
	}
	yielded := 0
	for piece, err := range c.Stream(context.Background(), "hi", GreedyCompletionOptions(4)) {
		yielded++
		if !errors.Is(err, ErrClosed) {
			t.Fatalf("Stream on zero Context = %v, want ErrClosed", err)
		}
		if piece != "" {
			t.Fatalf("Stream on zero Context yielded %q, want no pieces", piece)
		}
	}
	if yielded != 1 {
		t.Fatalf("Stream on zero Context yielded %d results, want 1 error", yielded)
	}
	if got := c.Metrics(); got != (Metrics{}) {
		t.Fatalf("Metrics on zero Context = %+v, want zero", got)
	}
	if got := c.TokenCount(); got != 0 {
		t.Fatalf("TokenCount on zero Context = %d, want 0", got)
	}
}

// TestClosedContextRejectsUse checks the same contract on a context that was
// opened and then closed, which is the case a server hits during shutdown.
func TestClosedContextRejectsUse(t *testing.T) {
	m := requireModel(t)
	c := newTestContext(t, m, ContextOptions{Embeddings: true})
	if err := c.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if _, err := c.Generate(context.Background(), "hi", GreedyCompletionOptions(4)); !errors.Is(err, ErrClosed) {
		t.Fatalf("Generate after Close = %v, want ErrClosed", err)
	}
	for piece, err := range c.Stream(context.Background(), "hi", GreedyCompletionOptions(4)) {
		if !errors.Is(err, ErrClosed) {
			t.Fatalf("Stream after Close = %v (piece %q), want ErrClosed", err, piece)
		}
	}
	if _, err := c.Embed(context.Background(), "hi"); !errors.Is(err, ErrClosed) {
		t.Fatalf("Embed after Close = %v, want ErrClosed", err)
	}
	if err := c.Reset(); !errors.Is(err, ErrClosed) {
		t.Fatalf("Reset after Close = %v, want ErrClosed", err)
	}
	// Metrics and ResetMetrics have no error return, so the closed context is
	// reported as zero metrics and a no-op reset rather than an error. A caller
	// that needs the distinction has to check its own closed flag.
	if got := c.Metrics(); got != (Metrics{}) {
		t.Fatalf("Metrics after Close = %+v, want zero", got)
	}
	c.ResetMetrics()
}

// Shared vector helpers for the model-backed tests and benchmarks.

func l2norm(v []float32) float64 {
	var sum float64
	for _, f := range v {
		sum += float64(f) * float64(f)
	}
	return math.Sqrt(sum)
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / math.Sqrt(na*nb)
}

func equalVectors(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
