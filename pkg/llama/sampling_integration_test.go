//go:build integration

package llama

import (
	"context"
	"os"
	"slices"
	"strings"
	"testing"
)

// integrationModel loads the model named by GO_LLAMA_TEST_MODEL on the CPU, so
// the run works on a machine with no usable GPU.
func integrationModel(t *testing.T) *Model {
	t.Helper()
	path := os.Getenv("GO_LLAMA_TEST_MODEL")
	if path == "" {
		t.Skip("GO_LLAMA_TEST_MODEL is not set")
	}
	m, err := LoadModel(path, ModelParams{GPULayers: 0})
	if err != nil {
		t.Fatalf("LoadModel(%s): %v", path, err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

// generateOnce runs one generation in a fresh context, so a reused KV cache
// cannot be the reason two runs match.
func generateOnce(t *testing.T, m *Model, opts CompletionOptions) Result {
	t.Helper()
	c, err := m.NewContext(ContextOptions{ContextSize: 512, Sequences: 1})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer c.Close()
	res, err := c.Generate(context.Background(), "The capital of France is", opts)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Tokens) == 0 {
		t.Fatal("Generate produced no tokens")
	}
	return res
}

// A fixed seed has to reproduce the same tokens, otherwise Seed is not reaching
// llama.cpp's distribution sampler.
func TestIntegrationSamplingSeedIsDeterministic(t *testing.T) {
	m := integrationModel(t)
	opts := CompletionOptions{
		MaxTokens:   24,
		Temperature: 0.8,
		TopK:        40,
		TopP:        0.95,
		MinP:        0.05,
		Seed:        424242,
		AddBOS:      true,
	}

	first := generateOnce(t, m, opts)
	second := generateOnce(t, m, opts)
	if !slices.Equal(first.Tokens, second.Tokens) {
		t.Fatalf("same seed produced different tokens:\n%v\n%v", first.Tokens, second.Tokens)
	}
	if first.Text != second.Text {
		t.Fatalf("same seed produced different text:\n%q\n%q", first.Text, second.Text)
	}
}

func TestIntegrationSamplingGreedyIsDeterministic(t *testing.T) {
	m := integrationModel(t)
	opts := GreedyCompletionOptions(24)

	first := generateOnce(t, m, opts)
	second := generateOnce(t, m, opts)
	if !slices.Equal(first.Tokens, second.Tokens) {
		t.Fatalf("greedy decoding produced different tokens:\n%v\n%v", first.Tokens, second.Tokens)
	}
}

// TopK above the vocabulary size is clamped rather than rejected, which is what
// llama.cpp does when it truncates the candidate list.
func TestIntegrationTopKClampedToVocab(t *testing.T) {
	m := integrationModel(t)
	size := m.Vocab().Size()
	if size <= 0 {
		t.Fatalf("vocabulary size = %d", size)
	}
	if got := clampTopK(int32(size)+1000, size); got != int32(size) {
		t.Fatalf("clampTopK above the vocabulary = %d, want %d", got, size)
	}

	opts := GreedyCompletionOptions(8)
	opts.Temperature = 0.7
	opts.TopK = int32(size) + 1000
	s, err := newSamplerFromOptions(m.Vocab(), opts)
	if err != nil {
		t.Fatalf("newSamplerFromOptions: %v", err)
	}
	defer s.Close()
	if !slices.Contains(s.Stages(), "top-k") {
		t.Fatalf("stages = %v, want a top-k stage", s.Stages())
	}
}

// DRY is the one stage that cannot be built without a model, so its chain
// construction is only reachable here.
func TestIntegrationSamplerDRYStage(t *testing.T) {
	m := integrationModel(t)
	opts := CompletionOptions{
		MaxTokens:           16,
		Temperature:         0.7,
		TopK:                40,
		DryMultiplier:       0.8,
		DryBase:             1.75,
		DryAllowedLength:    2,
		DryPenaltyLastN:     -1,
		DrySequenceBreakers: []string{"\n", ":", "*"},
		Seed:                99,
	}
	s, err := newSamplerFromOptions(m.Vocab(), opts)
	if err != nil {
		t.Fatalf("newSamplerFromOptions: %v", err)
	}
	defer s.Close()
	if !slices.Contains(s.Stages(), "dry") {
		t.Fatalf("stages = %v, want a dry stage", s.Stages())
	}
	// The whole chain has to run, not just build.
	if res := generateOnce(t, m, opts); len(res.Tokens) == 0 {
		t.Fatal("generation with DRY produced no tokens")
	}
}

// A stop sequence spanning several tokens must never reach the caller, which is
// the property the stop matcher exists for.
func TestIntegrationStopSequence(t *testing.T) {
	m := integrationModel(t)
	opts := CompletionOptions{
		MaxTokens:   64,
		Temperature: 0,
		Stop:        []string{"\n\n", "\n#"},
	}
	c, err := m.NewContext(ContextOptions{ContextSize: 512, Sequences: 1})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	defer c.Close()

	res, err := c.Generate(context.Background(), "List three fruits, one per line:", opts)
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	for _, stop := range opts.Stop {
		if strings.Contains(res.Text, stop) {
			t.Fatalf("result text contains the stop %q: %q", stop, res.Text)
		}
	}
}

// The stages added alongside the older ones still have to sample real tokens,
// which a stage count cannot prove.
func TestIntegrationSamplerExtraStagesRun(t *testing.T) {
	m := integrationModel(t)
	base := CompletionOptions{
		MaxTokens:   8,
		Temperature: 0.8,
		Seed:        31337,
		AddBOS:      true,
	}
	cases := map[string]func(*CompletionOptions){
		"xtc": func(o *CompletionOptions) {
			o.XTCProbability = 0.5
			o.XTCThreshold = 0.1
		},
		"top n sigma": func(o *CompletionOptions) { o.TopNSigma = 1.5 },
		"mirostat v2": func(o *CompletionOptions) {
			o.MirostatTau = 5
			o.MirostatEta = 0.1
		},
	}
	for name, apply := range cases {
		t.Run(name, func(t *testing.T) {
			opts := base
			apply(&opts)
			if res := generateOnce(t, m, opts); len(res.Tokens) == 0 {
				t.Fatalf("generation produced no tokens: %v", res.StopReason)
			}
		})
	}
}
