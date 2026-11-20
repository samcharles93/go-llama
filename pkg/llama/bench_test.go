package llama

import (
	"context"
	"testing"
)

// The benchmarks report throughput in tokens per second from llama.cpp's own
// perf counters. Each iteration does a fixed amount of work: a prompt of
// benchPromptTokens tokens to evaluate, or benchGenerationTokens tokens to
// decode. That makes -benchtime=1x meaningful, and the iteration count is a
// repeat count rather than a wall-clock target.
const (
	benchPromptTokens     = 512
	benchGenerationTokens = 32
	benchEmbedText        = "The quick brown fox jumps over the lazy dog."
)

// benchPrompt grows a prompt to at least tokens tokens. Tokenizing a paragraph
// and counting is the only way to know the work is the same on every machine,
// since the tokenizer is part of the model.
func benchPrompt(tb testing.TB, v *Vocab, tokens int) (string, int) {
	tb.Helper()
	sentence := "The quick brown fox jumps over the lazy dog. "
	prompt := sentence
	for {
		n, err := v.Tokenize(prompt, true)
		if err != nil {
			tb.Fatalf("Tokenize: %v", err)
		}
		if len(n) >= tokens {
			return prompt, len(n)
		}
		prompt += sentence
	}
}

// BenchmarkPromptEval measures prompt processing: the 512 token prompt is
// decoded from an empty cache on every iteration.
func BenchmarkPromptEval(b *testing.B) {
	m := requireModel(b)
	prompt, promptTokens := benchPrompt(b, m.Vocab(), benchPromptTokens)
	c := newTestContext(b, m)

	opts := GreedyCompletionOptions(1)
	opts.IgnoreEOS = true

	b.ResetTimer()
	for b.Loop() {
		if err := c.Reset(); err != nil {
			b.Fatalf("Reset: %v", err)
		}
		res, err := c.Generate(context.Background(), prompt, opts)
		if err != nil {
			b.Fatalf("Generate: %v", err)
		}
		if res.ReusedTokens != 0 {
			b.Fatalf("reused %d prompt tokens, so the whole prompt was not evaluated", res.ReusedTokens)
		}
		if res.Metrics.PromptTokens != promptTokens {
			b.Fatalf("evaluated %d prompt tokens, want %d", res.Metrics.PromptTokens, promptTokens)
		}
		b.ReportMetric(res.Metrics.PromptTokensPerSecond(), "tok/s")
	}
}

// BenchmarkGeneration measures decoding: exactly benchGenerationTokens tokens
// per iteration, with end-of-generation ignored so the count does not depend on
// what the model chose to say.
func BenchmarkGeneration(b *testing.B) {
	m := requireModel(b)
	c := newTestContext(b, m)
	prompt := chatPrompt(b, m, "Tell me a story about a small robot.")

	opts := GreedyCompletionOptions(benchGenerationTokens)
	opts.IgnoreEOS = true

	b.ResetTimer()
	for b.Loop() {
		if err := c.Reset(); err != nil {
			b.Fatalf("Reset: %v", err)
		}
		res, err := c.Generate(context.Background(), prompt, opts)
		if err != nil {
			b.Fatalf("Generate: %v", err)
		}
		if len(res.Tokens) != benchGenerationTokens {
			b.Fatalf("generated %d tokens, want %d", len(res.Tokens), benchGenerationTokens)
		}
		b.ReportMetric(res.Metrics.TokensPerSecond(), "tok/s")
	}
}

// BenchmarkEmbed measures the embedding path: tokenize, decode the whole input
// and pool it. PoolingMean is the mode the small test embedding models are
// trained with.
func BenchmarkEmbed(b *testing.B) {
	m := requireEmbedModel(b)
	c := newTestContext(b, m, ContextOptions{Embeddings: true, Pooling: PoolingMean})
	tokens, err := m.Vocab().Tokenize(benchEmbedText, true)
	if err != nil {
		b.Fatalf("Tokenize: %v", err)
	}
	want := m.EmbeddingSize()

	embedded := 0
	b.ResetTimer()
	for b.Loop() {
		vec, err := c.Embed(context.Background(), benchEmbedText)
		if err != nil {
			b.Fatalf("Embed: %v", err)
		}
		if len(vec) != want {
			b.Fatalf("embedding has %d dimensions, want %d", len(vec), want)
		}
		if l2norm(vec) == 0 {
			b.Fatalf("embedding is all zeros; the context needs the pooling mode the model was trained with")
		}
		embedded += len(tokens)
	}
	elapsed := b.Elapsed().Seconds()
	b.ReportMetric(float64(embedded)/elapsed, "tok/s")
	b.ReportMetric(float64(b.N)/elapsed, "emb/s")
}
