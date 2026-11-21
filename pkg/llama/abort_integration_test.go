//go:build integration

package llama

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

// TestZeroThreadsUsesDefaultThreads checks that an unset thread count reaches
// llama.cpp as DefaultThreads rather than llama.cpp's fixed default of 4, and
// that an explicit count is passed through untouched.
func TestZeroThreadsUsesDefaultThreads(t *testing.T) {
	m := loadCoreModel(t)

	c := newCoreContext(t, m, ContextOptions{ContextSize: 128})
	gen, batch := c.Threads()
	if gen != DefaultThreads() || batch != DefaultThreads() {
		t.Fatalf("zero Threads gave %d/%d, want DefaultThreads %d", gen, batch, DefaultThreads())
	}

	c2 := newCoreContext(t, m, ContextOptions{ContextSize: 128, Threads: 3, ThreadsBatch: 2})
	if gen, batch := c2.Threads(); gen != 3 || batch != 2 {
		t.Fatalf("explicit Threads gave %d/%d, want 3/2", gen, batch)
	}
}

// TestCancelDuringPromptEvaluation cancels while a long prompt is being
// evaluated. With one large batch the abort flag has to interrupt llama_decode
// mid-graph; with small batches the check between chunks does it. Either way
// the call must return promptly with context.Canceled, and the KV cache must
// still match the history so the next generation works.
func TestCancelDuringPromptEvaluation(t *testing.T) {
	m := loadCoreModel(t)
	prompt := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 300)
	tokens, err := m.Vocab().Tokenize(prompt, true)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}

	for _, batchSize := range []uint32{64, 4096} {
		t.Run(map[uint32]string{64: "chunked", 4096: "single-decode"}[batchSize], func(t *testing.T) {
			c := newCoreContext(t, m, ContextOptions{
				ContextSize: 4096,
				BatchSize:   batchSize,
				UBatchSize:  min(batchSize, 512),
			})

			// Time an uncancelled run first, so the bound below is relative to
			// this machine instead of a guess.
			start := time.Now()
			if _, err := c.GenerateTokens(context.Background(), tokens, GreedyCompletionOptions(1)); err != nil {
				t.Fatalf("baseline Generate: %v", err)
			}
			full := time.Since(start)
			if err := c.Reset(); err != nil {
				t.Fatal(err)
			}

			ctx, cancel := context.WithCancel(context.Background())
			time.AfterFunc(full/10, cancel)
			start = time.Now()
			res, err := c.GenerateTokens(ctx, tokens, GreedyCompletionOptions(1))
			elapsed := time.Since(start)
			cancel()

			if !errors.Is(err, context.Canceled) {
				t.Fatalf("Generate = %v, want context.Canceled (full run took %v)", err, full)
			}
			if res.StopReason != StopReasonCancelled {
				t.Fatalf("StopReason = %v, want %v", res.StopReason, StopReasonCancelled)
			}
			if elapsed > full/2 {
				t.Fatalf("cancelled run took %v of a %v prompt evaluation; cancellation did not interrupt it", elapsed, full)
			}
			t.Logf("full prompt %v, cancelled after %v", full, elapsed)

			// The cache must hold exactly what the history says, so the retry
			// reuses only real cached tokens and still produces a token.
			kept := c.TokenCount()
			if kept >= len(tokens) {
				t.Fatalf("history holds %d of %d tokens after cancellation", kept, len(tokens))
			}
			retry, err := c.GenerateTokens(context.Background(), tokens, GreedyCompletionOptions(1))
			if err != nil {
				t.Fatalf("Generate after cancellation: %v", err)
			}
			if retry.ReusedTokens != kept {
				t.Fatalf("retry reused %d tokens, want the %d left in history", retry.ReusedTokens, kept)
			}
			if len(retry.Tokens) != 1 {
				t.Fatalf("retry produced %d tokens, want 1", len(retry.Tokens))
			}
		})
	}
}
