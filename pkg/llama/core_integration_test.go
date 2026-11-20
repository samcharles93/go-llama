//go:build integration

package llama

import (
	"bytes"
	"context"
	"errors"
	"os"
	"strings"
	"testing"
	"time"
)

// These tests exercise the pieces of the core that only exist with a real model:
// the KV cache, context state serialisation and the shutdown ordering between
// Model and Context. They need GO_LLAMA_TEST_MODEL and skip without it.

func coreModelPath(t *testing.T) string {
	t.Helper()
	path := os.Getenv("GO_LLAMA_TEST_MODEL")
	if path == "" {
		t.Skip("GO_LLAMA_TEST_MODEL is not set")
	}
	return path
}

// loadTestModel loads the shared test model on the CPU, so several tests can run
// without competing for the GPU.
func loadCoreModel(t *testing.T) *Model {
	t.Helper()
	m, err := LoadModel(coreModelPath(t), ModelParams{GPULayers: 0, UseMmap: true})
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	t.Cleanup(func() {
		if err := m.Close(); err != nil {
			t.Errorf("model close: %v", err)
		}
	})
	return m
}

func newCoreContext(t *testing.T, m *Model, opts ContextOptions) *Context {
	t.Helper()
	c, err := m.NewContext(opts)
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	t.Cleanup(func() {
		if err := c.Close(); err != nil {
			t.Errorf("context close: %v", err)
		}
	})
	return c
}

func TestCountTokensMatchesTokenize(t *testing.T) {
	m := loadCoreModel(t)
	const text = "token counting should agree with tokenize"
	n, err := m.CountTokens(text, true)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	tokens, err := m.Vocab().Tokenize(text, true)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if n != len(tokens) {
		t.Fatalf("CountTokens = %d, Tokenize gave %d", n, len(tokens))
	}
	if n == 0 {
		t.Fatal("expected at least one token")
	}
}

func TestContextHistoryAndTruncate(t *testing.T) {
	m := loadCoreModel(t)
	c := newCoreContext(t, m, ContextOptions{ContextSize: 256, Threads: 4, ThreadsBatch: 4})

	res, err := c.Generate(context.Background(), "The quick brown fox", GreedyCompletionOptions(6))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	before := c.History()
	if len(before) == 0 {
		t.Fatal("history is empty after generation")
	}
	if len(before) != res.PromptTokens+res.Metrics.GeneratedTokens {
		// The final token is sampled before it is decoded, so a short off-by-one
		// is expected; anything larger means the accounting is wrong.
		if len(before) < res.PromptTokens {
			t.Fatalf("history holds %d tokens, prompt alone was %d", len(before), res.PromptTokens)
		}
	}

	keep := len(before) / 2
	if err := c.Truncate(keep); err != nil {
		t.Fatalf("Truncate: %v", err)
	}
	if got := len(c.History()); got != keep {
		t.Fatalf("history after Truncate = %d, want %d", got, keep)
	}
	if err := c.Truncate(keep); err != nil {
		t.Fatalf("Truncate to the same length: %v", err)
	}
	if err := c.Truncate(-1); err == nil {
		t.Fatal("Truncate(-1) returned nil")
	}

	// A second generation after a truncation must still work, which proves the
	// KV cache shrank with the history rather than being left inconsistent.
	if _, err := c.Generate(context.Background(), "and then", GreedyCompletionOptions(4)); err != nil {
		t.Fatalf("Generate after Truncate: %v", err)
	}
}

func TestSaveStateResumesConversation(t *testing.T) {
	m := loadCoreModel(t)
	const prompt = "Explain what a mutex is in one sentence."

	first := newCoreContext(t, m, ContextOptions{ContextSize: 512, Threads: 4, ThreadsBatch: 4})
	res, err := first.Generate(context.Background(), prompt, GreedyCompletionOptions(12))
	if err != nil {
		t.Fatalf("first Generate: %v", err)
	}
	wantHistory := len(first.History())

	state, err := first.SaveState()
	if err != nil {
		t.Fatalf("SaveState: %v", err)
	}
	if len(state) == 0 {
		t.Fatal("SaveState returned no bytes")
	}

	second := newCoreContext(t, m, ContextOptions{ContextSize: 512, Threads: 4, ThreadsBatch: 4})
	if err := second.LoadState(state); err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	if got := len(second.History()); got != wantHistory {
		t.Fatalf("restored history = %d tokens, want %d", got, wantHistory)
	}

	// Continuing from the restored state must reuse the whole prefix: nothing in
	// the prompt is re-evaluated.
	next := prompt + res.Text + " Why is it needed?"
	resumed, err := second.Generate(context.Background(), next, GreedyCompletionOptions(8))
	if err != nil {
		t.Fatalf("Generate on restored context: %v", err)
	}
	if resumed.ReusedTokens == 0 {
		t.Fatal("restored context reused no tokens, so the state did not carry the KV cache")
	}

	if err := second.LoadState([]byte("not a state")); err == nil {
		t.Fatal("LoadState accepted random bytes")
	}

	var buf bytes.Buffer
	if err := first.SaveStateTo(&buf); err != nil {
		t.Fatalf("SaveStateTo: %v", err)
	}
	third := newCoreContext(t, m, ContextOptions{ContextSize: 512, Threads: 4, ThreadsBatch: 4})
	if err := third.LoadStateFrom(&buf); err != nil {
		t.Fatalf("LoadStateFrom: %v", err)
	}
}

func TestCloseDuringGenerateStopsPromptly(t *testing.T) {
	m := loadCoreModel(t)
	c, err := m.NewContext(ContextOptions{ContextSize: 512, Threads: 4, ThreadsBatch: 4})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}

	done := make(chan error, 1)
	go func() {
		_, err := c.Generate(context.Background(), "Count to one hundred:", CompletionOptions{
			MaxTokens: 400, Temperature: 0.8, TopK: 40, TopP: 0.95, AddBOS: true,
		})
		done <- err
	}()

	time.Sleep(100 * time.Millisecond)
	closed := make(chan error, 1)
	go func() { closed <- c.Close() }()

	select {
	case err := <-closed:
		if err != nil {
			t.Fatalf("Close: %v", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Close blocked while a generation was running")
	}

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("Generate returned nil after its context was closed mid-flight")
		}
		if !errors.Is(err, ErrClosed) && !errors.Is(err, context.Canceled) {
			t.Fatalf("Generate after Close = %v, want ErrClosed", err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("Generate did not return after its context was closed")
	}
}

func TestModelCloseRefusesLiveContexts(t *testing.T) {
	m, err := LoadModel(coreModelPath(t), ModelParams{GPULayers: 0, UseMmap: true})
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}
	c, err := m.NewContext(ContextOptions{ContextSize: 128, Threads: 2, ThreadsBatch: 2})
	if err != nil {
		t.Fatalf("NewContext: %v", err)
	}
	if err := m.Close(); err == nil {
		t.Fatal("closing a model with an open context returned nil")
	}
	if err := c.Close(); err != nil {
		t.Fatalf("context close: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("model close after the context closed: %v", err)
	}
	if _, err := m.NewContext(); !errors.Is(err, ErrClosed) {
		t.Fatalf("NewContext after Close = %v, want ErrClosed", err)
	}
}

func TestKVReuseOnExactExtension(t *testing.T) {
	m := loadCoreModel(t)
	c := newCoreContext(t, m, ContextOptions{ContextSize: 1024, Threads: 4, ThreadsBatch: 4})
	ctx := context.Background()

	prompt, err := m.Vocab().Tokenize("The quick brown fox", true)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if _, err := c.GenerateTokens(ctx, prompt, GreedyCompletionOptions(4)); err != nil {
		t.Fatalf("first GenerateTokens: %v", err)
	}
	history := c.History()
	if len(history) == 0 {
		t.Fatal("no history after generation")
	}

	// Extending the cached tokens exactly is the case every caller gets for free:
	// nothing in the prefix is evaluated again.
	extra, err := m.Vocab().Tokenize(" jumps", false)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	next := append(append([]Token{}, history...), extra...)

	res, err := c.GenerateTokens(ctx, next, GreedyCompletionOptions(2))
	if err != nil {
		t.Fatalf("second GenerateTokens: %v", err)
	}
	if res.PromptTokens != len(next) {
		t.Fatalf("PromptTokens = %d, want %d", res.PromptTokens, len(next))
	}
	if res.ReusedTokens != len(history) {
		t.Fatalf("ReusedTokens = %d, want %d (the whole cached prefix)", res.ReusedTokens, len(history))
	}
}

func TestHybridModelFallsBackToFullRecompute(t *testing.T) {
	m := loadCoreModel(t)
	if !m.IsHybrid() && !m.IsRecurrent() {
		t.Skip("model keeps an attention cache, so it can trim its state")
	}
	c := newCoreContext(t, m, ContextOptions{ContextSize: 512, Threads: 4, ThreadsBatch: 4})

	prompt, err := m.Vocab().Tokenize("The quick brown fox", true)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if _, err := c.GenerateTokens(context.Background(), prompt, GreedyCompletionOptions(4)); err != nil {
		t.Fatalf("first GenerateTokens: %v", err)
	}
	history := c.History()
	if len(history) < 3 {
		t.Fatalf("history too short to truncate: %d", len(history))
	}

	// A prompt that diverges before the end of the cache cannot reuse anything:
	// llama.cpp refuses to drop the tail of a recurrent state, so the context has
	// to re-evaluate the prompt. ReusedTokens reports that honestly instead of
	// leaving a stale state behind.
	diverging := append([]Token{}, history[:len(history)-2]...)
	diverging = append(diverging, history[len(history)-1])

	res, err := c.GenerateTokens(context.Background(), diverging, GreedyCompletionOptions(2))
	if err != nil {
		t.Fatalf("diverging GenerateTokens: %v", err)
	}
	if res.ReusedTokens != 0 {
		t.Fatalf("ReusedTokens = %d, want 0 for a model that cannot trim its state", res.ReusedTokens)
	}
}

func TestFitMessagesDropsOldestTurnsAndKeepsSystem(t *testing.T) {
	m := loadCoreModel(t)

	system := SystemMessage("Answer in one word.")
	msgs := []ChatMessage{system}
	for range 12 {
		msgs = append(msgs, UserMessage(strings.Repeat("padding words for the context budget ", 6)))
		msgs = append(msgs, AssistantMessage("ok"))
	}
	msgs = append(msgs, UserMessage("last question?"))

	const budget = 128
	fitted, err := m.FitMessages(msgs, budget)
	if err != nil {
		t.Fatalf("FitMessages: %v", err)
	}
	if len(fitted) >= len(msgs) {
		t.Fatalf("FitMessages returned %d messages from %d, expected truncation", len(fitted), len(msgs))
	}
	if fitted[0].Role != "system" || fitted[0].Content != system.Content {
		t.Fatalf("system message was dropped or reordered: %+v", fitted[0])
	}
	if last := fitted[len(fitted)-1]; last.Content != "last question?" {
		t.Fatalf("newest turn was dropped: %+v", last)
	}

	prompt, err := m.ApplyChatTemplate(fitted, true)
	if err != nil {
		t.Fatalf("ApplyChatTemplate: %v", err)
	}
	n, err := m.CountTokens(prompt, true)
	if err != nil {
		t.Fatalf("CountTokens: %v", err)
	}
	if n > budget {
		t.Fatalf("fitted prompt is %d tokens, budget was %d", n, budget)
	}

	if _, err := m.FitMessages(msgs, 4); err == nil {
		t.Fatal("FitMessages with an impossible budget returned no error")
	}
	if _, err := m.FitMessages(msgs, 0); err == nil {
		t.Fatal("FitMessages with a zero budget returned no error")
	}
}

func TestRestoredStateReproducesGreedyReply(t *testing.T) {
	m := loadCoreModel(t)
	const prompt = "Name three colours."
	opts := GreedyCompletionOptions(6)

	first := newCoreContext(t, m, ContextOptions{ContextSize: 512, Threads: 4, ThreadsBatch: 4})
	res1, err := first.Generate(context.Background(), prompt, opts)
	if err != nil {
		t.Fatalf("first Generate: %v", err)
	}
	state, err := first.SaveState()
	if err != nil {
		t.Fatalf("SaveState: %v", err)
	}

	// The saved state carries the KV cache but not the output logits, so a
	// restored context that reuses its cache without decoding would sample from
	// whatever the buffer held. Greedy decoding makes the divergence obvious.
	second := newCoreContext(t, m, ContextOptions{ContextSize: 512, Threads: 4, ThreadsBatch: 4})
	if err := second.LoadState(state); err != nil {
		t.Fatalf("LoadState: %v", err)
	}
	res2, err := second.Generate(context.Background(), prompt, opts)
	if err != nil {
		t.Fatalf("Generate after LoadState: %v", err)
	}
	if res1.Text != res2.Text {
		t.Fatalf("restored context produced %q, original produced %q", res2.Text, res1.Text)
	}
}

func TestGrammarConstrainsOutput(t *testing.T) {
	m := loadCoreModel(t)
	c := newCoreContext(t, m, ContextOptions{ContextSize: 256, Threads: 4, ThreadsBatch: 4})

	opts := GreedyCompletionOptions(8)
	// Four lowercase letters is satisfiable by any BPE vocabulary, so the test does
	// not depend on how a particular model spells "yes".
	opts.Grammar = `root ::= [a-z]{1,4}`

	res, err := c.Generate(context.Background(), "Say any word.", opts)
	if err != nil {
		t.Fatalf("Generate with a grammar: %v", err)
	}
	if res.Text == "" {
		t.Fatal("grammar-constrained generation produced no text")
	}
	for _, r := range res.Text {
		if r < 'a' || r > 'z' {
			t.Fatalf("grammar allowed %q in %q, only lowercase letters were legal", r, res.Text)
		}
	}
	if len(res.Text) > 4 {
		t.Fatalf("generated %q, the grammar caps it at four letters", res.Text)
	}

	// A malformed grammar is rejected before any decode.
	bad := GreedyCompletionOptions(4)
	bad.Grammar = `root ::= [unclosed`
	if _, err := c.Generate(context.Background(), "x", bad); !errors.Is(err, ErrGrammar) {
		t.Fatalf("Generate with a broken grammar = %v, want ErrGrammar", err)
	}
}
