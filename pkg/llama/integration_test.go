//go:build integration

package llama

import (
	"context"
	"errors"
	"fmt"
	"math"
	"os"
	"strings"
	"testing"
	"time"
)

// TestTokenizeRoundTrip checks that text survives tokenization in both
// directions. Byte-level tokenizers split multi-byte runes, so joining the
// pieces back together is the assertion that catches a broken piece buffer.
func TestTokenizeRoundTrip(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	cases := []struct {
		name string
		text string
	}{
		{"sentence", "The quick brown fox jumps over the lazy dog."},
		{"leading space", " hello there"},
		{"unicode", "Ünïcodé vowels and 日本語 text"},
		{"whitespace", "line one\nline two\ttabbed  twice"},
		{"long", strings.Repeat("A story about a small robot. ", 40)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tokens, err := v.Tokenize(tc.text, false)
			if err != nil {
				t.Fatalf("Tokenize(%q): %v", tc.text, err)
			}
			if len(tokens) == 0 {
				t.Fatalf("Tokenize(%q) returned no tokens", tc.text)
			}
			var pieces strings.Builder
			for _, tok := range tokens {
				piece, err := v.TokenToPiece(tok)
				if err != nil {
					t.Fatalf("TokenToPiece(%d) of %q: %v", tok, tc.text, err)
				}
				pieces.WriteString(piece)
			}
			if got := pieces.String(); got != tc.text {
				t.Fatalf("pieces of %q join to %q", tc.text, got)
			}
			back, err := v.Detokenize(tokens)
			if err != nil {
				t.Fatalf("Detokenize(%v): %v", tokens, err)
			}
			if back != tc.text {
				t.Fatalf("Detokenize(Tokenize(%q)) = %q", tc.text, back)
			}

			// llama.cpp only prepends the beginning-of-sequence token for
			// tokenizers configured to add one, so the count grows for an
			// instruct model and stays the same for a tokenizer like Qwen's.
			withBOS, err := v.Tokenize(tc.text, true)
			if err != nil {
				t.Fatalf("Tokenize(%q, addSpecial=true): %v", tc.text, err)
			}
			if len(withBOS) < len(tokens) {
				t.Fatalf("addSpecial token count = %d, want at least the plain count %d", len(withBOS), len(tokens))
			}
			if bos := v.BOS(); bos >= 0 && len(withBOS) == len(tokens)+1 && withBOS[0] != bos {
				t.Fatalf("first token with addSpecial = %d, want BOS %d", withBOS[0], bos)
			}
			if stripped, err := v.Detokenize(withBOS); err != nil {
				t.Fatalf("Detokenize(%v): %v", withBOS, err)
			} else if stripped != tc.text {
				t.Fatalf("Detokenize with addSpecial = %q, want %q", stripped, tc.text)
			}
		})
	}
}

// TestTokenizeEmpty covers the empty prompt, which llama.cpp reports as zero
// tokens rather than an error.
func TestTokenizeEmpty(t *testing.T) {
	m := requireModel(t)
	v := m.Vocab()

	tokens, err := v.Tokenize("", false)
	if err != nil {
		t.Fatalf("Tokenize(\"\"): %v", err)
	}
	if len(tokens) != 0 {
		t.Fatalf("Tokenize(\"\") = %v, want no tokens", tokens)
	}
	back, err := v.Detokenize(nil)
	if err != nil {
		t.Fatalf("Detokenize(nil): %v", err)
	}
	if back != "" {
		t.Fatalf("Detokenize(nil) = %q, want \"\"", back)
	}
	c := newTestContext(t, m)
	if _, err := c.Generate(context.Background(), "", GreedyCompletionOptions(4)); !errors.Is(err, ErrDecode) {
		t.Fatalf("Generate(\"\") = %v, want ErrDecode for an empty prompt", err)
	}
}

// TestGreedyDecodingIsDeterministic runs the same prompt three times: twice on
// one context with a reset between, and once on a second context. Temperature 0
// must give byte-identical output every time.
func TestGreedyDecodingIsDeterministic(t *testing.T) {
	m := requireModel(t)
	prompt := chatPrompt(t, m, "Name one colour, then stop.")
	opts := GreedyCompletionOptions(16)

	first := newTestContext(t, m)
	second := newTestContext(t, m)

	runA, err := first.Generate(context.Background(), prompt, opts)
	if err != nil {
		t.Fatalf("run A: %v", err)
	}
	if len(runA.Tokens) == 0 {
		t.Fatal("run A generated no tokens")
	}
	if runA.StopReason == StopReasonContextFull {
		t.Fatalf("run A stopped with %v on a full-size context", runA.StopReason)
	}

	if err := first.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	runB, err := first.Generate(context.Background(), prompt, opts)
	if err != nil {
		t.Fatalf("run B: %v", err)
	}
	runC, err := second.Generate(context.Background(), prompt, opts)
	if err != nil {
		t.Fatalf("run C: %v", err)
	}

	if !equalTokens(runB.Tokens, runA.Tokens) {
		t.Errorf("run B tokens = %v, want %v", runB.Tokens, runA.Tokens)
	}
	if !equalTokens(runC.Tokens, runA.Tokens) {
		t.Errorf("run C tokens on a fresh context = %v, want %v", runC.Tokens, runA.Tokens)
	}
	if runB.Text != runA.Text {
		t.Errorf("run B text = %q, want %q", runB.Text, runA.Text)
	}
	if runC.Text != runA.Text {
		t.Errorf("run C text = %q, want %q", runC.Text, runA.Text)
	}
	t.Logf("greedy reply: %q (stop %v)", runA.Text, runA.StopReason)
}

// TestMaxTokensRespected checks the cap, including the boundary where the model
// would otherwise keep going.
func TestMaxTokensRespected(t *testing.T) {
	m := requireModel(t)
	prompt := chatPrompt(t, m, "Count slowly from one to twenty.")

	for _, maxTokens := range []int{1, 5, 20} {
		t.Run(fmt.Sprintf("max_%d", maxTokens), func(t *testing.T) {
			c := newTestContext(t, m)
			res, err := c.Generate(context.Background(), prompt, GreedyCompletionOptions(maxTokens))
			if err != nil {
				t.Fatalf("Generate: %v", err)
			}
			if len(res.Tokens) > maxTokens {
				t.Fatalf("generated %d tokens, want at most MaxTokens=%d", len(res.Tokens), maxTokens)
			}
			if res.StopReason == StopReasonMaxTokens && len(res.Tokens) != maxTokens {
				t.Fatalf("StopReasonMaxTokens with %d tokens, want exactly %d", len(res.Tokens), maxTokens)
			}
			if res.StopReason == StopReasonContextFull {
				t.Fatalf("StopReasonContextFull with MaxTokens=%d on a full-size context", maxTokens)
			}
		})
	}
}

// TestStopSequenceHonoured first generates a free-running reply, then repeats
// the same prompt with the tail of that reply as a stop sequence. Greedy
// decoding makes the second run reproduce the first, so the stop must land at
// the same place and never appear in the text.
func TestStopSequenceHonoured(t *testing.T) {
	m := requireModel(t)
	prompt := chatPrompt(t, m, "Describe the colour blue in one sentence.")
	opts := GreedyCompletionOptions(64)

	c := newTestContext(t, m)
	full, err := c.Generate(context.Background(), prompt, opts)
	if err != nil {
		t.Fatalf("free-running Generate: %v", err)
	}
	if len(full.Text) < 12 {
		t.Fatalf("free-running text %q is too short to pick a stop sequence from", full.Text)
	}
	stop := full.Text[len(full.Text)/2:]
	if strings.TrimSpace(stop) == "" {
		t.Fatalf("stop sequence %q from %q is blank", stop, full.Text)
	}

	if err := c.Reset(); err != nil {
		t.Fatalf("Reset: %v", err)
	}
	opts.Stop = []string{stop}
	res, err := c.Generate(context.Background(), prompt, opts)
	if err != nil {
		t.Fatalf("Generate with Stop=%q: %v", stop, err)
	}
	if res.StopReason != StopReasonStopSequence {
		t.Fatalf("StopReason = %v (text %q), want %v for Stop=%q", res.StopReason, res.Text, StopReasonStopSequence, stop)
	}
	if strings.Contains(res.Text, stop) {
		t.Fatalf("text %q contains the stop sequence %q", res.Text, stop)
	}
	if !strings.HasPrefix(full.Text, res.Text) {
		t.Fatalf("text %q is not a prefix of the free-running text %q", res.Text, full.Text)
	}
	if res.Text != strings.TrimSuffix(full.Text, stop) {
		t.Fatalf("text %q, want the free-running text %q without its stop sequence", res.Text, full.Text)
	}
}

// TestStopReasons covers each way generation ends, including a context too
// small to hold a reply.
func TestStopReasons(t *testing.T) {
	m := requireModel(t)
	prompt := chatPrompt(t, m, "Name one colour, then stop.")

	t.Run("eos", func(t *testing.T) {
		c := newTestContext(t, m)
		res, err := c.Generate(context.Background(), prompt, GreedyCompletionOptions(64))
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if res.StopReason != StopReasonEOS {
			t.Fatalf("StopReason = %v (text %q), want %v", res.StopReason, res.Text, StopReasonEOS)
		}
	})

	t.Run("max tokens", func(t *testing.T) {
		c := newTestContext(t, m)
		res, err := c.Generate(context.Background(), prompt, GreedyCompletionOptions(1))
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if res.StopReason != StopReasonMaxTokens {
			t.Fatalf("StopReason = %v after %d tokens, want %v", res.StopReason, len(res.Tokens), StopReasonMaxTokens)
		}
	})

	t.Run("context full", func(t *testing.T) {
		c := newTestContext(t, m, ContextOptions{ContextSize: 64})
		opts := GreedyCompletionOptions(0)
		// Without IgnoreEOS the model would answer and stop long before the
		// cache fills.
		opts.IgnoreEOS = true
		res, err := c.Generate(context.Background(), "Repeat after me:", opts)
		if err != nil {
			t.Fatalf("Generate: %v", err)
		}
		if res.StopReason != StopReasonContextFull {
			t.Fatalf("StopReason = %v after %d tokens, want %v", res.StopReason, len(res.Tokens), StopReasonContextFull)
		}
		if len(res.Tokens) == 0 {
			t.Fatal("generated no tokens before the context filled")
		}
		if limit := c.ContextSize() + 1; len(res.Tokens) > limit {
			t.Fatalf("generated %d tokens in a %d token context", len(res.Tokens), c.ContextSize())
		}
		t.Logf("filled a %d token context with %d tokens", c.ContextSize(), len(res.Tokens))
	})

	t.Run("cancelled", func(t *testing.T) {
		c := newTestContext(t, m)
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		res, err := c.Generate(ctx, prompt, GreedyCompletionOptions(8))
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Generate on a cancelled context = %v, want context.Canceled", err)
		}
		if res.StopReason != StopReasonCancelled {
			t.Fatalf("StopReason = %v, want %v", res.StopReason, StopReasonCancelled)
		}
	})
}

// TestCancellationReturnsPartialResult cancels in the middle of decoding and
// checks that the tokens produced so far come back with the cancellation.
func TestCancellationReturnsPartialResult(t *testing.T) {
	m := requireModel(t)
	prompt := chatPrompt(t, m, "Write a long paragraph about the sea.")

	// Warming the cache keeps prompt evaluation out of the cancellation window,
	// so the test is about decoding rather than about how fast this machine
	// evaluates a prompt.
	warm := newTestContext(t, m)
	if _, err := warm.Generate(context.Background(), prompt, GreedyCompletionOptions(8)); err != nil {
		t.Fatalf("warm-up Generate: %v", err)
	}
	opts := GreedyCompletionOptions(4096)
	opts.IgnoreEOS = true

	t.Run("generate", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		time.AfterFunc(400*time.Millisecond, cancel)
		res, err := warm.Generate(ctx, prompt, opts)
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Generate: %v, want context.Canceled", err)
		}
		if res.StopReason != StopReasonCancelled {
			t.Fatalf("StopReason = %v, want %v", res.StopReason, StopReasonCancelled)
		}
		if len(res.Tokens) == 0 {
			t.Fatal("cancelled generation returned no partial tokens")
		}
		if len(res.Tokens) >= opts.MaxTokens {
			t.Fatalf("generated %d tokens, so cancellation did not interrupt a %d token cap", len(res.Tokens), opts.MaxTokens)
		}
		t.Logf("cancelled after %d tokens: %q", len(res.Tokens), res.Text)
	})

	t.Run("stream", func(t *testing.T) {
		if err := warm.Reset(); err != nil {
			t.Fatalf("Reset: %v", err)
		}
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		pieces := 0
		var streamErr error
		for _, err := range warm.Stream(ctx, prompt, opts) {
			if err != nil {
				streamErr = err
				break
			}
			pieces++
			if pieces == 3 {
				cancel()
			}
		}
		if !errors.Is(streamErr, context.Canceled) {
			t.Fatalf("Stream after cancel = %v, want context.Canceled", streamErr)
		}
		if pieces < 3 {
			t.Fatalf("Stream yielded %d pieces before cancelling, want at least 3", pieces)
		}
	})

	t.Run("break ends cleanly", func(t *testing.T) {
		if err := warm.Reset(); err != nil {
			t.Fatalf("Reset: %v", err)
		}
		pieces := 0
		for _, err := range warm.Stream(context.Background(), prompt, opts) {
			if err != nil {
				t.Fatalf("Stream: %v", err)
			}
			if pieces++; pieces == 3 {
				break
			}
		}
		if pieces != 3 {
			t.Fatalf("Stream yielded %d pieces, want 3", pieces)
		}
		// The context must still be usable after the consumer walked away.
		if _, err := warm.Generate(context.Background(), prompt, GreedyCompletionOptions(4)); err != nil {
			t.Fatalf("Generate after breaking out of Stream: %v", err)
		}
	})
}

// TestKVReuseSecondTurn checks that a second chat turn only evaluates the new
// tail of the prompt instead of the whole conversation.
func TestKVReuseSecondTurn(t *testing.T) {
	m := requireModel(t)
	c := newTestContext(t, m)

	system := "You are a terse assistant. Answer every question with exactly one short sentence, " +
		"then stop. Never explain your reasoning, never apologise, and never add a follow-up question."
	messages := []ChatMessage{
		SystemMessage(system),
		UserMessage("Name one primary colour."),
	}
	prompt, err := m.ApplyChatTemplate(messages, true)
	if err != nil {
		t.Fatalf("ApplyChatTemplate: %v", err)
	}

	turn1, err := c.Generate(context.Background(), prompt, GreedyCompletionOptions(24))
	if err != nil {
		t.Fatalf("turn 1 Generate: %v", err)
	}
	if turn1.Text == "" {
		t.Fatal("turn 1 generated no text")
	}
	if turn1.ReusedTokens != 0 {
		t.Fatalf("turn 1 ReusedTokens = %d on an empty cache, want 0", turn1.ReusedTokens)
	}

	messages = append(messages, AssistantMessage(turn1.Text), UserMessage("Name another primary colour."))
	prompt2, err := m.ApplyChatTemplate(messages, true)
	if err != nil {
		t.Fatalf("ApplyChatTemplate turn 2: %v", err)
	}
	turn2, err := c.Generate(context.Background(), prompt2, GreedyCompletionOptions(24))
	if err != nil {
		t.Fatalf("turn 2 Generate: %v", err)
	}
	if turn2.Text == "" {
		t.Fatal("turn 2 generated no text")
	}
	if turn2.ReusedTokens == 0 {
		t.Fatalf("turn 2 ReusedTokens = 0, want the first turn's cache to be reused (prompt %d tokens)", turn2.PromptTokens)
	}
	if turn2.ReusedTokens >= turn2.PromptTokens {
		t.Fatalf("turn 2 reused %d of %d prompt tokens, want at least one new token evaluated", turn2.ReusedTokens, turn2.PromptTokens)
	}
	if turn2.Metrics.PromptTokens >= turn1.Metrics.PromptTokens {
		t.Fatalf("turn 2 evaluated %d prompt tokens, want fewer than turn 1's %d (reused %d)",
			turn2.Metrics.PromptTokens, turn1.Metrics.PromptTokens, turn2.ReusedTokens)
	}
	t.Logf("turn 1: %d prompt tokens, %d reused; turn 2: %d prompt tokens, %d reused",
		turn1.PromptTokens, turn1.ReusedTokens, turn2.PromptTokens, turn2.ReusedTokens)
}

// TestChatTemplateRendering checks that the model's own template wraps the
// messages and leaves the assistant turn open only when asked.
func TestChatTemplateRendering(t *testing.T) {
	m := requireModel(t)
	if tmpl := m.ChatTemplate(""); tmpl == "" {
		t.Fatalf("model %s has no chat template; the chat tests need an instruct model", m.Path())
	}

	messages := []ChatMessage{
		SystemMessage("You are terse."),
		UserMessage("Name one colour."),
	}
	closed, err := m.ApplyChatTemplate(messages, false)
	if err != nil {
		t.Fatalf("ApplyChatTemplate(addAssistant=false): %v", err)
	}
	open, err := m.ApplyChatTemplate(messages, true)
	if err != nil {
		t.Fatalf("ApplyChatTemplate(addAssistant=true): %v", err)
	}

	for _, want := range []string{"You are terse.", "Name one colour."} {
		if !strings.Contains(closed, want) {
			t.Fatalf("rendered prompt %q does not contain %q", closed, want)
		}
	}
	if !strings.HasPrefix(open, closed) {
		t.Fatalf("open prompt %q does not extend the closed prompt %q", open, closed)
	}
	if len(open) <= len(closed) {
		t.Fatalf("open prompt is %d bytes, want more than the closed prompt's %d", len(open), len(closed))
	}

	withReply := append(append([]ChatMessage{}, messages...), AssistantMessage("Blue."))
	rendered, err := m.ApplyChatTemplate(withReply, true)
	if err != nil {
		t.Fatalf("ApplyChatTemplate with an assistant turn: %v", err)
	}
	if !strings.Contains(rendered, "Blue.") {
		t.Fatalf("rendered prompt %q does not contain the assistant turn", rendered)
	}

	empty, err := m.ApplyChatTemplate(nil, true)
	if err != nil {
		t.Fatalf("ApplyChatTemplate(nil): %v", err)
	}
	if empty != "" {
		t.Fatalf("ApplyChatTemplate(nil) = %q, want \"\"", empty)
	}
	t.Logf("rendered prompt: %q", open)
}

// TestEmbeddings checks the vector width, that identical text gives an
// identical vector, and that related sentences land closer than unrelated ones
// once the caller normalises.
func TestEmbeddings(t *testing.T) {
	m := requireEmbedModel(t)
	c := newTestContext(t, m, ContextOptions{Embeddings: true, Pooling: PoolingMean})
	want := m.EmbeddingSize()

	base, err := c.Embed(context.Background(), "A cat sits on a warm windowsill.")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(base) != want {
		t.Fatalf("embedding has %d dimensions, want EmbeddingSize %d", len(base), want)
	}
	for i, f := range base {
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			t.Fatalf("embedding[%d] = %v, want a finite value", i, f)
		}
	}
	norm := l2norm(base)
	if norm <= 0 {
		t.Fatalf("embedding norm = %v, want a positive value", norm)
	}
	// Embed documents an unnormalised vector, so normalising it has to be the
	// caller's one extra step.
	normalised := make([]float32, len(base))
	for i, f := range base {
		normalised[i] = f / float32(norm)
	}
	if got := l2norm(normalised); math.Abs(got-1) > 1e-5 {
		t.Fatalf("normalised embedding norm = %v, want 1", got)
	}

	again, err := c.Embed(context.Background(), "A cat sits on a warm windowsill.")
	if err != nil {
		t.Fatalf("second Embed: %v", err)
	}
	if !equalVectors(again, base) {
		t.Fatal("embedding the same text twice gave different vectors")
	}

	near, err := c.Embed(context.Background(), "A kitten rests on a sunny window ledge.")
	if err != nil {
		t.Fatalf("Embed near: %v", err)
	}
	far, err := c.Embed(context.Background(), "The stock market fell sharply on Tuesday.")
	if err != nil {
		t.Fatalf("Embed far: %v", err)
	}
	cosNear := cosine(base, near)
	cosFar := cosine(base, far)
	if cosNear <= cosFar+0.1 {
		t.Fatalf("cosine to a related sentence = %.3f, to an unrelated one = %.3f; want a clear gap", cosNear, cosFar)
	}
	t.Logf("%d dimensions, norm %.3f, cosine near %.3f far %.3f", len(base), norm, cosNear, cosFar)
}

// TestCloseModelWithOpenContext checks that the model refuses to free weights a
// context still points into, and that the failure leaves everything usable.
func TestCloseModelWithOpenContext(t *testing.T) {
	m := requireModel(t)
	c := newTestContext(t, m)

	err := m.Close()
	if err == nil {
		t.Fatal("Close on a model with an open context returned nil, want an error")
	}
	if !errors.Is(err, ErrClosed) {
		t.Fatalf("Close with an open context = %v, want ErrClosed", err)
	}

	res, err := c.Generate(context.Background(), "Name one colour.", GreedyCompletionOptions(8))
	if err != nil {
		t.Fatalf("Generate after the refused Close: %v", err)
	}
	if len(res.Tokens) == 0 {
		t.Fatal("Generate after the refused Close produced no tokens")
	}

	if err := c.Close(); err != nil {
		t.Fatalf("context Close: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("model Close after closing the context: %v", err)
	}
}

// TestGPUOffload loads every layer onto the GPU. It is opt-in because CI has no
// GPU, and it is the only test that leaves the CPU path.
func TestGPUOffload(t *testing.T) {
	path := os.Getenv(envTestModel)
	if path == "" {
		t.Skipf("%s is unset; run scripts/fetch-test-models.sh and export it", envTestModel)
	}
	if os.Getenv(envTestGPU) != "1" {
		t.Skipf("%s is not 1; set it to run the offload test", envTestGPU)
	}

	m, err := LoadModel(path, ModelParams{GPULayers: -1, UseMmap: true})
	if err != nil {
		t.Fatalf("LoadModel(%s) with GPULayers=-1: %v", path, err)
	}
	defer m.Close()
	c := newTestContext(t, m)

	res, err := c.Generate(context.Background(), "Name one colour.", GreedyCompletionOptions(16))
	if err != nil {
		t.Fatalf("Generate: %v", err)
	}
	if len(res.Tokens) == 0 {
		t.Fatal("Generate produced no tokens")
	}
	t.Logf("%s offloaded (%s): %d tokens at %.1f tok/s", path, m.Description(), len(res.Tokens), res.Metrics.TokensPerSecond())
}
