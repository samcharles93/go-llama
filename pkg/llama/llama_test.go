package llama

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"testing"
)

// These tests need llama.cpp linked but no model file, so they run in CI.
// Model-backed tests live in integration_test.go behind the integration tag.

func TestBatchCapacityIsEnforced(t *testing.T) {
	b, err := NewBatch(2, 1)
	if err != nil {
		t.Fatalf("NewBatch: %v", err)
	}
	defer b.Close()

	for i := range 2 {
		if err := b.AddOne(Token(i), int32(i), true); err != nil {
			t.Fatalf("Add %d: %v", i, err)
		}
	}
	if b.Len() != 2 {
		t.Fatalf("Len = %d, want 2", b.Len())
	}
	if err := b.AddOne(Token(3), 2, true); !errors.Is(err, ErrBatchFull) {
		t.Fatalf("overflowing Add = %v, want ErrBatchFull", err)
	}

	b.Reset()
	if b.Len() != 0 {
		t.Fatalf("Len after Reset = %d, want 0", b.Len())
	}
	if err := b.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if err := b.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	if err := b.AddOne(Token(1), 0, true); !errors.Is(err, ErrClosed) {
		t.Fatalf("Add after Close = %v, want ErrClosed", err)
	}
}

func TestBatchRejectsTooManySequences(t *testing.T) {
	b, err := NewBatch(4, 1)
	if err != nil {
		t.Fatalf("NewBatch: %v", err)
	}
	defer b.Close()

	if err := b.Add(Token(1), 0, []SeqID{0, 1}, true); err == nil {
		t.Fatal("Add with two sequence ids on a single-sequence batch succeeded")
	}
	if err := b.Add(Token(1), 0, nil, true); err == nil {
		t.Fatal("Add with no sequence ids succeeded")
	}
}

func TestStopMatcherHoldsBackPartialSequence(t *testing.T) {
	m := newStopMatcher([]string{"</s>"})

	if got, hit := m.Push("hello</"); hit || got != "hello" {
		t.Fatalf("Push(hello</) = %q, %v; want \"hello\", false", got, hit)
	}
	if got, hit := m.Push("s>"); !hit || got != "" {
		t.Fatalf("Push(s>) = %q, %v; want \"\", true", got, hit)
	}
}

func TestStopMatcherCarriesSplitRunes(t *testing.T) {
	m := newStopMatcher([]string{"END"})
	// "é" is two bytes; the rune is split across two pieces.
	first := "caf\xc3"
	second := "\xa9 END"
	got, hit := m.Push(first)
	if hit {
		t.Fatal("unexpected stop hit")
	}
	if got != "caf" {
		t.Fatalf("first piece emitted %q, want \"caf\" (partial rune must be held back)", got)
	}
	rest, hit := m.Push(second)
	if !hit {
		t.Fatal("stop sequence not detected")
	}
	if rest != "\xc3\xa9 " {
		t.Fatalf("rest = %q, want the completed rune and no stop text", rest)
	}
}

func TestStopMatcherFlushReturnsHeldText(t *testing.T) {
	m := newStopMatcher([]string{"STOP"})
	if got, _ := m.Push("abcST"); got != "abc" {
		t.Fatalf("Push = %q, want \"abc\"", got)
	}
	if got := m.Flush(); got != "ST" {
		t.Fatalf("Flush = %q, want \"ST\"", got)
	}
}

func TestCompletionOptionsValidate(t *testing.T) {
	cases := map[string]CompletionOptions{
		"negative top k":   {TopK: -1},
		"top p above one":  {TopP: 1.5},
		"min p below zero": {MinP: -0.1},
		"empty stop":       {Stop: []string{""}},
		"top k out":        {RepeatLastN: -2},
	}
	for name, opts := range cases {
		if err := opts.validate(); err == nil {
			t.Errorf("%s: validate returned nil", name)
		}
	}
	if err := DefaultCompletionOptions().validate(); err != nil {
		t.Errorf("defaults do not validate: %v", err)
	}
}

func TestLoadModelRejectsMissingFile(t *testing.T) {
	_, err := LoadModel("/nonexistent/model.gguf")
	if !errors.Is(err, ErrModelLoad) {
		t.Fatalf("LoadModel = %v, want ErrModelLoad", err)
	}
}

func TestGenerateRequiresLiveContext(t *testing.T) {
	c := &Context{}
	if _, err := c.Generate(context.Background(), "hi", GreedyCompletionOptions(4)); !errors.Is(err, ErrClosed) {
		t.Fatalf("Generate on closed context = %v, want ErrClosed", err)
	}
}

// logRecorder captures the records the log bridge emits.
type logRecorder struct {
	mu      sync.Mutex
	records []slog.Record
}

func (h *logRecorder) Enabled(context.Context, slog.Level) bool { return true }
func (h *logRecorder) Handle(_ context.Context, r slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.records = append(h.records, r.Clone())
	return nil
}
func (h *logRecorder) WithAttrs([]slog.Attr) slog.Handler { return h }
func (h *logRecorder) WithGroup(string) slog.Handler      { return h }

func (h *logRecorder) take() []slog.Record {
	h.mu.Lock()
	defer h.mu.Unlock()
	out := h.records
	h.records = nil
	return out
}

func TestLogBridgeMapsSeverities(t *testing.T) {
	rec := &logRecorder{}
	SetLogger(slog.New(rec))
	defer SetLogger(nil)

	cases := map[int]slog.Level{
		levelDebug: slog.LevelDebug,
		levelInfo:  slog.LevelInfo,
		levelWarn:  slog.LevelWarn,
		levelError: slog.LevelError,
	}
	for level, want := range cases {
		handleLogFragment(level, "message at level\n")
		got := rec.take()
		if len(got) != 1 {
			t.Fatalf("level %d produced %d records, want 1", level, len(got))
		}
		if got[0].Level != want {
			t.Errorf("level %d mapped to %v, want %v", level, got[0].Level, want)
		}
		if got[0].Message != "message at level" {
			t.Errorf("level %d message = %q", level, got[0].Message)
		}
	}
}

func TestLogBridgeAssemblesContinuations(t *testing.T) {
	rec := &logRecorder{}
	SetLogger(slog.New(rec))
	defer SetLogger(nil)

	handleLogFragment(levelWarn, "first half ")
	handleLogFragment(levelCont, "second half")
	if got := rec.take(); len(got) != 0 {
		t.Fatalf("emitted %d records before the message ended: %v", len(got), got)
	}
	handleLogFragment(levelCont, "\n")
	got := rec.take()
	if len(got) != 1 {
		t.Fatalf("emitted %d records, want 1", len(got))
	}
	if got[0].Message != "first half second half" {
		t.Fatalf("assembled message = %q", got[0].Message)
	}
	if got[0].Level != slog.LevelWarn {
		t.Fatalf("continuation changed severity to %v", got[0].Level)
	}
}

func TestCloseIsIdempotentAndOrdered(t *testing.T) {
	m := &Model{path: "test"}
	if err := m.Close(); err != nil {
		t.Fatalf("Close on empty model: %v", err)
	}
	if err := m.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	c := &Context{}
	if err := c.Close(); err != nil {
		t.Fatalf("Close on empty context: %v", err)
	}
}
