package llama

import (
	"slices"
	"testing"
)

// These tests pin behaviour that changed under the llama.cpp b12603 API: the
// load_mode enum that replaced use_mmap and use_mlock, and penalty windows,
// which llama.cpp now clamps to 0 (disabled) when negative.

func TestLoadModeMapsToLlamaEnum(t *testing.T) {
	tests := []struct {
		mmap, mlock bool
		want        string
	}{
		{true, false, "mmap"},
		{false, true, "mlock"},
		{true, true, "mmap+mlock"},
		{false, false, "none"},
	}
	for _, tt := range tests {
		p := ModelParams{UseMmap: tt.mmap, UseMlock: tt.mlock}
		if got := p.loadModeName(); got != tt.want {
			t.Errorf("UseMmap=%v UseMlock=%v maps to %q, want %q", tt.mmap, tt.mlock, got, tt.want)
		}
	}
}

func TestResolveWindow(t *testing.T) {
	tests := []struct {
		in, want int32
	}{
		{64, 64},
		{0, 0},
		{-1, int32(MaxDefaultContextSize)},
	}
	for _, tt := range tests {
		if got := resolveWindow(tt.in, nil); got != tt.want {
			t.Errorf("resolveWindow(%d, nil) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

// TestWholeContextPenaltyStaysEnabled guards the documented RepeatLastN of -1.
// Passed through unresolved, llama.cpp would build a disabled "?penalties" stage.
func TestWholeContextPenaltyStaysEnabled(t *testing.T) {
	s, err := newSamplerFromOptions(nil, CompletionOptions{RepeatPenalty: 1.2, RepeatLastN: -1})
	if err != nil {
		t.Fatalf("newSamplerFromOptions: %v", err)
	}
	defer s.Close()
	if got := s.Stages(); !slices.Equal(got, []string{"penalties", "greedy"}) {
		t.Fatalf("stages = %v, want an enabled penalties stage then greedy", got)
	}

	hand := NewSamplerChain().AddPenalties(-1, 1.2, 0, 0).AddGreedy()
	defer hand.Close()
	if got := hand.Stages(); !slices.Equal(got, []string{"penalties", "greedy"}) {
		t.Fatalf("AddPenalties(-1) stages = %v, want an enabled penalties stage", got)
	}
}

func TestDryPenaltyLastNDefaults(t *testing.T) {
	tests := []struct {
		in, want int32
	}{
		{0, 64},
		{128, 128},
		{-1, -1},
	}
	for _, tt := range tests {
		if got := (CompletionOptions{DryPenaltyLastN: tt.in}).dryPenaltyLastN(); got != tt.want {
			t.Errorf("dryPenaltyLastN(%d) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

// TestReleaseLorasDropsAppliedCounts covers the bookkeeping that lets an
// adapter close once the contexts using it are gone. RemoveAllLoras and
// Context.Close used to leave the count raised, so Close refused forever.
func TestReleaseLorasDropsAppliedCounts(t *testing.T) {
	a := &LoraAdapter{path: "a.gguf", applied: 2}
	b := &LoraAdapter{path: "b.gguf", applied: 1}
	releaseLoras([]loraSlot{{adapter: a, scale: 1}, {adapter: b, scale: 0.5}})
	if a.Applied() != 1 || b.Applied() != 0 {
		t.Fatalf("applied counts = %d, %d, want 1, 0", a.Applied(), b.Applied())
	}
	if err := b.Close(); err != nil {
		t.Fatalf("Close after release: %v", err)
	}
	// Never below zero, even if released twice.
	releaseLoras([]loraSlot{{adapter: b}})
	if b.Applied() != 0 {
		t.Fatalf("applied = %d after a second release, want 0", b.Applied())
	}
}
