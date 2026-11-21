package llama

import "testing"

func TestClampThreads(t *testing.T) {
	tests := []struct {
		name        string
		cores, ncpu int
		want        int32
	}{
		{"detected", 4, 16, 4},
		{"detection failed, large machine", 0, 16, 8},
		{"detection failed, small machine", 0, 4, 4},
		{"affinity narrower than cores", 12, 6, 6},
		{"no cpus reported", 0, 0, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := clampThreads(tt.cores, tt.ncpu); got != tt.want {
				t.Errorf("clampThreads(%d, %d) = %d, want %d", tt.cores, tt.ncpu, got, tt.want)
			}
		})
	}
}

func TestDefaultThreadsIsPositive(t *testing.T) {
	if n := DefaultThreads(); n < 1 {
		t.Fatalf("DefaultThreads() = %d, want at least 1", n)
	}
}

func TestDefaultContextOptionsUseDefaultThreads(t *testing.T) {
	o := DefaultContextOptions()
	if o.Threads != DefaultThreads() || o.ThreadsBatch != DefaultThreads() {
		t.Fatalf("DefaultContextOptions threads = %d/%d, want %d", o.Threads, o.ThreadsBatch, DefaultThreads())
	}
	if o.FlashAttention != FlashAttentionAuto {
		t.Fatalf("DefaultContextOptions FlashAttention = %v, want auto", o.FlashAttention)
	}
}

// TestFlashAttentionModeMapsToLlamaEnum checks each mode against llama.cpp's
// own name for the enum value it maps to. A bool used to map false to nothing,
// which left llama.cpp's default of auto in place, so "off" never disabled it.
func TestFlashAttentionModeMapsToLlamaEnum(t *testing.T) {
	tests := []struct {
		mode FlashAttentionMode
		want string
	}{
		{FlashAttentionAuto, "auto"},
		{FlashAttentionOn, "enabled"},
		{FlashAttentionOff, "disabled"},
	}
	for _, tt := range tests {
		if got := tt.mode.llamaName(); got != tt.want {
			t.Errorf("%v maps to llama.cpp %q, want %q", tt.mode, got, tt.want)
		}
	}
}
