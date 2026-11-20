package llama

import (
	"strings"
	"testing"
	"unicode/utf8"
)

// pushAll feeds pieces to the matcher and returns the emitted text and whether a
// stop was hit, stopping at the first hit the way generation does.
func pushAll(m *stopMatcher, pieces ...string) (string, bool) {
	var sb strings.Builder
	for _, p := range pieces {
		out, hit := m.Push(p)
		sb.WriteString(out)
		if hit {
			return sb.String(), true
		}
	}
	return sb.String(), false
}

// Overlapping stops are the case a naive suffix check gets wrong: "\n" is both a
// stop on its own and the first byte of "\n\n", so the shorter one has to win and
// nothing that matches it may be emitted.
func TestStopMatcherOverlappingStops(t *testing.T) {
	cases := []struct {
		name   string
		stops  []string
		pieces []string
		want   string
		hit    bool
	}{
		{
			name:   "short stop split from long stop",
			stops:  []string{"\n", "\n\n"},
			pieces: []string{"line", "\n"},
			want:   "line",
			hit:    true,
		},
		{
			name:   "long stop configured first",
			stops:  []string{"\n\n", "\n"},
			pieces: []string{"line", "\n"},
			want:   "line",
			hit:    true,
		},
		{
			name:   "long stop in one piece",
			stops:  []string{"\n", "\n\n"},
			pieces: []string{"a\n\nb"},
			want:   "a",
			hit:    true,
		},
		{
			name:   "prefix byte held for the long stop",
			stops:  []string{"\n\n"},
			pieces: []string{"a\n", "b"},
			want:   "a\nb",
			hit:    false,
		},
		{
			name:   "single newline does not hit a doubled stop",
			stops:  []string{"\n\n"},
			pieces: []string{"a", "\n", "b", "\n", "\n"},
			want:   "a\nb",
			hit:    true,
		},
		{
			name:   "suffix of one stop starts another",
			stops:  []string{"bc", "b"},
			pieces: []string{"ab", "c"},
			want:   "a",
			hit:    true,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, hit := pushAll(newStopMatcher(tc.stops), tc.pieces...)
			if hit != tc.hit {
				t.Fatalf("hit = %v, want %v", hit, tc.hit)
			}
			if got != tc.want {
				t.Fatalf("emitted %q, want %q", got, tc.want)
			}
			for _, stop := range tc.stops {
				if strings.Contains(got, stop) {
					t.Fatalf("emitted %q contains stop %q", got, stop)
				}
			}
		})
	}
}

func TestStopMatcherDropsDuplicateStops(t *testing.T) {
	m := newStopMatcher([]string{"END", "END", "", "STOP", "END"})
	if got, want := len(m.stops), 2; got != want {
		t.Fatalf("matcher kept %d stops (%v), want %d", got, m.stops, want)
	}
	got, hit := pushAll(m, "abc", "END")
	if !hit || got != "abc" {
		t.Fatalf("Push = %q, %v; want \"abc\", true", got, hit)
	}
}

func TestStopMatcherEmptyInput(t *testing.T) {
	m := newStopMatcher([]string{"END"})
	if got, hit := m.Push(""); got != "" || hit {
		t.Fatalf("Push(\"\") = %q, %v; want \"\", false", got, hit)
	}
	if got := m.Flush(); got != "" {
		t.Fatalf("Flush after empty push = %q, want \"\"", got)
	}

	none := newStopMatcher(nil)
	if got, _ := none.Push("text"); got != "text" {
		t.Fatalf("matcher without stops emitted %q, want the piece unchanged", got)
	}
}

// A three-byte rune split over three pieces must never reach the caller in
// pieces, because each one would be invalid UTF-8 on its own.
func TestStopMatcherHoldsSplitRunes(t *testing.T) {
	m := newStopMatcher([]string{"STOP"})
	// "€" is E2 82 AC.
	if got, _ := m.Push("cost \xe2"); got != "cost " {
		t.Fatalf("first piece emitted %q, want \"cost \"", got)
	}
	if got, _ := m.Push("\x82"); got != "" {
		t.Fatalf("second piece emitted %q, want nothing", got)
	}
	got, hit := m.Push("\xac each")
	if hit {
		t.Fatal("unexpected stop hit")
	}
	if want := "€ each"; got != want {
		t.Fatalf("third piece emitted %q, want %q", got, want)
	}
}

func TestStopMatcherNothingAfterHit(t *testing.T) {
	m := newStopMatcher([]string{"END"})
	if _, hit := m.Push("done END"); !hit {
		t.Fatal("stop sequence not detected")
	}
	if got, hit := m.Push("trailing"); got != "" || !hit {
		t.Fatalf("Push after hit = %q, %v; want \"\", true", got, hit)
	}
	if got := m.Flush(); got != "" {
		t.Fatalf("Flush after hit = %q, want \"\"", got)
	}
}

func TestStopMatcherFlushIsIdempotent(t *testing.T) {
	m := newStopMatcher([]string{"END"})
	if got, _ := m.Push("abcEN"); got != "abc" {
		t.Fatalf("Push = %q, want \"abc\"", got)
	}
	if got := m.Flush(); got != "EN" {
		t.Fatalf("Flush = %q, want \"EN\"", got)
	}
	if got := m.Flush(); got != "" {
		t.Fatalf("second Flush = %q, want \"\"", got)
	}
}

// A piece with nothing to hold back must not allocate, which is the whole point
// of keeping one pending string instead of rebuilding a buffer per piece.
func TestStopMatcherPushAllocations(t *testing.T) {
	m := newStopMatcher([]string{"\n\n", "END"})
	allocs := testing.AllocsPerRun(100, func() {
		if out, _ := m.Push("steady"); out != "steady" {
			t.Fatalf("Push = %q", out)
		}
	})
	if allocs != 0 {
		t.Fatalf("Push allocated %v times per call, want 0", allocs)
	}
}

func BenchmarkStopMatcherPush(b *testing.B) {
	m := newStopMatcher([]string{"\n\n", "END"})
	b.ReportAllocs()
	for range b.N {
		m.Push("steady")
	}
}

// FuzzStopMatcher asserts the two properties the matcher exists for: a stop
// sequence never reaches the caller, and a valid UTF-8 stream is never split
// mid-rune. The chunk parameter varies how the text is cut into pieces, because
// the interesting failures are at piece boundaries.
func FuzzStopMatcher(f *testing.F) {
	f.Add("hello\nworld\n\n", "\n", 3)
	f.Add("cost €5 each", "€", 1)
	f.Add("one END two", "END", 2)
	f.Add("", "\n", 1)
	f.Add("aaaa", "aa", 4)

	f.Fuzz(func(t *testing.T, text, stop string, chunk int) {
		if stop == "" {
			// Empty stops are filtered out, so they test nothing here.
			return
		}
		if chunk < 1 {
			chunk = 1
		}
		// A duplicated stop and a shorter overlapping one exercise the
		// dedupe and prefix-hold paths at the same time.
		stops := []string{stop, stop + stop, stop[:1]}

		m := newStopMatcher(stops)
		var out strings.Builder
		validUTF8 := utf8.ValidString(text) && utf8.ValidString(stop)

		for i := 0; i < len(text); i += chunk {
			end := min(i+chunk, len(text))
			emit, hit := m.Push(text[i:end])
			out.WriteString(emit)
			if strings.Contains(m.pending, stop) {
				t.Fatalf("pending %q contains the stop %q", m.pending, stop)
			}
			if got := out.String(); strings.Contains(got, stop) {
				t.Fatalf("emitted %q contains the stop %q", got, stop)
			}
			if validUTF8 && !utf8.ValidString(emit) {
				t.Fatalf("emitted invalid UTF-8 %q from valid input %q", emit, text)
			}
			if hit {
				break
			}
		}

		if tail := m.Flush(); tail != "" {
			out.WriteString(tail)
			if validUTF8 && !utf8.ValidString(tail) {
				t.Fatalf("Flush emitted invalid UTF-8 %q from valid input %q", tail, text)
			}
		}
		if got := out.String(); strings.Contains(got, stop) {
			t.Fatalf("output %q contains the stop %q", got, stop)
		}
		if got := m.Flush(); got != "" {
			t.Fatalf("second Flush = %q, want \"\"", got)
		}
	})
}
