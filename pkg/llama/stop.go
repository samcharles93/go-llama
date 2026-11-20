package llama

import "strings"

// stopMatcher turns a token-by-token stream into text that never contains a
// stop sequence.
//
// It exists because a stop string can straddle tokens ("</" and "s>" arrive as
// separate pieces), so each piece has to be held back until it cannot be the
// start of a stop sequence. It also refuses to emit a partial UTF-8 rune, since
// byte-level tokenizers can split a multi-byte character across two tokens.
type stopMatcher struct {
	stops  []string
	maxLen int
	buf    strings.Builder
}

func newStopMatcher(stops []string) *stopMatcher {
	m := &stopMatcher{}
	for _, s := range stops {
		if s == "" {
			continue
		}
		m.stops = append(m.stops, s)
		if len(s) > m.maxLen {
			m.maxLen = len(s)
		}
	}
	return m
}

// active reports whether any stop sequence is configured.
func (m *stopMatcher) active() bool { return m != nil && len(m.stops) > 0 }

// Push appends a piece of generated text and returns the part that is safe to
// emit (may be empty). When a stop sequence matches, hit is true and the caller
// must stop generating; the stop text itself is dropped.
func (m *stopMatcher) Push(piece string) (emit string, hit bool) {
	if m == nil {
		return piece, false
	}
	m.buf.WriteString(piece)
	return m.drain(false)
}

// Flush returns anything still held back, for use when generation ends without
// a stop sequence.
func (m *stopMatcher) Flush() string {
	if m == nil {
		return ""
	}
	out, _ := m.drain(true)
	return out
}

func (m *stopMatcher) drain(final bool) (string, bool) {
	s := m.buf.String()
	m.buf.Reset()
	if len(m.stops) == 0 {
		return s, false
	}

	// Earliest complete stop sequence wins; ties go to the leftmost match,
	// which is the one the model actually produced first.
	cut := -1
	for _, stop := range m.stops {
		if i := strings.Index(s, stop); i >= 0 && (cut < 0 || i < cut) {
			cut = i
		}
	}
	if cut >= 0 {
		return s[:cut], true
	}
	if final {
		return s, false
	}

	// Hold back any suffix that could still grow into a stop sequence, and any
	// trailing bytes that do not yet form a complete rune: byte-level
	// tokenizers can split a multi-byte character across two tokens, and a
	// half-written rune is not valid output.
	hold := 0
	for _, stop := range m.stops {
		max := len(stop) - 1
		if max > len(s) {
			max = len(s)
		}
		for n := max; n > hold; n-- {
			if strings.HasSuffix(s, stop[:n]) {
				hold = n
				break
			}
		}
	}
	if n := incompleteRuneSuffix(s); n > hold {
		hold = n
	}
	if hold > len(s) {
		hold = len(s)
	}
	emit := s[:len(s)-hold]
	m.buf.WriteString(s[len(s)-hold:])
	return emit, false
}

// incompleteRuneSuffix reports how many trailing bytes of s belong to a rune
// that has not fully arrived yet.
func incompleteRuneSuffix(s string) int {
	n := len(s)
	for i := 0; i < 4 && i < n; i++ {
		b := s[n-1-i]
		if b&0xC0 == 0x80 {
			continue // continuation byte, keep looking for the lead
		}
		want := runeLen(b)
		have := i + 1
		if want > have {
			return have
		}
		return 0
	}
	return 0
}

// runeLen reports how many bytes a rune starting with lead occupies.
func runeLen(lead byte) int {
	switch {
	case lead < 0x80:
		return 1
	case lead&0xE0 == 0xC0:
		return 2
	case lead&0xF0 == 0xE0:
		return 3
	case lead&0xF8 == 0xF0:
		return 4
	}
	return 1
}
