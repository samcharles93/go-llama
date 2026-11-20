package llama

import "strings"

// stopMatcher turns a token-by-token stream into text that never contains a
// stop sequence.
//
// It exists because a stop string can straddle tokens ("</" and "s>" arrive as
// separate pieces), so each piece has to be held back until it cannot be the
// start of a stop sequence. It also refuses to emit a partial UTF-8 rune, since
// byte-level tokenizers can split a multi-byte character across two tokens.
//
// The held-back bytes live in a single pending string rather than a buffer that
// is copied back and forth, so a piece with nothing to hold costs no
// allocation. Only one pending suffix is enough: a stop can only ever start in
// the run of bytes that has not been proven free of stop prefixes.
type stopMatcher struct {
	stops   []string
	pending string
	hit     bool
}

func newStopMatcher(stops []string) *stopMatcher {
	m := &stopMatcher{}
	seen := make(map[string]struct{}, len(stops))
	for _, s := range stops {
		if s == "" {
			continue
		}
		if _, dup := seen[s]; dup {
			// A duplicate would only make every scan slower.
			continue
		}
		seen[s] = struct{}{}
		m.stops = append(m.stops, s)
	}
	return m
}

// Push appends a piece of generated text and returns the part that is safe to
// emit (may be empty). When a stop sequence matches, hit is true and the caller
// must stop generating; the stop text itself is dropped.
func (m *stopMatcher) Push(piece string) (emit string, hit bool) {
	if m == nil || len(m.stops) == 0 {
		return piece, false
	}
	if m.hit {
		// A stop already ended the stream; nothing after it is output.
		return "", true
	}
	s := piece
	if m.pending != "" {
		s = m.pending + piece
		m.pending = ""
	}
	return m.drain(s)
}

// Flush returns anything still held back, for use when generation ends without
// a stop sequence. Held-back bytes are by definition not the start of a
// complete stop, so they are always safe to release.
func (m *stopMatcher) Flush() string {
	if m == nil || m.hit {
		return ""
	}
	out := m.pending
	m.pending = ""
	return out
}

// drain emits the prefix of s that cannot be part of a stop sequence and parks
// the rest in m.pending. s is either a fresh piece or pending plus a piece, so
// the caller owns it.
func (m *stopMatcher) drain(s string) (string, bool) {
	// Earliest complete stop sequence wins; ties go to the leftmost match,
	// which is the one the model actually produced first. This is what makes
	// overlapping stops behave: with "\n" and "\n\n" configured, the first
	// newline is the stop, so the second never reaches the caller.
	cut := -1
	for _, stop := range m.stops {
		if i := strings.Index(s, stop); i >= 0 && (cut < 0 || i < cut) {
			cut = i
		}
	}
	if cut >= 0 {
		m.hit = true
		return s[:cut], true
	}

	// Hold back any suffix that could still grow into a stop sequence: a
	// proper prefix of a stop at the end of the piece may be completed by the
	// next one.
	hold := 0
	for _, stop := range m.stops {
		limit := len(stop) - 1
		if limit > len(s) {
			limit = len(s)
		}
		for n := limit; n > hold; n-- {
			if strings.HasSuffix(s, stop[:n]) {
				hold = n
				break
			}
		}
	}
	emit := s[:len(s)-hold]
	// Trailing bytes that do not yet form a complete rune are held too:
	// byte-level tokenizers split multi-byte characters across tokens, and a
	// half-written rune is not valid output.
	if n := incompleteRuneSuffix(emit); n > 0 {
		hold += n
		emit = s[:len(s)-hold]
	}
	m.pending = s[len(emit):]
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
