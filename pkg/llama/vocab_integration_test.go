//go:build integration

package llama

import (
	"errors"
	"os"
	"regexp"
	"strings"
	"testing"
)

// These tests need a real vocabulary, so they run under the integration tag and
// skip unless GO_LLAMA_TEST_MODEL is set.

func vocabModel(t *testing.T) *Model {
	t.Helper()
	path := os.Getenv("GO_LLAMA_TEST_MODEL")
	if path == "" {
		t.Skip("GO_LLAMA_TEST_MODEL is not set")
	}
	// GPULayers 0 keeps the test on the CPU so parallel lanes do not contend for
	// the card.
	m, err := LoadModel(path, ModelParams{GPULayers: 0})
	if err != nil {
		t.Fatalf("LoadModel(%s): %v", path, err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

func TestTokenizeDetokenizeRoundTrip(t *testing.T) {
	v := vocabModel(t).Vocab()
	const text = "The quick brown fox jumps over the lazy dog."

	tokens, err := v.Tokenize(text, false)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	if len(tokens) == 0 {
		t.Fatalf("Tokenize(%q) returned no tokens", text)
	}
	if bos := v.BOS(); bos >= 0 && tokens[0] == bos {
		t.Fatalf("Tokenize with addSpecial=false still prepended BOS %d", bos)
	}
	got, err := v.Detokenize(tokens)
	if err != nil {
		t.Fatalf("Detokenize: %v", err)
	}
	if got != text {
		t.Fatalf("round trip = %q, want %q", got, text)
	}

	withBOS, err := v.Tokenize(text, true)
	if err != nil {
		t.Fatalf("Tokenize with addSpecial: %v", err)
	}
	if bos := v.BOS(); bos >= 0 {
		if len(withBOS) != len(tokens)+1 || withBOS[0] != bos {
			t.Fatalf("addSpecial=true gave %d tokens starting %d, want %d tokens starting with BOS %d",
				len(withBOS), withBOS[0], len(tokens)+1, bos)
		}
	}
}

func TestPieceToTokenRoundTrip(t *testing.T) {
	v := vocabModel(t).Vocab()

	tokens, err := v.Tokenize("the quick brown fox", false)
	if err != nil {
		t.Fatalf("Tokenize: %v", err)
	}
	for _, tok := range tokens {
		piece, err := v.TokenToPiece(tok)
		if err != nil {
			t.Fatalf("TokenToPiece(%d): %v", tok, err)
		}
		back, err := v.PieceToToken(piece)
		if err != nil {
			t.Fatalf("token %d renders as %q but did not map back: %v", tok, piece, err)
		}
		if back != tok {
			t.Fatalf("token %d renders as %q and mapped back to %d", tok, piece, back)
		}
	}
}

func TestPieceToTokenRejectsUnknownPieces(t *testing.T) {
	v := vocabModel(t).Vocab()

	if _, err := v.PieceToToken("the quick brown fox jumps over the lazy dog"); !errors.Is(err, ErrTokenize) {
		t.Fatalf("PieceToToken of several tokens = %v, want ErrTokenize", err)
	}
}

func TestSpecialTokenPieceMapsBack(t *testing.T) {
	v := vocabModel(t).Vocab()

	bos := v.BOS()
	if bos < 0 {
		t.Skip("model has no BOS token")
	}
	piece, err := v.TokenToPiece(bos)
	if err != nil {
		t.Fatalf("TokenToPiece(BOS): %v", err)
	}
	if !strings.HasPrefix(piece, "<") {
		t.Skipf("BOS token %d renders as %q, which is not a special token spelling", bos, piece)
	}
	back, err := v.PieceToToken(piece)
	if err != nil {
		t.Fatalf("PieceToToken(%q) for BOS: %v", piece, err)
	}
	if back != bos {
		t.Fatalf("BOS token %d renders as %q and mapped back to %d", bos, piece, back)
	}
}

func TestInvalidTokenIDsAreRejected(t *testing.T) {
	v := vocabModel(t).Vocab()

	for _, tok := range []Token{-1, Token(v.Size())} {
		if _, err := v.TokenToPiece(tok); !errors.Is(err, ErrTokenize) {
			t.Fatalf("TokenToPiece(%d) = %v, want ErrTokenize", tok, err)
		}
	}
	if _, err := v.Detokenize([]Token{1, Token(v.Size())}); !errors.Is(err, ErrTokenize) {
		t.Fatalf("Detokenize with an out of range id = %v, want ErrTokenize", err)
	}
}

func TestVocabPredicatesStayInRange(t *testing.T) {
	v := vocabModel(t).Vocab()

	if v.IsControl(Token(-1)) || v.IsControl(Token(v.Size())) {
		t.Fatal("IsControl reported true for an id outside the vocabulary")
	}
	if v.IsByte(Token(-1)) || v.IsByte(Token(v.Size())) {
		t.Fatal("IsByte reported true for an id outside the vocabulary")
	}

	controls, bytes := 0, 0
	for i := range v.Size() {
		if v.IsControl(Token(i)) {
			controls++
		}
		if v.IsByte(Token(i)) {
			bytes++
		}
	}
	t.Logf("%d tokens: %d control, %d byte", v.Size(), controls, bytes)
}

var bytePiece = regexp.MustCompile(`^<0x[0-9A-Fa-f]{2}>$`)

func TestByteTokensAreReported(t *testing.T) {
	path := os.Getenv("GO_LLAMA_TEST_EMBED_MODEL")
	if path == "" {
		t.Skip("GO_LLAMA_TEST_EMBED_MODEL is not set")
	}
	m, err := LoadModel(path, ModelParams{GPULayers: 0})
	if err != nil {
		t.Fatalf("LoadModel(%s): %v", path, err)
	}
	t.Cleanup(func() { _ = m.Close() })
	v := m.Vocab()

	found := 0
	for i := range v.Size() {
		tok := Token(i)
		if !v.IsByte(tok) {
			continue
		}
		found++
		piece, err := v.TokenToPiece(tok)
		if err != nil {
			t.Fatalf("TokenToPiece of byte token %d: %v", tok, err)
		}
		// Not every vocabulary spells a byte token as "<0xNN>"; where it does,
		// the piece has to map back to the same id.
		if !bytePiece.MatchString(piece) {
			continue
		}
		back, err := v.PieceToToken(piece)
		if err != nil {
			t.Fatalf("PieceToToken(%q) for byte token %d: %v", piece, tok, err)
		}
		if back != tok {
			t.Fatalf("byte token %d renders as %q and mapped back to %d", tok, piece, back)
		}
	}
	if found == 0 {
		t.Skipf("the vocabulary of %s has no byte tokens", path)
	}
	t.Logf("%d byte tokens of %d", found, v.Size())
}
