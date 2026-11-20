package llama

import (
	"errors"
	"testing"
)

// The tokenizer needs a model file for anything that reaches llama.cpp, so the
// model-backed tokenizer tests live in vocab_integration_test.go. The cases here
// exercise the paths that return before llama.cpp is touched.

func TestVocabWithoutModelHandlesEmptyAndOutOfRangeInputs(t *testing.T) {
	v := &Vocab{} // no model behind it, so nothing below may reach llama.cpp

	if tokens, err := v.Tokenize("", true); err != nil || len(tokens) != 0 {
		t.Fatalf("Tokenize(\"\") = %v, %v; want no tokens and no error", tokens, err)
	}
	if got, err := v.Detokenize(nil); err != nil || got != "" {
		t.Fatalf("Detokenize(nil) = %q, %v; want an empty string and no error", got, err)
	}
	if _, err := v.PieceToToken(""); !errors.Is(err, ErrTokenize) {
		t.Fatalf("PieceToToken(\"\") = %v, want ErrTokenize", err)
	}
	if v.IsControl(-1) || v.IsByte(-1) {
		t.Fatal("a negative token id was reported as a control or byte token")
	}
}
