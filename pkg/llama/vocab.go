package llama

/*
#include "llama.h"
#include <stdlib.h>
*/
import "C"

import (
	"math"
	"unsafe"
)

// Token is a vocabulary id.
type Token int32

// Vocab is a model's tokenizer. It is owned by the model and stays valid for as
// long as the model does.
type Vocab struct {
	vocab *C.struct_llama_vocab
	model *C.struct_llama_model
	nEmbd int
}

func newVocab(m *C.struct_llama_model) *Vocab {
	return &Vocab{
		vocab: C.llama_model_get_vocab(m),
		model: m,
		nEmbd: int(C.llama_model_n_embd_out(m)),
	}
}

// Size reports the vocabulary size.
func (v *Vocab) Size() int { return int(C.llama_vocab_n_tokens(v.vocab)) }

// BOS is the beginning-of-sequence token, or a negative value if the model has
// none.
func (v *Vocab) BOS() Token { return Token(C.llama_vocab_bos(v.vocab)) }

// EOS is the end-of-sequence token, or a negative value if the model has none.
func (v *Vocab) EOS() Token { return Token(C.llama_vocab_eos(v.vocab)) }

// IsEOG reports whether a token ends generation. For chat models this covers
// every end-of-generation token, not just EOS.
func (v *Vocab) IsEOG(t Token) bool { return bool(C.llama_vocab_is_eog(v.vocab, C.llama_token(t))) }

// IsControl reports whether t is a control token: a formatting marker such as
// "<|im_start|>" that carries no text of its own.
//
// An id outside the vocabulary reports false, because llama.cpp indexes its
// attribute table without bounds checking.
func (v *Vocab) IsControl(t Token) bool {
	if !v.known(t) {
		return false
	}
	return bool(C.llama_vocab_is_control(v.vocab, C.llama_token(t)))
}

// IsByte reports whether t is a byte token, the vocabulary's carrier for one
// byte of a byte-level BPE sequence.
//
// llama.h exposes no llama_vocab_is_byte, so this reads the LLAMA_TOKEN_ATTR_BYTE
// attribute that llama.cpp's own predicate tests. An id outside the vocabulary
// reports false.
func (v *Vocab) IsByte(t Token) bool {
	if !v.known(t) {
		return false
	}
	return C.llama_vocab_get_attr(v.vocab, C.llama_token(t))&C.LLAMA_TOKEN_ATTR_BYTE != 0
}

// known reports whether t can be looked up in llama.cpp's token tables.
func (v *Vocab) known(t Token) bool { return t >= 0 && int(t) < v.Size() }

// Tokenize converts text to tokens.
//
// addSpecial prepends BOS (and any other special tokens the model wants at the
// start of a prompt). Special tokens spelled out in the text, such as
// "<|im_end|>", are always parsed as special tokens so chat templates work. An
// empty text returns no tokens, even with addSpecial set.
//
// llama_tokenize reports the required buffer size as a negative return value;
// the loop here grows the buffer until it fits, where the previous
// implementation could index a negative-length slice and panic.
func (v *Vocab) Tokenize(text string, addSpecial bool) ([]Token, error) {
	if text == "" {
		return nil, nil
	}
	if len(text) > math.MaxInt32 {
		return nil, errorf(ErrTokenize, "text of %d bytes is too long to tokenize", len(text))
	}
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))
	textLen := C.int32_t(len(text))

	// llama_tokenize needs room for every token plus any specials it adds, and
	// the buffer is always non-empty so indexing it cannot panic.
	buf := make([]C.llama_token, len(text)+8)
	for {
		n := C.llama_tokenize(v.vocab, cText, textLen, &buf[0], C.int32_t(len(buf)), C.bool(addSpecial), true)
		if n > 0 {
			out := make([]Token, n)
			for i := range out {
				out[i] = Token(buf[i])
			}
			return out, nil
		}
		if n == math.MinInt32 {
			// Documented overflow marker: the result does not fit in int32.
			return nil, errorf(ErrTokenize, "text of %d bytes is too long to tokenize", len(text))
		}
		if n == 0 {
			return nil, nil
		}
		need := int(-n)
		if need <= len(buf) {
			return nil, errorf(ErrTokenize, "llama_tokenize asked for %d tokens and then rejected them", need)
		}
		buf = make([]C.llama_token, need)
	}
}

// PieceToToken maps a token's text back to its id. The piece is tokenized, so it
// has to be the text of exactly one token: a piece that splits into several
// tokens, or that matches nothing at all, is an error rather than a guess.
//
// A special token spelled out in full, such as "<|im_end|>", resolves to that
// special token, which is how a template's marker strings can be looked up. The
// piece is text, not the raw vocabulary entry, so " word" is accepted where the
// stored entry is "\xe2\x96\x81word".
func (v *Vocab) PieceToToken(piece string) (Token, error) {
	tokens, err := v.Tokenize(piece, false)
	if err != nil {
		return 0, err
	}
	if len(tokens) != 1 {
		return 0, errorf(ErrTokenize, "%q is not one token: it tokenizes to %d tokens", piece, len(tokens))
	}
	return tokens[0], nil
}

// TokenToPiece converts one token back to text. Special tokens render as their
// literal spelling.
func (v *Vocab) TokenToPiece(t Token) (string, error) {
	if !v.known(t) {
		return "", errorf(ErrTokenize, "token %d is outside the vocabulary of %d tokens", t, v.Size())
	}
	buf := make([]byte, 32)
	for {
		n := C.llama_token_to_piece(v.vocab, C.llama_token(t), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)), 0, true)
		if n >= 0 {
			return string(buf[:n]), nil
		}
		need := int(-n)
		if need <= len(buf) || need > 1<<20 {
			return "", errorf(ErrTokenize, "cannot render token %d", t)
		}
		buf = make([]byte, need)
	}
}

// Detokenize converts tokens back to text, dropping special tokens.
func (v *Vocab) Detokenize(tokens []Token) (string, error) {
	if len(tokens) == 0 {
		return "", nil
	}
	ctoks := make([]C.llama_token, len(tokens))
	for i, t := range tokens {
		if !v.known(t) {
			return "", errorf(ErrTokenize, "token %d at index %d is outside the vocabulary of %d tokens", t, i, v.Size())
		}
		ctoks[i] = C.llama_token(t)
	}
	buf := make([]byte, 4*len(tokens)+32)
	for {
		n := C.llama_detokenize(v.vocab, &ctoks[0], C.int32_t(len(ctoks)), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)), true, false)
		if n >= 0 {
			return string(buf[:n]), nil
		}
		need := int(-n)
		if need <= len(buf) || need > 1<<26 {
			return "", errorf(ErrTokenize, "cannot detokenize %d tokens", len(tokens))
		}
		buf = make([]byte, need)
	}
}
