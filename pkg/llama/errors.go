package llama

import (
	"errors"
	"fmt"
	"strings"
)

// Errors returned by this package. Callers can test for them with errors.Is;
// every error also carries the operation that failed, so a bare ctx.Err()
// never reaches the caller without its context.
var (
	// ErrModelLoad reports a model file that could not be opened or parsed.
	ErrModelLoad = errors.New("llama: load model")
	// ErrContextCreate reports a context that could not be allocated, usually
	// because the requested context size does not fit in memory.
	ErrContextCreate = errors.New("llama: create context")
	// ErrDecode reports a llama_decode failure: the batch did not fit in the
	// context, or the backend rejected it.
	ErrDecode = errors.New("llama: decode")
	// ErrTokenize reports a tokenizer failure.
	ErrTokenize = errors.New("llama: tokenize")
	// ErrTemplate reports a chat template that could not be applied.
	ErrTemplate = errors.New("llama: apply chat template")
	// ErrEmbedding reports an embedding request against a context that has no
	// embeddings enabled, or a model with no embedding output.
	ErrEmbedding = errors.New("llama: embeddings")
	// ErrGrammar reports a GBNF grammar llama.cpp rejected at parse time.
	ErrGrammar = errors.New("llama: grammar")
	// ErrContextFull is returned when generation cannot continue because the
	// context window is exhausted.
	ErrContextFull = errors.New("llama: context window exhausted")
	// ErrClosed is returned by a Model or Context that has been closed.
	ErrClosed = errors.New("llama: use after close")
	// ErrNotInitialized is returned when the backend was never initialised.
	ErrNotInitialized = errors.New("llama: backend not initialised, call llama.Init")
)

// wrap annotates err with the operation that failed while keeping errors.Is
// working against both the sentinel and any wrapped cause such as
// context.Canceled.
func wrap(kind error, op string, err error) error {
	if err == nil {
		return nil
	}
	return fmt.Errorf("%w: %s: %w", kind, op, err)
}

// opError is used where a formatted message is more useful than a cause chain.
type opError struct {
	kind error
	msg  string
}

func (e *opError) Error() string { return e.kind.Error() + ": " + e.msg }
func (e *opError) Unwrap() error { return e.kind }

func errorf(kind error, format string, args ...any) error {
	return &opError{kind: kind, msg: fmt.Sprintf(format, args...)}
}

// errUnknownEncoding is returned when a backend function reports failure
// without a usable message.
func cFail(kind error, what string, code int32) error {
	if code == 0 {
		return nil
	}
	return errorf(kind, "%s failed with code %d", what, code)
}

func trimNul(s string) string { return strings.TrimRight(s, "\x00") }
