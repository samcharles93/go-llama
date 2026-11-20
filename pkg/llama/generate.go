package llama

/*
#include "llama.h"
#include <stdlib.h>
*/
import "C"
import (
	"context"
	"errors"
	"iter"
	"strings"
)

// StopReason explains why generation ended.
type StopReason int

// Stop reasons reported on Result.
const (
	// StopReasonNone means generation never started or ended without a reason.
	StopReasonNone StopReason = iota
	// StopReasonEOS means the model emitted an end-of-generation token.
	StopReasonEOS
	// StopReasonStopToken means one of CompletionOptions.StopTokens appeared.
	StopReasonStopToken
	// StopReasonStopSequence means one of CompletionOptions.Stop appeared in the
	// text.
	StopReasonStopSequence
	// StopReasonMaxTokens means MaxTokens was reached.
	StopReasonMaxTokens
	// StopReasonContextFull means the KV cache had no room for another token.
	StopReasonContextFull
	// StopReasonCancelled means the caller's context was cancelled or the
	// consuming loop stopped early.
	StopReasonCancelled
)

func (r StopReason) String() string {
	switch r {
	case StopReasonEOS:
		return "eos"
	case StopReasonStopToken:
		return "stop-token"
	case StopReasonStopSequence:
		return "stop-sequence"
	case StopReasonMaxTokens:
		return "max-tokens"
	case StopReasonContextFull:
		return "context-full"
	case StopReasonCancelled:
		return "cancelled"
	default:
		return "none"
	}
}

// Result is one generation: the text, the tokens behind it, why it stopped, and
// how fast it ran.
type Result struct {
	// Text is the generated text with stop sequences removed.
	Text string
	// Tokens are the generated token ids, including the stop token when one
	// ended generation.
	Tokens []Token
	// StopReason explains why generation ended.
	StopReason StopReason
	// PromptTokens counts the tokens in the prompt, including any that were
	// reused from the previous turn's KV cache.
	PromptTokens int
	// ReusedTokens counts prompt tokens served from the existing KV cache
	// instead of being evaluated again.
	ReusedTokens int
	// Metrics reports llama.cpp's timings for this generation.
	Metrics Metrics
}

// Generate runs a prompt to completion and returns the whole reply.
//
// Cancelling ctx stops generation at the next token and returns the partial
// result together with an error that satisfies errors.Is(err,
// context.Canceled).
func (c *Context) Generate(ctx context.Context, prompt string, opts CompletionOptions) (Result, error) {
	var sb strings.Builder
	res, err := c.generate(ctx, prompt, opts, func(piece string) bool {
		sb.WriteString(piece)
		return true
	})
	res.Text = sb.String()
	return res, err
}

// Stream returns an iterator over the generated pieces. Break out of the loop to
// stop generation; the context is left consistent for the next call, and the KV
// cache keeps what was generated so far.
//
//	for piece, err := range c.Stream(ctx, prompt, opts) {
//		if err != nil { return err }
//		fmt.Print(piece)
//	}
//
// Pieces are not lines or words: they are token-aligned and can be partial
// UTF-8-free text, so write them straight out without trimming.
func (c *Context) Stream(ctx context.Context, prompt string, opts CompletionOptions) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		var failed bool
		_, err := c.generate(ctx, prompt, opts, func(piece string) bool {
			if !yield(piece, nil) {
				return false
			}
			return true
		})
		if err != nil && !failed {
			failed = true
			yield("", err)
		}
	}
}

// generate is the single generation engine behind Generate and Stream. emit
// returns false when the consumer wants to stop.
func (c *Context) generate(ctx context.Context, prompt string, opts CompletionOptions, emit func(string) bool) (Result, error) {
	res := Result{}
	if c == nil || c.closed.Load() || c.ptr == nil || c.model == nil {
		return res, ErrClosed
	}
	if err := opts.validate(); err != nil {
		return res, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		res.StopReason = StopReasonCancelled
		return res, wrap(ErrDecode, "generate", err)
	}

	tokens, err := c.model.vocab.Tokenize(prompt, opts.AddBOS)
	if err != nil {
		return res, err
	}
	res.PromptTokens = len(tokens)
	if len(tokens) == 0 {
		return res, errorf(ErrDecode, "prompt produced no tokens")
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ptr == nil {
		return res, ErrClosed
	}

	sampler, err := newSamplerFromOptions(c.model.vocab, opts)
	if err != nil {
		return res, err
	}
	defer sampler.Close()

	C.llama_perf_context_reset(c.ptr)
	reused, err := c.syncKV(tokens)
	res.ReusedTokens = reused
	if err != nil {
		return res, err
	}

	stop := newStopMatcher(opts.Stop)
	extraStops := make(map[Token]struct{}, len(opts.StopTokens))
	for _, t := range opts.StopTokens {
		extraStops[t] = struct{}{}
	}
	eos := c.model.vocab.EOS()

	var text strings.Builder
	for {
		// Close sets the flag before waiting for the lock, so a reply long
		// enough to outlive its owner stops here instead of blocking Close.
		if c.closed.Load() {
			res.StopReason = StopReasonCancelled
			res.Text = text.String()
			res.Metrics = c.metrics()
			return res, ErrClosed
		}
		if err := ctx.Err(); err != nil {
			res.StopReason = StopReasonCancelled
			res.Metrics = c.metrics()
			return res, wrap(ErrDecode, "generate", err)
		}

		token := sampler.Sample(c, -1)
		if token < 0 {
			return res, errorf(ErrDecode, "sampler returned an invalid token")
		}
		sampler.Accept(token)

		if _, ok := extraStops[token]; ok {
			res.Tokens = append(res.Tokens, token)
			res.StopReason = StopReasonStopToken
			break
		}
		if !opts.IgnoreEOS && (token == eos || c.model.vocab.IsEOG(token)) {
			res.Tokens = append(res.Tokens, token)
			res.StopReason = StopReasonEOS
			break
		}

		piece, perr := c.model.vocab.TokenToPiece(token)
		if perr != nil {
			return res, perr
		}
		res.Tokens = append(res.Tokens, token)

		out, hit := stop.Push(piece)
		if out != "" {
			text.WriteString(out)
			if !emit(out) {
				res.StopReason = StopReasonCancelled
				res.Metrics = c.metrics()
				res.Text = text.String()
				return res, nil
			}
		}
		if hit {
			res.StopReason = StopReasonStopSequence
			break
		}

		if opts.MaxTokens > 0 && len(res.Tokens) >= opts.MaxTokens {
			res.StopReason = StopReasonMaxTokens
			break
		}
		if len(c.history) >= c.nCtx {
			res.StopReason = StopReasonContextFull
			break
		}

		c.history = append(c.history, token)
		if err := c.decodeOne(token, int32(len(c.history)-1)); err != nil {
			return res, err
		}
	}

	if res.StopReason != StopReasonStopSequence && res.StopReason != StopReasonCancelled {
		if tail := stop.Flush(); tail != "" {
			text.WriteString(tail)
			emit(tail)
		}
	}
	res.Text = text.String()
	res.Metrics = c.metrics()
	return res, nil
}

// syncKV aligns the KV cache with tokens, keeping the longest common prefix so a
// multi-turn chat only evaluates the new turn instead of the whole
// conversation.
func (c *Context) syncKV(tokens []Token) (int, error) {
	reused := commonPrefix(c.history, tokens)

	if reused < len(c.history) {
		mem := C.llama_get_memory(c.ptr)
		if !bool(C.llama_memory_seq_rm(mem, 0, C.llama_pos(reused), -1)) {
			C.llama_memory_clear(mem, true)
			reused = 0
		}
		c.history = c.history[:reused]
	}
	if reused == len(tokens) {
		return reused, nil
	}
	if len(tokens) > c.nCtx {
		return reused, errorf(ErrContextFull, "prompt is %d tokens, context holds %d", len(tokens), c.nCtx)
	}

	for start := reused; start < len(tokens); start += c.nBatch {
		end := start + c.nBatch
		if end > len(tokens) {
			end = len(tokens)
		}
		if err := c.decodeRange(tokens[start:end], int32(start), end == len(tokens), false); err != nil {
			return reused, err
		}
		c.history = append(c.history, tokens[start:end]...)
	}
	return reused, nil
}

// decodeRange evaluates a slice of tokens starting at absolute position pos.
// logitsLast asks for output on the final token, which is all sampling needs;
// logitsAll asks for output on every token, which is what pooling embeddings
// require.
func (c *Context) decodeRange(tokens []Token, pos int32, logitsLast, logitsAll bool) error {
	b, err := c.scratchBatch(len(tokens))
	if err != nil {
		return err
	}
	b.Reset()
	for i, t := range tokens {
		want := logitsAll || (logitsLast && i == len(tokens)-1)
		if err := b.AddOne(t, pos+int32(i), want); err != nil {
			return err
		}
	}
	return c.decode(b)
}

// decodeOne evaluates a single already-cached token to get its logits.
func (c *Context) decodeOne(token Token, pos int32) error {
	b, err := c.scratchBatch(1)
	if err != nil {
		return err
	}
	b.Reset()
	if err := b.AddOne(token, pos, true); err != nil {
		return err
	}
	return c.decode(b)
}

// scratchBatch returns the context's reusable batch, growing it when a call needs
// more room than the current one provides.
func (c *Context) scratchBatch(n int) (*Batch, error) {
	if c.batch != nil && c.batch.Cap() >= n {
		return c.batch, nil
	}
	capacity := max(n, c.nBatch)
	if c.batch != nil {
		_ = c.batch.Close()
		c.batch = nil
	}
	b, err := NewBatch(int32(capacity), 1)
	if err != nil {
		return nil, err
	}
	c.batch = b
	return b, nil
}

// decode hands a batch to llama.cpp, translating its status codes into errors.
func (c *Context) decode(b *Batch) error {
	rc := C.llama_decode(c.ptr, b.batch)
	switch rc {
	case 0:
		return nil
	case 1:
		return errorf(ErrDecode, "llama_decode could not find a KV cache slot (context full at %d tokens)", c.nCtx)
	case 2:
		return errorf(ErrDecode, "llama_decode could not allocate a compute buffer")
	case -1:
		return errorf(ErrDecode, "llama_decode rejected the batch (invalid token id or position)")
	default:
		return errorf(ErrDecode, "llama_decode failed with code %d", rc)
	}
}

// Decode evaluates a batch the caller built, which is the low-level escape hatch
// for batching several sequences by hand. The context's token accounting is not
// updated, so Reset before the next Generate.
func (c *Context) Decode(b *Batch) error {
	if c == nil || c.closed.Load() {
		return ErrClosed
	}
	if b == nil {
		return errors.New("llama: nil batch")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.decode(b)
}

func commonPrefix(a, b []Token) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	for i := 0; i < n; i++ {
		if a[i] != b[i] {
			return i
		}
	}
	return n
}

// Chat generates a reply to a message list using the model's own chat template.
func (c *Context) Chat(ctx context.Context, messages []ChatMessage, opts CompletionOptions) (Result, error) {
	prompt, err := c.model.ApplyChatTemplate(messages, true)
	if err != nil {
		return Result{}, err
	}
	return c.Generate(ctx, prompt, opts)
}

// ChatStream is Chat as a streaming iterator.
func (c *Context) ChatStream(ctx context.Context, messages []ChatMessage, opts CompletionOptions) (iter.Seq2[string, error], error) {
	prompt, err := c.model.ApplyChatTemplate(messages, true)
	if err != nil {
		return nil, err
	}
	return c.Stream(ctx, prompt, opts), nil
}
