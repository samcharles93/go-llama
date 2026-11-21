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
	if c == nil || c.closed.Load() || c.model == nil {
		return Result{}, ErrClosed
	}
	tokens, err := c.model.vocab.Tokenize(prompt, opts.AddBOS)
	if err != nil {
		return Result{}, err
	}
	return c.GenerateTokens(ctx, tokens, opts)
}

// GenerateTokens is Generate for a prompt that is already tokenized, which is
// what a caller wants when it manages the token stream itself, for example to
// continue exactly from Context.History without paying for a re-tokenization
// that might merge differently.
func (c *Context) GenerateTokens(ctx context.Context, prompt []Token, opts CompletionOptions) (Result, error) {
	var sb strings.Builder
	res, err := c.generateTokens(ctx, prompt, opts, func(piece string) bool {
		sb.WriteString(piece)
		return true
	})
	res.Text = sb.String()
	return res, err
}

// StreamTokens is Stream for an already tokenized prompt.
func (c *Context) StreamTokens(ctx context.Context, prompt []Token, opts CompletionOptions) iter.Seq2[string, error] {
	return func(yield func(string, error) bool) {
		_, err := c.generateTokens(ctx, prompt, opts, func(piece string) bool {
			return yield(piece, nil)
		})
		if err != nil {
			yield("", err)
		}
	}
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
	if c == nil || c.closed.Load() || c.model == nil {
		return func(yield func(string, error) bool) { yield("", ErrClosed) }
	}
	tokens, err := c.model.vocab.Tokenize(prompt, opts.AddBOS)
	if err != nil {
		return func(yield func(string, error) bool) { yield("", err) }
	}
	return c.StreamTokens(ctx, tokens, opts)
}

// generateTokens is the single generation engine behind Generate, Stream and
// their token-typed variants. emit returns false when the consumer wants to stop.
func (c *Context) generateTokens(ctx context.Context, tokens []Token, opts CompletionOptions, emit func(string) bool) (Result, error) {
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
	defer func() { _ = sampler.Close() }()

	C.llama_perf_context_reset(c.ptr)
	stopWatch := c.watchCancel(ctx)
	defer stopWatch()

	// finish fills in the fields every exit shares, so a cancelled, closed or
	// failed generation reports the same partial result a completed one would.
	var text strings.Builder
	finish := func(reason StopReason, err error) (Result, error) {
		if reason != StopReasonNone {
			res.StopReason = reason
		}
		res.Text = text.String()
		res.Metrics = c.metrics()
		return res, err
	}
	// cancelled turns a decode failure caused by cancellation into the
	// documented context.Canceled error. llama.cpp reports an abort as a plain
	// status code, so the Go context is the only record of why it happened.
	cancelled := func(err error) (Result, error) {
		if cerr := ctx.Err(); cerr != nil {
			return finish(StopReasonCancelled, wrap(ErrDecode, "generate", cerr))
		}
		return finish(StopReasonNone, err)
	}

	reused, err := c.syncKV(ctx, tokens)
	res.ReusedTokens = reused
	if err != nil {
		return cancelled(err)
	}

	stop := newStopMatcher(opts.Stop)
	extraStops := make(map[Token]struct{}, len(opts.StopTokens))
	for _, t := range opts.StopTokens {
		extraStops[t] = struct{}{}
	}
	eos := c.model.vocab.EOS()
	// Reused for every token: the candidate array is one entry per vocabulary
	// item, so reallocating it per token would be the slowest part of sampling.
	cands, err := newCandidateBuffer(c.model.vocab.Size())
	if err != nil {
		return res, err
	}
	defer cands.free()

	for {
		// Close sets the flag before waiting for the lock, so a reply long
		// enough to outlive its owner stops here instead of blocking Close.
		if c.closed.Load() {
			return finish(StopReasonCancelled, ErrClosed)
		}
		if err := ctx.Err(); err != nil {
			return finish(StopReasonCancelled, wrap(ErrDecode, "generate", err))
		}

		token, serr := sampler.sampleWith(c, -1, cands)
		if serr != nil {
			return finish(StopReasonNone, serr)
		}

		// Stopping tokens are checked before Accept, the way llama.cpp's own
		// examples do it. Feeding an end-of-generation token to a grammar sampler
		// that has already reached its end drives llama.cpp into
		// "Unexpected empty grammar stack", which throws a C++ exception and
		// aborts the process.
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
		sampler.Accept(token)

		piece, perr := c.model.vocab.TokenToPiece(token)
		if perr != nil {
			return finish(StopReasonNone, perr)
		}
		res.Tokens = append(res.Tokens, token)

		out, hit := stop.Push(piece)
		if out != "" {
			text.WriteString(out)
			if !emit(out) {
				return finish(StopReasonCancelled, nil)
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
			c.rollback(len(c.history) - 1)
			return cancelled(err)
		}
	}

	if res.StopReason != StopReasonStopSequence {
		if tail := stop.Flush(); tail != "" {
			text.WriteString(tail)
			emit(tail)
		}
	}
	return finish(StopReasonNone, nil)
}

// syncKV aligns the KV cache with tokens, keeping the longest common prefix so a
// multi-turn chat only evaluates the new turn instead of the whole
// conversation.
//
// A prompt longer than the batch size is decoded in chunks, and ctx is checked
// between them, so cancellation interrupts a long prompt even on backends that
// do not poll the abort flag mid-graph.
func (c *Context) syncKV(ctx context.Context, tokens []Token) (int, error) {
	reused := commonPrefix(c.history, tokens)

	// The caller is about to sample from the last position, so a decode has to
	// have run for it. An exact cache hit normally skips decoding entirely, which
	// is safe only while the logits from the previous decode are still current;
	// after LoadState they are not.
	if reused == len(tokens) && !c.logitsValid {
		reused--
		if reused < 0 {
			reused = 0
		}
	}

	if reused < len(c.history) {
		// Recurrent and hybrid models keep one state per sequence rather than a
		// cell per position, so llama.cpp refuses to drop their tail. Going
		// straight to a full clear saves a call that can only fail and keeps the
		// reported reuse honest.
		mem := C.llama_get_memory(c.ptr)
		trimmed := false
		if !c.model.IsRecurrent() && !c.model.IsHybrid() {
			trimmed = bool(C.llama_memory_seq_rm(mem, 0, C.llama_pos(reused), -1))
		}
		if !trimmed {
			C.llama_memory_clear(mem, true)
			reused = 0
		}
		c.history = c.history[:reused]
		c.logitsValid = false
	}
	if reused == len(tokens) {
		return reused, nil
	}
	if len(tokens) > c.nCtx {
		return reused, errorf(ErrContextFull, "prompt is %d tokens, context holds %d", len(tokens), c.nCtx)
	}

	for start := reused; start < len(tokens); start += c.nBatch {
		if err := ctx.Err(); err != nil {
			return reused, wrap(ErrDecode, "evaluate prompt", err)
		}
		end := min(start+c.nBatch, len(tokens))
		if err := c.decodeRange(tokens[start:end], int32(start), end == len(tokens), false); err != nil {
			c.rollback(len(c.history))
			return reused, err
		}
		c.history = append(c.history, tokens[start:end]...)
	}
	return reused, nil
}

// rollback makes the KV cache hold exactly the first n tokens of the history
// again. A decode that fails part way, including one aborted by cancellation,
// leaves the micro-batches it already processed in the cache; without this the
// cache would hold positions the history does not know about, and the next
// generation would decode on top of them.
func (c *Context) rollback(n int) {
	n = max(0, min(n, len(c.history)))
	mem := C.llama_get_memory(c.ptr)
	trimmed := false
	if !c.model.IsRecurrent() && !c.model.IsHybrid() {
		trimmed = bool(C.llama_memory_seq_rm(mem, 0, C.llama_pos(n), -1))
	}
	if !trimmed {
		C.llama_memory_clear(mem, true)
		n = 0
	}
	c.history = c.history[:n]
	c.logitsValid = false
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
	if err := c.decode(b); err != nil {
		return err
	}
	c.logitsValid = logitsAll || logitsLast
	return nil
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
	if err := c.decode(b); err != nil {
		return err
	}
	c.logitsValid = true
	return nil
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
//
// llama.cpp documents the return values as 0 success, 1 no KV slot, 2 aborted,
// -1 invalid batch, and anything below -1 a fatal error after which the memory
// state is left inconsistent. A fatal error therefore clears the cache and the
// history, so the next generation starts from a known state instead of failing
// forever.
func (c *Context) decode(b *Batch) error {
	rc := C.llama_decode(c.ptr, b.batch)
	switch rc {
	case 0:
		return nil
	case 1:
		return errorf(ErrContextFull, "llama_decode found no KV cache slot for %d tokens (context holds %d)", b.Len(), c.nCtx)
	case 2:
		return errorf(ErrDecode, "llama_decode was aborted (context cancelled)")
	case -1:
		return errorf(ErrDecode, "llama_decode rejected the batch (invalid token id or position)")
	default:
		C.llama_memory_clear(C.llama_get_memory(c.ptr), true)
		c.history = c.history[:0]
		c.logitsValid = false
		return errorf(ErrDecode, "llama_decode failed with code %d; the context was reset", rc)
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
