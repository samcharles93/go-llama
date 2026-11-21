package llama

/*
#include "llama.h"
#include <stdlib.h>

// sample_chain applies the sampler chain to the context's logits for one
// position and returns the selected token without accepting it. It exists as C
// because llama_token_data_array points at a candidate array, and a Go array
// nested inside a Go struct cannot be handed to cgo.
//
// The buffer has one entry per vocabulary item and is owned by the caller, so a
// generation loop allocates it once instead of once per token.
static llama_token sample_chain(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx, struct llama_token_data * buf, int32_t n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, idx);
    if (logits == NULL) {
        return -1;
    }
    for (int32_t i = 0; i < n_vocab; i++) {
        buf[i].id    = i;
        buf[i].logit = logits[i];
        buf[i].p     = 0.0f;
    }
    struct llama_token_data_array cur = { buf, (size_t) n_vocab, -1, false };
    llama_sampler_apply(smpl, &cur);
    if (cur.selected < 0 || cur.selected >= n_vocab) {
        return -1;
    }
    return buf[cur.selected].id;
}

static struct llama_token_data * alloc_candidates(int32_t n_vocab) {
    return (struct llama_token_data *) calloc((size_t) n_vocab, sizeof(struct llama_token_data));
}
*/
import "C"

import (
	"sync/atomic"
	"unsafe"
)

// Sampler is a llama.cpp sampler chain. Build a chain with NewSamplerChain and
// Add* calls, or let a Context derive one from CompletionOptions.
//
// A Sampler is stateful: call Accept for every token you take from it, or the
// repetition counters drift.
type Sampler struct {
	sampler *C.struct_llama_sampler
	closed  atomic.Bool
}

// NewSamplerChain creates an empty chain. A sampler with no stages returns
// undefined output, so always terminate the chain with AddGreedy or AddDist.
func NewSamplerChain() *Sampler {
	params := C.llama_sampler_chain_default_params()
	return &Sampler{sampler: C.llama_sampler_chain_init(params)}
}

// newSamplerFromOptions builds the chain that CompletionOptions describes.
//
// Stage order mirrors llama.cpp's own default in common/sampling.cpp: penalties,
// DRY, top-n-sigma, top-k, typical, top-p, min-p, XTC, temperature, picker. The
// penalties come first because they have to see the raw distribution, and the
// picker comes last because it returns a token instead of rewriting the
// candidate list. Greedy decoding skips every stochastic stage so Temperature 0
// is genuinely deterministic.
//
// Every failure after the chain exists closes it, so a rejected option set never
// leaves a half-built chain behind.
func newSamplerFromOptions(vocab *Vocab, opts CompletionOptions) (_ *Sampler, err error) {
	if err := opts.validate(); err != nil {
		return nil, err
	}

	s := NewSamplerChain()
	if s.sampler == nil {
		return nil, errorf(ErrDecode, "out of memory allocating sampler")
	}
	defer func() {
		if err != nil {
			_ = s.Close()
		}
	}()

	if g := opts.Grammar; g != "" {
		if vocab == nil {
			return nil, errorf(ErrGrammar, "a grammar needs the model vocabulary")
		}
		cGrammar := C.CString(g)
		defer C.free(unsafe.Pointer(cGrammar))
		cRoot := C.CString("root")
		defer C.free(unsafe.Pointer(cRoot))
		if err := s.add(C.llama_sampler_init_grammar(vocab.vocab, cGrammar, cRoot), ErrGrammar); err != nil {
			return nil, err
		}
	}

	if opts.penaltiesEnabled() {
		repeat := opts.RepeatPenalty
		if repeat <= 0 {
			repeat = 1
		}
		if err := s.add(C.llama_sampler_init_penalties(
			C.int32_t(vocabSize(vocab)), C.int32_t(resolveWindow(opts.repeatLastN(), vocab)),
			C.float(repeat), C.float(opts.FrequencyPenalty), C.float(opts.PresencePenalty),
		), ErrDecode); err != nil {
			return nil, err
		}
	}

	if opts.dryEnabled() {
		if err := s.addDry(vocab, opts); err != nil {
			return nil, err
		}
	}

	if opts.greedy() {
		if err := s.add(C.llama_sampler_init_greedy(), ErrDecode); err != nil {
			return nil, err
		}
		return s, nil
	}

	if opts.TopNSigma > 0 {
		if err := s.add(C.llama_sampler_init_top_n_sigma(C.float(opts.TopNSigma)), ErrDecode); err != nil {
			return nil, err
		}
	}
	if k := clampTopK(opts.TopK, vocabSize(vocab)); k > 0 {
		if err := s.add(C.llama_sampler_init_top_k(C.int32_t(k)), ErrDecode); err != nil {
			return nil, err
		}
	}
	if opts.TypicalP > 0 && opts.TypicalP < 1 {
		if err := s.add(C.llama_sampler_init_typical(C.float(opts.TypicalP), 1), ErrDecode); err != nil {
			return nil, err
		}
	}
	if opts.TopP > 0 && opts.TopP < 1 {
		if err := s.add(C.llama_sampler_init_top_p(C.float(opts.TopP), 1), ErrDecode); err != nil {
			return nil, err
		}
	}
	if opts.MinP > 0 {
		if err := s.add(C.llama_sampler_init_min_p(C.float(opts.MinP), 1), ErrDecode); err != nil {
			return nil, err
		}
	}
	if opts.xtcEnabled() {
		if err := s.add(C.llama_sampler_init_xtc(C.float(opts.XTCProbability), C.float(opts.XTCThreshold), 1, C.uint32_t(opts.Seed)), ErrDecode); err != nil {
			return nil, err
		}
	}
	if err := s.add(C.llama_sampler_init_temp(C.float(opts.Temperature)), ErrDecode); err != nil {
		return nil, err
	}

	if opts.mirostatEnabled() {
		// Mirostat picks the token itself, so it replaces the distribution
		// sampler rather than running alongside it.
		if err := s.add(C.llama_sampler_init_mirostat_v2(C.uint32_t(opts.Seed), C.float(opts.MirostatTau), C.float(opts.MirostatEta)), ErrDecode); err != nil {
			return nil, err
		}
		return s, nil
	}
	if err := s.add(C.llama_sampler_init_dist(C.uint32_t(opts.Seed)), ErrDecode); err != nil {
		return nil, err
	}
	return s, nil
}

// add appends a stage llama.cpp just built. The initialisers only return nil
// when they cannot allocate, and a nil stage in the chain would crash on first
// use, so the failure is reported instead.
func (s *Sampler) add(stage *C.struct_llama_sampler, kind error) error {
	if stage == nil {
		return errorf(kind, "llama.cpp could not allocate a sampler stage")
	}
	C.llama_sampler_chain_add(s.sampler, stage)
	return nil
}

// addDry appends the DRY stage. DRY tokenises its sequence breakers against the
// vocabulary, so it cannot be built without a model.
func (s *Sampler) addDry(vocab *Vocab, opts CompletionOptions) error {
	if vocab == nil || vocab.vocab == nil {
		return errorf(ErrDecode, "DRY needs the model vocabulary")
	}

	breakers := make([]*C.char, len(opts.DrySequenceBreakers))
	for i, b := range opts.DrySequenceBreakers {
		breakers[i] = C.CString(b)
	}
	defer func() {
		for _, b := range breakers {
			C.free(unsafe.Pointer(b))
		}
	}()

	var breakerPtr **C.char
	if len(breakers) > 0 {
		breakerPtr = (**C.char)(unsafe.Pointer(&breakers[0]))
	}

	stage := C.llama_sampler_init_dry(
		vocab.vocab,
		C.float(opts.DryMultiplier), C.float(opts.dryBase()),
		C.int32_t(opts.dryAllowedLength()), C.int32_t(resolveWindow(opts.dryPenaltyLastN(), vocab)),
		breakerPtr, C.size_t(len(breakers)),
	)
	return s.add(stage, ErrDecode)
}

// resolveWindow turns a penalty window into the count llama.cpp expects. This
// llama.cpp clamps a negative window to 0, which disables the stage, so the
// documented -1 for "the whole context" is resolved here: to the model's
// trained context, or MaxDefaultContextSize when there is no model to ask.
func resolveWindow(n int32, vocab *Vocab) int32 {
	if n >= 0 {
		return n
	}
	if vocab != nil && vocab.model != nil {
		if train := int32(C.llama_model_n_ctx_train(vocab.model)); train > 0 {
			return train
		}
	}
	return int32(MaxDefaultContextSize)
}

// vocabSize reports the vocabulary size, or 0 when there is no model to ask.
func vocabSize(v *Vocab) int {
	if v == nil || v.vocab == nil {
		return 0
	}
	return v.Size()
}

// clampTopK keeps TopK inside the vocabulary. llama.cpp truncates its candidate
// list to whatever k it is given, so clamping only saves work; a vocabulary size
// of 0 means the size is unknown and the value is passed through.
func clampTopK(k int32, vocabSize int) int32 {
	if k <= 0 || vocabSize <= 0 {
		return k
	}
	if int64(k) > int64(vocabSize) {
		return int32(vocabSize)
	}
	return k
}

// AddTopK keeps only the k most likely tokens.
func (s *Sampler) AddTopK(k int32) *Sampler {
	if s.sampler != nil && k > 0 {
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_top_k(C.int32_t(k)))
	}
	return s
}

// AddTopP keeps the smallest token set whose probabilities sum to p.
func (s *Sampler) AddTopP(p float32, minKeep int) *Sampler {
	if s.sampler != nil && p > 0 && p < 1 {
		if minKeep < 1 {
			minKeep = 1
		}
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_top_p(C.float(p), C.size_t(minKeep)))
	}
	return s
}

// AddMinP drops tokens less than p times as likely as the best one.
func (s *Sampler) AddMinP(p float32, minKeep int) *Sampler {
	if s.sampler != nil && p > 0 {
		if minKeep < 1 {
			minKeep = 1
		}
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_min_p(C.float(p), C.size_t(minKeep)))
	}
	return s
}

// AddTypical applies locally typical sampling.
func (s *Sampler) AddTypical(p float32, minKeep int) *Sampler {
	if s.sampler != nil && p > 0 && p < 1 {
		if minKeep < 1 {
			minKeep = 1
		}
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_typical(C.float(p), C.size_t(minKeep)))
	}
	return s
}

// AddTopNSigma keeps tokens within n standard deviations of the best logit.
// Zero or a negative value leaves the chain alone.
func (s *Sampler) AddTopNSigma(n float32) *Sampler {
	if s.sampler != nil && n > 0 {
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_top_n_sigma(C.float(n)))
	}
	return s
}

// AddXTC drops the most likely tokens with probability p and samples from what
// is left, which breaks repetitive loops. A threshold above 0.5 disables the
// stage, matching llama.cpp's own cut-off.
func (s *Sampler) AddXTC(p, threshold float32, minKeep int, seed uint32) *Sampler {
	if s.sampler == nil || p <= 0 || p > 1 || threshold > 0.5 {
		return s
	}
	if minKeep < 1 {
		minKeep = 1
	}
	C.llama_sampler_chain_add(s.sampler,
		C.llama_sampler_init_xtc(C.float(p), C.float(threshold), C.size_t(minKeep), C.uint32_t(seed)))
	return s
}

// AddMirostatV2 holds the output at a target surprise level. It picks the token
// itself, so it must be the last stage in the chain.
func (s *Sampler) AddMirostatV2(tau, eta float32, seed uint32) *Sampler {
	if s.sampler == nil || tau <= 0 || eta <= 0 {
		return s
	}
	C.llama_sampler_chain_add(s.sampler,
		C.llama_sampler_init_mirostat_v2(C.uint32_t(seed), C.float(tau), C.float(eta)))
	return s
}

// AddDRY penalises repeats of sequences already in the window. lastN is the
// window in tokens: 0 uses the default of 64 and -1 the whole context. breakers
// restart the count, for example "\n". DRY tokenises the breakers, so the stage
// is skipped when vocab is nil.
func (s *Sampler) AddDRY(vocab *Vocab, multiplier, base float32, allowedLength, lastN int32, breakers []string) *Sampler {
	if s.sampler == nil || multiplier == 0 || vocab == nil || vocab.vocab == nil {
		return s
	}
	_ = s.addDry(vocab, CompletionOptions{
		DryMultiplier:       multiplier,
		DryBase:             base,
		DryAllowedLength:    allowedLength,
		DryPenaltyLastN:     lastN,
		DrySequenceBreakers: breakers,
	})
	return s
}

// AddTemp scales the distribution. Temperatures at or below zero fall back to
// greedy inside llama.cpp, which is why the option builder skips this stage.
func (s *Sampler) AddTemp(t float32) *Sampler {
	if s.sampler != nil && t > 0 {
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_temp(C.float(t)))
	}
	return s
}

// AddPenalties discourages tokens already present in the last lastN positions.
// repeat is the divisor penalty (1 disables it); frequency and presence are the
// additive OpenAI-style penalties.
func (s *Sampler) AddPenalties(lastN int32, repeat, frequency, presence float32) *Sampler {
	if s.sampler == nil || (repeat <= 1 && frequency == 0 && presence == 0) {
		return s
	}
	if lastN == 0 {
		lastN = 64
	}
	// A chain built by hand has no model, so "whole context" falls back to
	// MaxDefaultContextSize, and n_vocab, which only sizes llama.cpp's
	// GPU-side sampling path, is left at 0.
	C.llama_sampler_chain_add(s.sampler,
		C.llama_sampler_init_penalties(0, C.int32_t(resolveWindow(lastN, nil)), C.float(repeat), C.float(frequency), C.float(presence)))
	return s
}

// AddGreedy always takes the most likely token, with no temperature or top-k
// stage. Terminates a deterministic chain.
func (s *Sampler) AddGreedy() *Sampler {
	if s.sampler != nil {
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_greedy())
	}
	return s
}

// AddDist samples from the distribution. Terminates a stochastic chain.
func (s *Sampler) AddDist(seed uint32) *Sampler {
	if s.sampler != nil {
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_dist(C.uint32_t(seed)))
	}
	return s
}

// candidateBuffer holds one llama_token_data per vocabulary item in C memory,
// which is what the sample path hands to llama_sampler_apply.
type candidateBuffer struct {
	ptr    *C.struct_llama_token_data
	nVocab int32
}

func newCandidateBuffer(nVocab int) (*candidateBuffer, error) {
	if nVocab <= 0 {
		return nil, errorf(ErrDecode, "model vocabulary is empty")
	}
	ptr := C.alloc_candidates(C.int32_t(nVocab))
	if ptr == nil {
		return nil, errorf(ErrDecode, "out of memory allocating %d sampling candidates", nVocab)
	}
	return &candidateBuffer{ptr: ptr, nVocab: int32(nVocab)}, nil
}

func (b *candidateBuffer) free() {
	if b != nil && b.ptr != nil {
		C.free(unsafe.Pointer(b.ptr))
		b.ptr = nil
	}
}

// Sample picks the next token for the context's last decoded position, without
// feeding it back into the chain. Call Accept for every token you take, or the
// repetition and grammar state drift.
//
// It deliberately does not use llama_sampler_sample: that function applies the
// chain, picks a token and accepts it in one call, so a caller following the
// documented sample-then-accept pattern would accept every token twice. Double
// acceptance double-counts the penalties and drives a grammar sampler past the
// end of its grammar, where llama.cpp throws a C++ exception that aborts the
// process. Apply, select and accept stay separate here for that reason.
func (s *Sampler) Sample(c *Context, idx int32) Token {
	if c == nil || c.model == nil {
		return -1
	}
	buf, err := newCandidateBuffer(c.model.vocab.Size())
	if err != nil {
		return -1
	}
	defer buf.free()
	token, err := s.sampleWith(c, idx, buf)
	if err != nil {
		return -1
	}
	return token
}

// sampleWith is Sample with a caller-supplied candidate buffer, so a generation
// loop does not reallocate a vocabulary-sized array for every token.
func (s *Sampler) sampleWith(c *Context, idx int32, buf *candidateBuffer) (Token, error) {
	if s == nil || s.sampler == nil {
		return -1, errorf(ErrDecode, "sampler is closed")
	}
	if c == nil || c.ptr == nil || c.model == nil {
		return -1, ErrClosed
	}
	if buf == nil || buf.ptr == nil {
		return -1, errorf(ErrDecode, "no sampling buffer")
	}
	if C.llama_get_logits_ith(c.ptr, C.int32_t(idx)) == nil {
		return -1, errorf(ErrDecode, "no logits for position %d (was the prompt decoded?)", idx)
	}

	token := C.sample_chain(s.sampler, c.ptr, C.int32_t(idx), buf.ptr, C.int32_t(buf.nVocab))
	if token < 0 {
		return -1, errorf(ErrDecode, "the sampler selected no token, so a grammar or filter excluded every candidate")
	}
	return Token(token), nil
}

// Accept feeds a token back into the chain so penalties and grammar state stay
// correct.
func (s *Sampler) Accept(t Token) {
	if s != nil && s.sampler != nil {
		C.llama_sampler_accept(s.sampler, C.llama_token(t))
	}
}

// Stages reports llama.cpp's name for each stage in the chain, in the order it
// applies them. It is how a caller, or a test, checks that a chain matches the
// documented order instead of inferring it from sampled tokens.
func (s *Sampler) Stages() []string {
	if s == nil || s.sampler == nil {
		return nil
	}
	n := int(C.llama_sampler_chain_n(s.sampler))
	stages := make([]string, 0, n)
	for i := range n {
		stage := C.llama_sampler_chain_get(s.sampler, C.int32_t(i))
		if stage == nil {
			continue
		}
		stages = append(stages, C.GoString(C.llama_sampler_name(stage)))
	}
	return stages
}

// Seed reports the seed the chain actually uses, which matters when it was
// picked at random.
func (s *Sampler) Seed() uint32 {
	if s == nil || s.sampler == nil {
		return 0
	}
	return uint32(C.llama_sampler_get_seed(s.sampler))
}

// Close frees the chain. Safe to call more than once.
func (s *Sampler) Close() error {
	if s == nil || s.closed.Swap(true) {
		return nil
	}
	if s.sampler != nil {
		C.llama_sampler_free(s.sampler)
		s.sampler = nil
	}
	return nil
}

// Free is Close, kept for callers written against the old API.
//
// Deprecated: use Close.
func (s *Sampler) Free() { _ = s.Close() }
