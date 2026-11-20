package llama

/*
#include "llama.h"
#include <stdlib.h>
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
// Order follows llama.cpp's own recommendation: penalties first (they must see
// the raw distribution), then the truncation stages from most to least
// aggressive, then temperature, then the token picker. Greedy decoding skips
// every stochastic stage so Temperature 0 is genuinely deterministic.
func newSamplerFromOptions(vocab *Vocab, opts CompletionOptions) (*Sampler, error) {
	s := NewSamplerChain()
	if s.sampler == nil {
		return nil, errorf(ErrDecode, "out of memory allocating sampler")
	}

	if g := opts.Grammar; g != "" {
		if vocab == nil {
			s.Close()
			return nil, errorf(ErrGrammar, "a grammar needs the model vocabulary")
		}
		cGrammar := C.CString(g)
		defer C.free(unsafe.Pointer(cGrammar))
		cRoot := C.CString("root")
		defer C.free(unsafe.Pointer(cRoot))
		gs := C.llama_sampler_init_grammar(vocab.vocab, cGrammar, cRoot)
		if gs == nil {
			s.Close()
			return nil, errorf(ErrGrammar, "llama.cpp rejected the grammar")
		}
		C.llama_sampler_chain_add(s.sampler, gs)
	}

	if opts.RepeatPenalty > 0 && opts.RepeatPenalty != 1 ||
		opts.PresencePenalty != 0 || opts.FrequencyPenalty != 0 {
		lastN := opts.RepeatLastN
		if lastN == 0 {
			lastN = 64
		}
		repeat := opts.RepeatPenalty
		if repeat <= 0 {
			repeat = 1
		}
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_penalties(
			C.int32_t(lastN), C.float(repeat), C.float(opts.FrequencyPenalty), C.float(opts.PresencePenalty),
		))
	}

	if opts.greedy() {
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_greedy())
		return s, nil
	}

	if opts.TopK > 0 {
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_top_k(C.int32_t(opts.TopK)))
	}
	if opts.TypicalP > 0 && opts.TypicalP < 1 {
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_typical(C.float(opts.TypicalP), 1))
	}
	if opts.TopP > 0 && opts.TopP < 1 {
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_top_p(C.float(opts.TopP), 1))
	}
	if opts.MinP > 0 {
		C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_min_p(C.float(opts.MinP), 1))
	}
	C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_temp(C.float(opts.Temperature)))
	C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_dist(C.uint32_t(opts.Seed)))
	return s, nil
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
	C.llama_sampler_chain_add(s.sampler,
		C.llama_sampler_init_penalties(C.int32_t(lastN), C.float(repeat), C.float(frequency), C.float(presence)))
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

// Sample picks the next token. idx is the index into the last decoded batch,
// with -1 meaning the final token.
func (s *Sampler) Sample(c *Context, idx int32) Token {
	if s == nil || s.sampler == nil || c == nil || c.ptr == nil {
		return -1
	}
	return Token(C.llama_sampler_sample(s.sampler, c.ptr, C.int32_t(idx)))
}

// Accept feeds a token back into the chain so penalties and grammar state stay
// correct.
func (s *Sampler) Accept(t Token) {
	if s != nil && s.sampler != nil {
		C.llama_sampler_accept(s.sampler, C.llama_token(t))
	}
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
