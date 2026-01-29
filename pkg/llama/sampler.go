package llama

/*
#cgo CFLAGS: -I${SRCDIR}/../../llama.cpp/include -I${SRCDIR}/../../llama.cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../../llama.cpp/build/bin -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,${SRCDIR}/../../llama.cpp/build/bin
#include "llama.h"
*/
import "C"

type Sampler struct {
	sampler *C.struct_llama_sampler
}

func NewSamplerChain() *Sampler {
	params := C.llama_sampler_chain_default_params()
	return &Sampler{sampler: C.llama_sampler_chain_init(params)}
}

func (s *Sampler) AddGreedy() {
	C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_greedy())
}

func (s *Sampler) AddDist(seed uint32) {
	C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_dist(C.uint32_t(seed)))
}

func (s *Sampler) AddTopK(k int32) {
	C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_top_k(C.int32_t(k)))
}

func (s *Sampler) AddTopP(p float32, minKeep int) {
	C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_top_p(C.float(p), C.size_t(minKeep)))
}

func (s *Sampler) AddTemp(t float32) {
	C.llama_sampler_chain_add(s.sampler, C.llama_sampler_init_temp(C.float(t)))
}

func (s *Sampler) Free() {
	C.llama_sampler_free(s.sampler)
}

func (s *Sampler) Sample(ctx *Context, idx int32) Token {
	return Token(C.llama_sampler_sample(s.sampler, ctx.ctx, C.int32_t(idx)))
}

func (s *Sampler) Accept(token Token) {
	C.llama_sampler_accept(s.sampler, C.llama_token(token))
}
