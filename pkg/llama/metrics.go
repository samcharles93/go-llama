package llama

/*
#include "llama.h"
#include <stdlib.h>
*/
import "C"

// Metrics reports llama.cpp's own timings for one generation.
type Metrics struct {
	// PromptTokens is how many prompt tokens were actually evaluated. Tokens
	// served from the KV cache are not counted.
	PromptTokens int
	// GeneratedTokens is how many tokens were evaluated during generation.
	GeneratedTokens int
	// PromptMS is the wall time spent evaluating the prompt.
	PromptMS float64
	// GenerateMS is the wall time spent generating.
	GenerateMS float64
	// LoadMS is the time llama.cpp spent loading the model.
	LoadMS float64
}

// PromptTokensPerSecond reports prompt processing throughput.
func (m Metrics) PromptTokensPerSecond() float64 {
	if m.PromptMS <= 0 {
		return 0
	}
	return float64(m.PromptTokens) * 1000 / m.PromptMS
}

// TokensPerSecond reports generation throughput, the number people quote when
// comparing hardware.
func (m Metrics) TokensPerSecond() float64 {
	if m.GenerateMS <= 0 {
		return 0
	}
	return float64(m.GeneratedTokens) * 1000 / m.GenerateMS
}

// metrics reads llama.cpp's perf counters for the context. Callers hold the
// context lock.
func (c *Context) metrics() Metrics {
	if c == nil || c.ptr == nil {
		return Metrics{}
	}
	d := C.llama_perf_context(c.ptr)
	return Metrics{
		PromptTokens:    int(d.n_p_eval),
		GeneratedTokens: int(d.n_eval),
		PromptMS:        float64(d.t_p_eval_ms),
		GenerateMS:      float64(d.t_eval_ms),
		LoadMS:          float64(d.t_load_ms),
	}
}

// Metrics reports the context's cumulative perf counters.
func (c *Context) Metrics() Metrics {
	if c == nil || c.closed.Load() {
		return Metrics{}
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.metrics()
}

// ResetMetrics zeroes the context's perf counters, which is useful before
// timing a specific run.
func (c *Context) ResetMetrics() {
	if c == nil || c.closed.Load() {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ptr != nil {
		C.llama_perf_context_reset(c.ptr)
	}
}
