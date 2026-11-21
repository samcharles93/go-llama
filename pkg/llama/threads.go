package llama

import (
	"runtime"
	"sync"
)

// DefaultThreads is the thread count a context uses when ContextOptions leaves
// Threads or ThreadsBatch at zero: one thread per physical performance core.
//
// runtime.NumCPU counts SMT siblings and, on hybrid Intel parts, efficiency
// cores. ggml splits each matrix multiply evenly across its threads, so every
// step waits for the slowest one, and a thread on an E-core or sharing a core
// with its sibling drags the whole pool down. In a llama-bench sweep on an
// i5-12500H (4 P-cores with SMT, 8 E-cores, 16 logical CPUs) with Qwen2.5 0.5B,
// 16 threads generated at about 35 tok/s and 4 at about 60. This mirrors
// llama.cpp's own cpu_get_num_math; set Threads explicitly to tune further.
func DefaultThreads() int32 { return defaultThreads() }

var defaultThreads = sync.OnceValue(func() int32 {
	return clampThreads(mathCores(), runtime.NumCPU())
})

// clampThreads turns a detected core count into a usable thread count. A failed
// detection (cores <= 0) falls back to half the logical CPUs, the usual SMT
// ratio, and the result never exceeds what the process may run on.
func clampThreads(cores, ncpu int) int32 {
	if ncpu < 1 {
		ncpu = 1
	}
	if cores <= 0 {
		cores = ncpu
		if ncpu > 4 {
			cores = ncpu / 2
		}
	}
	return int32(min(cores, ncpu))
}
