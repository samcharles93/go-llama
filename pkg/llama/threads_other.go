//go:build !linux

package llama

// mathCores reports 0 (unknown) off Linux, so DefaultThreads falls back to half
// the logical CPUs.
func mathCores() int { return 0 }
