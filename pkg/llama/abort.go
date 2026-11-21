package llama

/*
#include "llama.h"
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>

// The flag is a C atomic rather than Go memory because ggml polls it from its
// own compute threads in the middle of a graph, where calling back into Go for
// every check would cost a cgo transition per node. cgo cannot name _Atomic
// types, so the Go side only ever holds a void pointer.

static bool go_llama_abort_cb(void * data) {
    return atomic_load_explicit((atomic_bool *) data, memory_order_relaxed);
}

static void * go_llama_abort_new(void) {
    atomic_bool * f = malloc(sizeof *f);
    if (f) {
        atomic_init(f, false);
    }
    return f;
}

static void go_llama_abort_set(void * f, bool v) {
    atomic_store_explicit((atomic_bool *) f, v, memory_order_relaxed);
}

static void go_llama_abort_install(struct llama_context * ctx, void * f) {
    llama_set_abort_callback(ctx, go_llama_abort_cb, f);
}
*/
import "C"

import (
	"context"
	"sync"
)

// installAbort gives the context a cancellation flag that llama.cpp checks
// while it computes a graph, so cancelling a Go context interrupts a long
// prompt evaluation instead of waiting for it to finish.
//
// Only backends that implement ggml_backend_set_abort_callback poll the flag:
// CPU and Metal do, CUDA does not. On CUDA a cancelled prompt still stops at
// the next batch boundary, because syncKV checks the Go context between
// batches.
func (c *Context) installAbort() error {
	f := C.go_llama_abort_new()
	if f == nil {
		return errorf(ErrContextCreate, "could not allocate the abort flag")
	}
	C.go_llama_abort_install(c.ptr, f)
	c.abort = f
	return nil
}

// freeAbort releases the flag. It must run after llama_free, which is the last
// point llama.cpp can read it.
func (c *Context) freeAbort() {
	if c.abort != nil {
		C.free(c.abort)
		c.abort = nil
	}
}

// watchCancel raises the abort flag when ctx is cancelled, until the returned
// stop function runs. The caller holds c.mu for the whole span.
//
// stop lowers the flag again, and the active guard stops a callback that was
// already running when stop was called from raising it afterwards, which
// would abort the next, unrelated decode.
func (c *Context) watchCancel(ctx context.Context) (stop func()) {
	if c.abort == nil {
		return func() {}
	}
	C.go_llama_abort_set(c.abort, false)
	if ctx.Done() == nil {
		return func() {}
	}
	var mu sync.Mutex
	active := true
	f := c.abort
	stopAfter := context.AfterFunc(ctx, func() {
		mu.Lock()
		defer mu.Unlock()
		if active {
			C.go_llama_abort_set(f, true)
		}
	})
	return func() {
		stopAfter()
		mu.Lock()
		active = false
		C.go_llama_abort_set(f, false)
		mu.Unlock()
	}
}
