package llama

/*
#include "llama.h"
#include <stdlib.h>
*/
import "C"

import (
	"os"
	"slices"
	"sync"
	"unsafe"
)

// LoraAdapter is a LoRA (or QLoRA) adapter loaded from a GGUF file. One adapter
// can be applied to several contexts of the same model, each with its own scale.
//
// The adapter is owned by the model: llama.cpp frees every adapter it created
// when the model is freed. Close drops this package's reference rather than
// freeing the C object, so a model that is still loaded never holds a dangling
// adapter, and it refuses while a context still has the adapter applied.
type LoraAdapter struct {
	ptr   *C.struct_llama_adapter_lora
	model *Model
	path  string

	mu     sync.Mutex
	closed bool
	// applied counts the contexts that have this adapter applied.
	applied int
}

// LoadLora loads an adapter for this model. The model must outlive the adapter.
func (m *Model) LoadLora(path string) (*LoraAdapter, error) {
	if m == nil || m.closed.Load() {
		return nil, ErrClosed
	}
	if _, err := os.Stat(path); err != nil {
		return nil, wrap(ErrModelLoad, "stat "+path, err)
	}
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	ptr := C.llama_adapter_lora_init(m.ptr, cPath)
	if ptr == nil {
		return nil, errorf(ErrModelLoad, "%s: llama.cpp could not load the LoRA adapter", path)
	}
	return &LoraAdapter{ptr: ptr, model: m, path: path}, nil
}

// Path reports the adapter file.
func (a *LoraAdapter) Path() string { return a.path }

// Apply attaches the adapter to a context at the given scale, where 1 uses the
// adapter as trained and 0 neutralises it. Applying the same adapter again
// replaces its scale rather than stacking a second copy.
func (a *LoraAdapter) Apply(c *Context, scale float32) error {
	if a == nil || c == nil {
		return ErrClosed
	}
	if a.model != c.model {
		return errorf(ErrDecode, "adapter %s belongs to a different model", a.path)
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed || a.ptr == nil {
		return ErrClosed
	}
	if a.model.closed.Load() {
		return ErrClosed
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ptr == nil {
		return ErrClosed
	}
	next := slices.Clone(c.loras)
	i := slices.IndexFunc(next, func(s loraSlot) bool { return s.adapter == a })
	if i >= 0 {
		next[i].scale = scale
	} else {
		next = append(next, loraSlot{adapter: a, scale: scale})
	}
	if err := c.setLoras(next); err != nil {
		return err
	}
	if i < 0 {
		a.applied++
	}
	return nil
}

// Remove detaches the adapter from a context. Removing an adapter that is not
// applied there is a no-op.
func (a *LoraAdapter) Remove(c *Context) error {
	if a == nil || c == nil {
		return ErrClosed
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed || a.ptr == nil {
		return ErrClosed
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ptr == nil {
		return ErrClosed
	}
	i := slices.IndexFunc(c.loras, func(s loraSlot) bool { return s.adapter == a })
	if i < 0 {
		return nil
	}
	if err := c.setLoras(slices.Delete(slices.Clone(c.loras), i, i+1)); err != nil {
		return err
	}
	a.applied--
	return nil
}

// Applied reports how many contexts currently have the adapter applied.
func (a *LoraAdapter) Applied() int {
	if a == nil {
		return 0
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.applied
}

// RemoveAllLoras detaches every adapter from a context in one call.
func (c *Context) RemoveAllLoras() error {
	if c == nil || c.closed.Load() {
		return ErrClosed
	}
	c.mu.Lock()
	if c.ptr == nil {
		c.mu.Unlock()
		return ErrClosed
	}
	if err := c.setLoras(nil); err != nil {
		c.mu.Unlock()
		return err
	}
	dropped := c.loras
	c.loras = nil
	c.mu.Unlock()
	releaseLoras(dropped)
	return nil
}

// loraSlot is one adapter applied to a context.
type loraSlot struct {
	adapter *LoraAdapter
	scale   float32
}

// setLoras hands llama.cpp the complete adapter list for the context, which is
// the only form this llama.cpp accepts: llama_set_adapters_lora replaced the
// per-adapter set, remove and clear calls. The caller holds c.mu, and c.loras
// changes only when llama.cpp accepted the list.
func (c *Context) setLoras(next []loraSlot) error {
	ptrs := make([]*C.struct_llama_adapter_lora, len(next))
	scales := make([]C.float, len(next))
	for i, s := range next {
		ptrs[i] = s.adapter.ptr
		scales[i] = C.float(s.scale)
	}
	var pp **C.struct_llama_adapter_lora
	var sp *C.float
	if len(next) > 0 {
		pp, sp = &ptrs[0], &scales[0]
	}
	if rc := C.llama_set_adapters_lora(c.ptr, pp, C.size_t(len(next)), sp); rc != 0 {
		return errorf(ErrDecode, "llama_set_adapters_lora failed with code %d", rc)
	}
	c.loras = next
	return nil
}

// releaseLoras drops a context's hold on adapters it no longer has applied. It
// runs without c.mu, because Apply takes an adapter's lock before the
// context's, and taking them in the other order here could deadlock.
func releaseLoras(slots []loraSlot) {
	for _, s := range slots {
		s.adapter.mu.Lock()
		if s.adapter.applied > 0 {
			s.adapter.applied--
		}
		s.adapter.mu.Unlock()
	}
}

// Close drops this package's reference to the adapter. It refuses while a context
// still has the adapter applied, because detaching is the caller's decision and
// silently changing a running context's weights would be worse than an error.
func (a *LoraAdapter) Close() error {
	if a == nil {
		return nil
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed {
		return nil
	}
	if a.applied > 0 {
		return errorf(ErrClosed, "adapter %s is still applied to %d context(s); remove it first", a.path, a.applied)
	}
	a.closed = true
	a.ptr = nil
	return nil
}

// MetaValue reads one metadata entry from the adapter GGUF, which is how a caller
// confirms which adapter it loaded and what it was trained for.
func (a *LoraAdapter) MetaValue(key string) (string, bool) {
	if a == nil || key == "" {
		return "", false
	}
	a.mu.Lock()
	ptr := a.ptr
	a.mu.Unlock()
	if ptr == nil {
		return "", false
	}
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	buf := make([]byte, 64)
	for {
		n := C.llama_adapter_meta_val_str(ptr, cKey, (*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)))
		if n >= 0 {
			return trimNul(string(buf[:min(int(n), len(buf))])), true
		}
		need := int(-n)
		if need <= len(buf) || need > 1<<20 {
			return "", false
		}
		buf = make([]byte, need+1)
	}
}
