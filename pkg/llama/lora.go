package llama

/*
#include "llama.h"
#include <stdlib.h>
*/
import "C"
import (
	"os"
	"sync"
	"unsafe"
)

// LoraAdapter is a LoRA (or QLoRA) adapter loaded from a GGUF file. One adapter
// can be applied to several contexts of the same model, each with its own scale.
//
// The adapter is owned by the model: llama.cpp frees every adapter it created
// when the model is freed, and llama_adapter_lora_free is deprecated in favour of
// that. Close therefore drops this package's reference rather than freeing the C
// object, and refuses while a context still has the adapter applied.
type LoraAdapter struct {
	ptr   *C.struct_llama_adapter_lora
	model *Model
	path  string

	mu      sync.Mutex
	closed  bool
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
// adapter as trained and 0 neutralises it. Applying the same adapter twice
// replaces the previous scale rather than stacking another reference.
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
	if rc := C.llama_set_adapter_lora(c.ptr, a.ptr, C.float(scale)); rc != 0 {
		return errorf(ErrDecode, "llama_set_adapter_lora failed with code %d", rc)
	}
	a.applied = min(a.applied+1, 1<<20)
	return nil
}

// Remove detaches the adapter from a context.
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
	if rc := C.llama_rm_adapter_lora(c.ptr, a.ptr); rc != 0 {
		return errorf(ErrDecode, "llama_rm_adapter_lora failed with code %d", rc)
	}
	if a.applied > 0 {
		a.applied--
	}
	return nil
}

// Applied reports how many Apply calls are outstanding. It is a coarse count: the
// same adapter applied twice to one context counts twice.
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
	defer c.mu.Unlock()
	if c.ptr == nil {
		return ErrClosed
	}
	C.llama_clear_adapter_lora(c.ptr)
	return nil
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
