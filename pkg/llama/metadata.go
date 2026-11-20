package llama

/*
#include "llama.h"
#include <stdlib.h>
*/
import "C"
import (
	"unsafe"
)

// Metadata returns the GGUF key/value metadata of the model, which is what
// tooling and servers use to report the architecture, quantisation and context
// window they are actually running.
//
// The map is built on each call; callers that need it repeatedly should keep the
// result.
func (m *Model) Metadata() map[string]string {
	if m == nil || m.closed.Load() || m.ptr == nil {
		return nil
	}
	n := int(C.llama_model_meta_count(m.ptr))
	if n <= 0 {
		return nil
	}
	out := make(map[string]string, n)
	for i := range n {
		key, ok := m.metaString(func(buf *C.char, size C.size_t) C.int32_t {
			return C.llama_model_meta_key_by_index(m.ptr, C.int32_t(i), buf, size)
		})
		if !ok {
			continue
		}
		val, ok := m.metaString(func(buf *C.char, size C.size_t) C.int32_t {
			return C.llama_model_meta_val_str_by_index(m.ptr, C.int32_t(i), buf, size)
		})
		if !ok {
			continue
		}
		out[key] = val
	}
	return out
}

// MetaValue returns one metadata entry, which is cheaper than building the whole
// map when only the architecture or the chat template is needed.
func (m *Model) MetaValue(key string) (string, bool) {
	if m == nil || m.closed.Load() || m.ptr == nil || key == "" {
		return "", false
	}
	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))
	return m.metaString(func(buf *C.char, size C.size_t) C.int32_t {
		return C.llama_model_meta_val_str(m.ptr, cKey, buf, size)
	})
}

// metaString calls a llama.cpp metadata accessor twice: once to learn the length
// and once to fill a buffer of that size. llama.cpp returns the needed size as a
// negative value when the buffer is too small.
func (m *Model) metaString(fn func(buf *C.char, size C.size_t) C.int32_t) (string, bool) {
	buf := make([]byte, 64)
	for {
		n := fn((*C.char)(unsafe.Pointer(&buf[0])), C.size_t(len(buf)))
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
