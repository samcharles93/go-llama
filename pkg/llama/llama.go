package llama

/*
#include "llama.h"
#include <stdlib.h>

extern void goLogBridge(int level, char * text, void * user_data);

static void c_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    goLogBridge((int) level, (char *) text, user_data);
}

static void install_log_bridge(void) {
    llama_log_set(c_log_callback, NULL);
}
*/
import "C"

import (
	"os"
	"sync"
	"sync/atomic"
	"unsafe"
)

// The llama.cpp backend is process-global: llama_backend_init and
// llama_backend_free must bracket every model, and calling either twice
// corrupts state. Init is therefore idempotent, and Shutdown only frees the
// backend once every model and context is closed.
var (
	backendOnce sync.Once
	// backendMu serialises Shutdown against Init, because rearming the Once
	// while another goroutine calls Init would otherwise race.
	backendMu  sync.Mutex
	backendUp  atomic.Bool
	openModels atomic.Int64
)

// Init prepares the llama.cpp backend and routes its logging into this
// package's logger. It is safe to call from several goroutines and from
// several places; only the first call does work.
//
// Models can be loaded without calling Init: LoadModel calls it for you.
func Init() {
	backendOnce.Do(func() {
		installLogBridgeOnce.Do(func() { C.install_log_bridge() })
		C.llama_backend_init()
		backendUp.Store(true)
	})
}

var installLogBridgeOnce sync.Once

// Shutdown releases the backend's global resources. It is a no-op while any
// Model or Context is still open, so closing a leaking context cannot leave a
// live pointer into freed global state.
//
// Most programs never need to call it; the process exit does the same work.
func Shutdown() {
	backendMu.Lock()
	defer backendMu.Unlock()
	if !backendUp.Load() || openModels.Load() > 0 {
		return
	}
	backendOnce = sync.Once{}
	backendUp.Store(false)
	C.llama_backend_free()
}

// Initialize is Init under the older name.
//
// Deprecated: use Init.
func Initialize() { Init() }

// Free is Shutdown under the older name.
//
// Deprecated: use Shutdown.
func Free() { Shutdown() }

// Model is a loaded GGUF file: weights plus vocabulary. It is safe for
// concurrent use by several contexts, which is how a server serves parallel
// requests from one copy of the weights.
type Model struct {
	ptr    *C.struct_llama_model
	vocab  *Vocab
	path   string
	params ModelParams

	nCtxTrain int
	nEmbdOut  int
	nParams   uint64
	desc      string

	// mu guards contexts and the free of ptr. closed is atomic so hot paths can
	// test it without taking the lock.
	mu       sync.Mutex
	closed   atomic.Bool
	contexts int
}

// LoadModel loads a GGUF file. Callers usually pass no params, which loads every
// layer onto the GPU when one is available (see DefaultModelParams).
func LoadModel(path string, params ...ModelParams) (*Model, error) {
	p := DefaultModelParams()
	if len(params) > 1 {
		return nil, errorf(ErrModelLoad, "expected at most one ModelParams, got %d", len(params))
	}
	if len(params) == 1 {
		p = params[0]
	}
	if err := p.validate(); err != nil {
		return nil, err
	}
	if _, err := os.Stat(path); err != nil {
		return nil, wrap(ErrModelLoad, "stat "+path, err)
	}

	Init()

	cParams := C.llama_model_default_params()
	cParams.n_gpu_layers = C.int32_t(p.GPULayers)
	cParams.main_gpu = C.int32_t(p.MainGPU)
	cParams.use_mmap = C.bool(p.UseMmap)
	cParams.use_mlock = C.bool(p.UseMlock)
	cParams.vocab_only = C.bool(p.VocabOnly)

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	ptr := C.llama_model_load_from_file(cPath, cParams)
	if ptr == nil {
		return nil, errorf(ErrModelLoad, "%s: llama.cpp could not load the file (wrong format, corrupt, or unsupported architecture)", path)
	}

	m := &Model{
		ptr:       ptr,
		vocab:     newVocab(ptr),
		path:      path,
		params:    p,
		nCtxTrain: int(C.llama_model_n_ctx_train(ptr)),
		nEmbdOut:  int(C.llama_model_n_embd_out(ptr)),
		nParams:   uint64(C.llama_model_n_params(ptr)),
	}
	descBuf := make([]byte, 256)
	n := C.llama_model_desc(ptr, (*C.char)(unsafe.Pointer(&descBuf[0])), C.size_t(len(descBuf)))
	if n > 0 {
		m.desc = trimNul(string(descBuf[:min(int(n), len(descBuf))]))
	}
	openModels.Add(1)
	return m, nil
}

// Vocab returns the tokenizer. It stays valid until the model is closed.
func (m *Model) Vocab() *Vocab { return m.vocab }

// CountTokens reports how many tokens text occupies, which is what a server
// needs to decide whether a conversation still fits in the context.
func (m *Model) CountTokens(text string, addSpecial bool) (int, error) {
	if m == nil || m.Vocab() == nil {
		return 0, ErrClosed
	}
	tokens, err := m.vocab.Tokenize(text, addSpecial)
	if err != nil {
		return 0, err
	}
	return len(tokens), nil
}

// Path reports the file the model was loaded from.
func (m *Model) Path() string { return m.path }

// Description reports llama.cpp's one-line summary of the architecture, size
// and quantisation.
func (m *Model) Description() string { return m.desc }

// ContextSize reports the context length the model was trained for (n_ctx_train).
func (m *Model) ContextSize() int { return m.nCtxTrain }

// EmbeddingSize reports the width of the vectors Embed returns.
func (m *Model) EmbeddingSize() int { return m.nEmbdOut }

// Params reports how many parameters the model has.
func (m *Model) Params() uint64 { return m.nParams }

// LoadParams reports the parameters the model was loaded with.
func (m *Model) LoadParams() ModelParams { return m.params }

// HasDecoder reports whether the model can generate text at all. Embedding-only
// architectures return false.
func (m *Model) HasDecoder() bool {
	if m == nil || m.closed.Load() || m.ptr == nil {
		return false
	}
	return bool(C.llama_model_has_decoder(m.ptr))
}

// IsRecurrent reports whether the model keeps a recurrent state (Mamba, RWKV and
// relatives) instead of a plain attention cache.
func (m *Model) IsRecurrent() bool {
	if m == nil || m.closed.Load() || m.ptr == nil {
		return false
	}
	return bool(C.llama_model_is_recurrent(m.ptr))
}

// IsHybrid reports whether the model mixes recurrent and attention layers, as
// LFM2 does.
//
// Hybrid and recurrent models cannot drop the tail of their state, so a turn that
// diverges from the cached prompt has to re-evaluate the whole prompt: expect
// Result.ReusedTokens to be zero for them unless the new prompt is an exact
// extension of the cached one.
func (m *Model) IsHybrid() bool {
	if m == nil || m.closed.Load() || m.ptr == nil {
		return false
	}
	return bool(C.llama_model_is_hybrid(m.ptr))
}

// NewContext creates a context for this model. Callers usually pass nothing,
// which gives a chat-ready context sized to the model's trained window.
func (m *Model) NewContext(opts ...ContextOptions) (*Context, error) {
	if m == nil {
		return nil, ErrClosed
	}
	// Context creation touches the model pointer and the context count, so it
	// takes the same lock Close does.
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed.Load() {
		return nil, ErrClosed
	}
	o := DefaultContextOptions()
	if len(opts) > 1 {
		return nil, errorf(ErrContextCreate, "expected at most one ContextOptions, got %d", len(opts))
	}
	if len(opts) == 1 {
		o = opts[0]
	}

	cParams := C.llama_context_default_params()
	cParams.n_ctx = C.uint32_t(o.resolveContextSize(m.nCtxTrain))
	if o.BatchSize > 0 {
		cParams.n_batch = C.uint32_t(o.BatchSize)
	}
	if o.UBatchSize > 0 {
		cParams.n_ubatch = C.uint32_t(o.UBatchSize)
	}
	if o.Sequences > 0 {
		cParams.n_seq_max = C.uint32_t(o.Sequences)
	}
	// llama.cpp's own default is a fixed 4 threads, which wastes a big machine
	// and oversubscribes a small one, so zero means DefaultThreads here.
	cParams.n_threads = C.int32_t(DefaultThreads())
	if o.Threads > 0 {
		cParams.n_threads = C.int32_t(o.Threads)
	}
	cParams.n_threads_batch = C.int32_t(DefaultThreads())
	if o.ThreadsBatch > 0 {
		cParams.n_threads_batch = C.int32_t(o.ThreadsBatch)
	}
	cParams.embeddings = C.bool(o.Embeddings)
	cParams.no_perf = C.bool(o.DisablePerf)
	if o.Pooling != PoolingUnspecified {
		cParams.pooling_type = C.enum_llama_pooling_type(o.Pooling)
	}
	cParams.flash_attn_type = o.FlashAttention.llamaType()

	ptr := C.llama_init_from_model(m.ptr, cParams)
	if ptr == nil {
		return nil, errorf(ErrContextCreate, "llama.cpp could not allocate a %d token context for %s (out of memory?); lower ContextOptions.ContextSize", cParams.n_ctx, m.path)
	}

	nCtx := int(C.llama_n_ctx(ptr))
	nBatch := int(C.llama_n_batch(ptr))
	if nBatch <= 0 || nBatch > nCtx {
		nBatch = nCtx
	}
	nUBatch := int(C.llama_n_ubatch(ptr))
	c := &Context{
		ptr:        ptr,
		model:      m,
		nCtx:       nCtx,
		nBatch:     nBatch,
		nUBatch:    nUBatch,
		embeddings: o.Embeddings,
		pooling:    o.Pooling,
		flashAttn:  o.FlashAttention,
	}
	if err := c.installAbort(); err != nil {
		C.llama_free(ptr)
		return nil, err
	}
	m.contexts++
	return c, nil
}

// Context holds a KV cache and the compute state for one conversation. It is
// safe for concurrent use: generation takes an internal lock, because a KV cache
// can only serve one decode at a time. Parallelism comes from several contexts
// sharing one Model.
type Context struct {
	ptr        *C.struct_llama_context
	model      *Model
	nCtx       int
	nBatch     int
	nUBatch    int
	embeddings bool
	pooling    PoolingType
	flashAttn  FlashAttentionMode
	batch      *Batch
	// abort is the C flag llama.cpp polls during graph compute; see abort.go.
	abort unsafe.Pointer

	mu      sync.Mutex
	history []Token // tokens already in the KV cache, in order
	// logitsValid records whether the context holds output logits for the last
	// cached position. Sampling without them reads whatever the buffer held,
	// which after LoadState is not the prompt at all.
	logitsValid bool
	closed      atomic.Bool
}

// Model returns the model the context belongs to.
func (c *Context) Model() *Model { return c.model }

// ContextSize reports the KV cache size in tokens.
func (c *Context) ContextSize() int { return c.nCtx }

// BatchSize reports the largest batch the context accepts.
func (c *Context) BatchSize() int { return c.nBatch }

// UBatchSize reports the micro-batch size, which is the longest text a pooled
// embedding can see in one piece.
func (c *Context) UBatchSize() int { return c.nUBatch }

// Threads reports the thread counts llama.cpp uses for generation and for
// batch (prompt) processing.
func (c *Context) Threads() (generation, batch int32) {
	if c == nil || c.closed.Load() {
		return 0, 0
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ptr == nil {
		return 0, 0
	}
	return int32(C.llama_n_threads(c.ptr)), int32(C.llama_n_threads_batch(c.ptr))
}

// FlashAttention reports the flash attention mode the context was created with.
func (c *Context) FlashAttention() FlashAttentionMode { return c.flashAttn }

// llamaType maps the mode onto llama.cpp's enum. Auto is spelled out rather
// than left to the default params so the mapping stays correct if llama.cpp
// ever changes its default.
func (m FlashAttentionMode) llamaType() C.enum_llama_flash_attn_type {
	switch m {
	case FlashAttentionOn:
		return C.LLAMA_FLASH_ATTN_TYPE_ENABLED
	case FlashAttentionOff:
		return C.LLAMA_FLASH_ATTN_TYPE_DISABLED
	default:
		return C.LLAMA_FLASH_ATTN_TYPE_AUTO
	}
}

// llamaName is llama.cpp's name for the mapped enum value, which lets a test
// check the mapping without cgo in the test file.
func (m FlashAttentionMode) llamaName() string {
	return C.GoString(C.llama_flash_attn_type_name(m.llamaType()))
}

// TokenCount reports how many tokens currently occupy the KV cache.
func (c *Context) TokenCount() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.history)
}

// Reset drops the KV cache, forgetting the conversation. Embedding contexts
// must be reset between unrelated documents.
func (c *Context) Reset() error {
	if c == nil || c.closed.Load() {
		return ErrClosed
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	C.llama_memory_clear(C.llama_get_memory(c.ptr), true)
	c.history = c.history[:0]
	c.logitsValid = false
	return nil
}

// Close frees the context. Safe to call more than once. The Model must outlive
// it.
func (c *Context) Close() error {
	if c == nil || c.closed.Swap(true) {
		return nil
	}
	// Generation checks the closed flag on every token, so waiting for the lock
	// returns promptly instead of blocking for the rest of a long reply.
	c.mu.Lock()
	if c.ptr != nil {
		C.llama_free(c.ptr)
		c.ptr = nil
	}
	c.freeAbort()
	// The scratch batch holds C memory that llama_batch_free owns, so it has to
	// go with the context.
	if c.batch != nil {
		_ = c.batch.Close()
		c.batch = nil
	}
	c.mu.Unlock()

	if c.model != nil {
		c.model.mu.Lock()
		c.model.contexts--
		c.model.mu.Unlock()
	}
	return nil
}

// Free is Close, kept for callers written against the old API.
//
// Deprecated: use Close.
func (c *Context) Free() { _ = c.Close() }

// Close frees the model. Contexts created from it must be closed first; the
// model refuses to free while they are open.
func (m *Model) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed.Load() {
		return nil
	}
	if m.contexts > 0 {
		// Freeing now would leave every live context pointing at freed weights.
		return errorf(ErrClosed, "model %s still has %d open context(s); close them first", m.path, m.contexts)
	}
	m.closed.Store(true)
	C.llama_model_free(m.ptr)
	m.ptr = nil
	openModels.Add(-1)
	return nil
}

// Free is Close, kept for callers written against the old API.
//
// Deprecated: use Close.
func (m *Model) Free() { _ = m.Close() }
