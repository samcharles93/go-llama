package llama

/*
#cgo CFLAGS: -I${SRCDIR}/../../llama.cpp/include -I${SRCDIR}/../../llama.cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../../llama.cpp/build/src -L${SRCDIR}/../../llama.cpp/build/ggml/src -lllama -lggml -lggml-base -lggml-cpu -lstdc++ -lm -fopenmp
#include "llama.h"
#include <stdlib.h>
#include <stdio.h>

// Declare the exported Go function
extern void GoLogCallback(int level, char * text);

// C callback that forwards to Go
void c_log_callback(enum ggml_log_level level, const char * text, void * user_data) {
    GoLogCallback((int)level, (char *)text);
}

// Function to enable logging via Go callback
void enable_go_logging() {
    llama_log_set(c_log_callback, NULL);
}
*/
import "C"
import (
	"fmt"
	"strings"
	"unsafe"
)

// Initialize the llama backend.
func Initialize() {
    // Redirect internal logs to Go slog
    C.enable_go_logging()
    
	C.llama_backend_init()
}

// Free the llama backend.
func Free() {
	C.llama_backend_free()
}

type Model struct {
	model *C.struct_llama_model
}

func (m *Model) Free() {
	C.llama_model_free(m.model)
}

func (m *Model) BOS() Token {
	vocab := C.llama_model_get_vocab(m.model)
	return Token(C.llama_vocab_bos(vocab))
}

func (m *Model) EOS() Token {
	vocab := C.llama_model_get_vocab(m.model)
	return Token(C.llama_vocab_eos(vocab))
}

type Context struct {
	ctx   *C.struct_llama_context
	model *Model
}

func (c *Context) Free() {
	C.llama_free(c.ctx)
}

func (c *Context) Generate(prompt string, opts CompletionOptions) (string, error) {
	ch, err := c.Stream(prompt, opts)
	if err != nil {
		return "", err
	}

	var sb strings.Builder
	for piece := range ch {
		sb.WriteString(piece)
	}
	return sb.String(), nil
}

func (c *Context) Stream(prompt string, opts CompletionOptions) (<-chan string, error) {
	out := make(chan string)

	go func() {
		defer close(out)

		tokens := c.model.Tokenize(prompt, true, true)
		if len(tokens) == 0 {
			return
		}

		batch := NewBatch(int32(len(tokens)+opts.MaxTokens), 0, 1)
		defer batch.Free()

		sampler := NewSamplerChain()
		defer sampler.Free()
		if opts.Temperature > 0 {
			sampler.AddTemp(opts.Temperature)
		}
		if opts.TopK > 0 {
			sampler.AddTopK(opts.TopK)
		}
		if opts.TopP > 0 {
			sampler.AddTopP(opts.TopP, 1)
		}
		sampler.AddDist(opts.Seed)

		// Initial prompt decode
		for i, t := range tokens {
			batch.Add(t, int32(i), []int32{0}, i == len(tokens)-1)
		}

		if err := c.Decode(batch); err != nil {
			return
		}

		curPos := int32(len(tokens))
		eos := c.model.EOS()

		for i := 0; i < opts.MaxTokens; i++ {
			newToken := sampler.Sample(c, -1)
			sampler.Accept(newToken)

			if newToken == eos {
				break
			}

			piece := c.model.TokenToPiece(newToken)
			out <- piece

			batch.Clear()
			batch.Add(newToken, curPos, []int32{0}, true)
			curPos++

			if err := c.Decode(batch); err != nil {
				break
			}
		}
	}()

	return out, nil
}

func LoadModel(path string) (*Model, error) {
	params := C.llama_model_default_params()
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	m := C.llama_model_load_from_file(cPath, params)
	if m == nil {
		return nil, fmt.Errorf("failed to load model from %s", path)
	}

	return &Model{model: m}, nil
}

type ContextParams struct {
	params C.struct_llama_context_params
}

func DefaultContextParams() ContextParams {
	return ContextParams{params: C.llama_context_default_params()}
}

func (p *ContextParams) SetPooling(poolingType int) {
	p.params.pooling_type = C.enum_llama_pooling_type(poolingType)
}

func (m *Model) NewContextWithParams(params ContextParams) (*Context, error) {
	c := C.llama_init_from_model(m.model, params.params)
	if c == nil {
		return nil, fmt.Errorf("failed to create context")
	}
	return &Context{ctx: c, model: m}, nil
}

func (m *Model) NewContext() (*Context, error) {
	params := C.llama_context_default_params()
	c := C.llama_init_from_model(m.model, params)
	if c == nil {
		return nil, fmt.Errorf("failed to create context")
	}
	return &Context{ctx: c, model: m}, nil
}

func (c *Context) Decode(batch *Batch) error {
	ret := C.llama_decode(c.ctx, batch.batch)
	if ret != 0 {
		return fmt.Errorf("llama_decode failed with code %d", ret)
	}
	return nil
}

func (c *Context) Embeddings(text string) ([]float32, error) {
	tokens := c.model.Tokenize(text, true, false)
	batch := NewBatch(int32(len(tokens)), 0, 1)
	defer batch.Free()

	for i, t := range tokens {
		batch.Add(t, int32(i), []int32{0}, i == len(tokens)-1)
	}

	if err := c.Decode(batch); err != nil {
		return nil, err
	}

	nEmbd := int(C.llama_model_n_embd(c.model.model))
	ptr := C.llama_get_embeddings_ith(c.ctx, C.int32_t(len(tokens)-1))
	if ptr == nil {
		// Try sequence embeddings if ith returns null (depends on pooling)
		ptr = C.llama_get_embeddings_seq(c.ctx, 0)
	}
	if ptr == nil {
		return nil, fmt.Errorf("failed to get embeddings (check if pooling is enabled)")
	}

	slice := (*[1 << 30]float32)(unsafe.Pointer(ptr))[:nEmbd:nEmbd]
	res := make([]float32, nEmbd)
	copy(res, slice)
	return res, nil
}

type Token int32

type ChatMessage struct {
	Role    string
	Content string
}

func (m *Model) ApplyTemplate(messages []ChatMessage, addAssistant bool) (string, error) {
	if len(messages) == 0 {
		return "", nil
	}

	cMessages := make([]C.struct_llama_chat_message, len(messages))
	// Keep pointers to C strings to prevent GC
	cStrings := make([]*C.char, len(messages)*2)

	for i, msg := range messages {
		role := C.CString(msg.Role)
		content := C.CString(msg.Content)
		cStrings[i*2] = role
		cStrings[i*2+1] = content
		
		cMessages[i].role = role
		cMessages[i].content = content
	}

	defer func() {
		for _, s := range cStrings {
			C.free(unsafe.Pointer(s))
		}
	}()

	// First call to get length
	tmpl := C.CString("") // Use default template (empty string usually works or nil)
    // Actually passing NULL (nil in Go) is safer for default
    // But C.CString("") creates an empty string. C.llama_chat_apply_template expects NULL for default.
    // Let's use nil.
    defer C.free(unsafe.Pointer(tmpl))

    	// Using nil for tmpl
        var tmplPtr *C.char = nil
    
    	n := C.llama_chat_apply_template(
    		tmplPtr,
    		&cMessages[0],
    		C.size_t(len(messages)),
    				C.bool(addAssistant),
    				nil,
    				0,
    			)
    		
    			if n < 0 {
    		
    		return "", fmt.Errorf("failed to calculate template length")
    	}
    	buf := make([]C.char, n+1)
	n = C.llama_chat_apply_template(
		tmplPtr,
		&cMessages[0],
		C.size_t(len(messages)),
		C.bool(addAssistant),
		&buf[0],
		n+1,
	)

	if n < 0 {
		return "", fmt.Errorf("failed to apply template")
	}

	return C.GoStringN(&buf[0], n), nil
}

func (m *Model) Tokenize(text string, addSpecial bool, parseSpecial bool) []Token {
	vocab := C.llama_model_get_vocab(m.model)
	cText := C.CString(text)
	defer C.free(unsafe.Pointer(cText))

	// Estimate max tokens: usually 1 token per char is safe for rough estimation
	nMax := C.int32_t(len(text) + 4)
	tokens := make([]C.llama_token, nMax)

	nTokens := C.llama_tokenize(vocab, cText, C.int32_t(len(text)), &tokens[0], nMax, C.bool(addSpecial), C.bool(parseSpecial))
	if nTokens < 0 {
		// Buffer too small, resize
		tokens = make([]C.llama_token, -nTokens)
		nTokens = C.llama_tokenize(vocab, cText, C.int32_t(len(text)), &tokens[0], -nTokens, C.bool(addSpecial), C.bool(parseSpecial))
	}

	res := make([]Token, nTokens)
	for i := 0; i < int(nTokens); i++ {
		res[i] = Token(tokens[i])
	}
	return res
}

func (m *Model) TokenToPiece(token Token) string {
	vocab := C.llama_model_get_vocab(m.model)
	buf := make([]C.char, 128)
	n := C.llama_token_to_piece(vocab, C.llama_token(token), &buf[0], C.int32_t(len(buf)), 0, true)
	if n < 0 {
		buf = make([]C.char, -n)
		n = C.llama_token_to_piece(vocab, C.llama_token(token), &buf[0], C.int32_t(len(buf)), 0, true)
	}
	return C.GoStringN(&buf[0], n)
}
