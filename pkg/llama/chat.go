package llama

/*
#include "llama.h"
#include <stdlib.h>
*/
import "C"
import "unsafe"

// ChatMessage is one turn of a conversation.
type ChatMessage struct {
	// Role is "system", "user" or "assistant", as the model's template expects.
	Role string
	// Content is the message text. It is sent as-is, so it must not contain
	// template control tokens.
	Content string
}

// SystemMessage, UserMessage and AssistantMessage build the common cases.
func SystemMessage(content string) ChatMessage { return ChatMessage{Role: "system", Content: content} }
func UserMessage(content string) ChatMessage   { return ChatMessage{Role: "user", Content: content} }
func AssistantMessage(content string) ChatMessage {
	return ChatMessage{Role: "assistant", Content: content}
}

// ChatTemplate returns the chat template embedded in the GGUF metadata, or ""
// when the model has none. name selects a named template; "" takes the default.
func (m *Model) ChatTemplate(name string) string {
	if m == nil || m.closed.Load() {
		return ""
	}
	var cName *C.char
	if name != "" {
		cName = C.CString(name)
		defer C.free(unsafe.Pointer(cName))
	}
	ptr := C.llama_model_chat_template(m.ptr, cName)
	if ptr == nil {
		return ""
	}
	return C.GoString(ptr)
}

// ApplyChatTemplate renders messages into the prompt the model expects, using
// the template stored in the GGUF file.
//
// addAssistant leaves the assistant turn open, which is what you want before
// generating a reply.
func (m *Model) ApplyChatTemplate(messages []ChatMessage, addAssistant bool) (string, error) {
	return m.applyChatTemplate(nil, messages, addAssistant)
}

// ApplyChatTemplateWith renders messages with an explicit Jinja-style template
// string instead of the one in the model. Use it for models whose embedded
// template llama.cpp does not recognise; see llama.cpp's
// templates-supported-by-llama_chat_apply_template for the accepted subset.
func (m *Model) ApplyChatTemplateWith(tmpl string, messages []ChatMessage, addAssistant bool) (string, error) {
	cTmpl := C.CString(tmpl)
	defer C.free(unsafe.Pointer(cTmpl))
	return m.applyChatTemplate(cTmpl, messages, addAssistant)
}

func (m *Model) applyChatTemplate(tmpl *C.char, messages []ChatMessage, addAssistant bool) (string, error) {
	if m == nil || m.closed.Load() {
		return "", ErrClosed
	}
	if len(messages) == 0 {
		return "", nil
	}

	cMsgs := make([]C.struct_llama_chat_message, len(messages))
	// C.CString allocates, so every string has to be freed even when a later
	// allocation fails.
	freed := make([]unsafe.Pointer, 0, 2*len(messages))
	defer func() {
		for _, p := range freed {
			C.free(p)
		}
	}()
	for i, msg := range messages {
		role := C.CString(msg.Role)
		freed = append(freed, unsafe.Pointer(role))
		content := C.CString(msg.Content)
		freed = append(freed, unsafe.Pointer(content))
		cMsgs[i].role = role
		cMsgs[i].content = content
	}

	n := C.llama_chat_apply_template(tmpl, &cMsgs[0], C.size_t(len(cMsgs)), C.bool(addAssistant), nil, 0)
	if n < 0 {
		return "", errorf(ErrTemplate, "model %s has no usable chat template; pass one with ApplyChatTemplateWith", m.path)
	}
	buf := make([]byte, int(n)+1)
	n = C.llama_chat_apply_template(tmpl, &cMsgs[0], C.size_t(len(cMsgs)), C.bool(addAssistant), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)))
	if n < 0 {
		return "", errorf(ErrTemplate, "llama.cpp rejected the rendered prompt")
	}
	return trimNul(string(buf[:min(int(n), len(buf))])), nil
}
