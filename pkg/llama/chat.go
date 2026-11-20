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

// SystemMessage builds a system turn.
func SystemMessage(content string) ChatMessage { return ChatMessage{Role: "system", Content: content} }

// UserMessage builds a user turn.
func UserMessage(content string) ChatMessage { return ChatMessage{Role: "user", Content: content} }

// AssistantMessage builds an assistant turn, which is what you append after a
// reply so the next turn sees it.
func AssistantMessage(content string) ChatMessage {
	return ChatMessage{Role: "assistant", Content: content}
}

// ChatTemplate returns the chat template embedded in the GGUF metadata, or ""
// when the model has none. name selects a named template; "" takes the default.
//
// Names are suffixes of llama.cpp's metadata key: "" reads
// tokenizer.chat_template, and "tool_use" reads tokenizer.chat_template.tool_use.
// A name the file does not carry returns "".
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
// addAssistant is llama_chat_apply_template's add_ass flag: it appends the
// assistant turn's opening marker with no content, which is the state a prompt
// has to be in before generating a reply. Pass false to render a finished
// conversation.
//
// llama_chat_apply_template would fall back to the built-in chatml template when
// handed no template, which renders another family's prompt for any model that
// is not chatml. A model whose GGUF carries no template is therefore an error
// naming the model; ApplyChatTemplateWith takes an explicit template instead.
func (m *Model) ApplyChatTemplate(messages []ChatMessage, addAssistant bool) (string, error) {
	if err := validateChatMessages(messages); err != nil {
		return "", err
	}
	if m == nil || m.closed.Load() {
		return "", ErrClosed
	}
	if len(messages) == 0 {
		return "", nil
	}
	tmpl := m.ChatTemplate("")
	if tmpl == "" {
		return "", errorf(ErrTemplate, "model %s has no chat template in its GGUF metadata (tokenizer.chat_template); render the prompt yourself or pass a template to ApplyChatTemplateWith", m.path)
	}
	return m.applyChatTemplate(tmpl, messages, addAssistant)
}

// ApplyChatTemplateWith renders messages with an explicit Jinja-style template
// string instead of the one in the model. Use it for models whose embedded
// template llama.cpp does not recognise; see llama.cpp's
// templates-supported-by-llama_chat_apply_template for the accepted subset.
//
// Passing "chatml" reproduces the fallback llama.cpp uses when a model carries no
// template at all.
func (m *Model) ApplyChatTemplateWith(tmpl string, messages []ChatMessage, addAssistant bool) (string, error) {
	if err := validateChatMessages(messages); err != nil {
		return "", err
	}
	if m == nil || m.closed.Load() {
		return "", ErrClosed
	}
	if tmpl == "" {
		return "", errorf(ErrTemplate, "no template was passed for model %s", m.path)
	}
	if len(messages) == 0 {
		return "", nil
	}
	return m.applyChatTemplate(tmpl, messages, addAssistant)
}

// validateChatMessages rejects turns that cannot be rendered. An empty role makes
// llama.cpp emit a marker with no name ("<|im_start|>") which the model then reads
// as part of the conversation, so it is caught here rather than silently shipped.
func validateChatMessages(messages []ChatMessage) error {
	for i, msg := range messages {
		if msg.Role == "" {
			return errorf(ErrTemplate, "message %d has an empty role; use \"system\", \"user\" or \"assistant\"", i)
		}
	}
	return nil
}

// applyChatTemplate renders non-empty messages with tmpl, which must not be
// empty.
func (m *Model) applyChatTemplate(tmpl string, messages []ChatMessage, addAssistant bool) (string, error) {
	cTmpl := C.CString(tmpl)
	defer C.free(unsafe.Pointer(cTmpl))

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

	n := C.llama_chat_apply_template(cTmpl, &cMsgs[0], C.size_t(len(cMsgs)), C.bool(addAssistant), nil, 0)
	if n < 0 {
		return "", errorf(ErrTemplate, "llama.cpp does not recognise the %d character chat template of model %s; it applies a fixed set of templates, so check the spelling or render the prompt yourself", len(tmpl), m.path)
	}
	buf := make([]byte, int(n)+1)
	n = C.llama_chat_apply_template(cTmpl, &cMsgs[0], C.size_t(len(cMsgs)), C.bool(addAssistant), (*C.char)(unsafe.Pointer(&buf[0])), C.int32_t(len(buf)))
	if n < 0 {
		return "", errorf(ErrTemplate, "llama.cpp rejected the rendered prompt for model %s", m.path)
	}
	return trimNul(string(buf[:min(int(n), len(buf))])), nil
}
