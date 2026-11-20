package llama

import (
	"errors"
	"strings"
	"testing"
)

// These tests need no model file: they cover the validation that runs before
// llama.cpp is touched. Template rendering needs an instruct model, so it lives
// in chat_integration_test.go.

func TestChatMessageConstructors(t *testing.T) {
	tests := []struct {
		name string
		msg  ChatMessage
		role string
	}{
		{name: "system", msg: SystemMessage("be terse"), role: "system"},
		{name: "user", msg: UserMessage("hello"), role: "user"},
		{name: "assistant", msg: AssistantMessage("hi"), role: "assistant"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if tc.msg.Role != tc.role {
				t.Fatalf("Role = %q, want %q", tc.msg.Role, tc.role)
			}
			if tc.msg.Content == "" {
				t.Fatal("Content was not carried through")
			}
		})
	}
}

func TestApplyChatTemplateRejectsEmptyRole(t *testing.T) {
	// A zero Model never reaches llama.cpp here: the role is checked first.
	m := &Model{}
	_, err := m.ApplyChatTemplate([]ChatMessage{UserMessage("hello"), {Content: "no role"}}, true)
	if !errors.Is(err, ErrTemplate) {
		t.Fatalf("ApplyChatTemplate with an empty role = %v, want ErrTemplate", err)
	}
	if !strings.Contains(err.Error(), "empty role") {
		t.Fatalf("error %q does not name the empty role", err)
	}
}

func TestApplyChatTemplateWithNoMessages(t *testing.T) {
	m := &Model{}
	got, err := m.ApplyChatTemplate(nil, true)
	if err != nil {
		t.Fatalf("ApplyChatTemplate with no messages: %v", err)
	}
	if got != "" {
		t.Fatalf("ApplyChatTemplate with no messages = %q, want an empty prompt", got)
	}
	if got, err := m.ApplyChatTemplateWith("chatml", nil, false); err != nil || got != "" {
		t.Fatalf("ApplyChatTemplateWith with no messages = %q, %v; want an empty prompt", got, err)
	}
}

func TestApplyChatTemplateWithRequiresATemplate(t *testing.T) {
	m := &Model{}
	if _, err := m.ApplyChatTemplateWith("", []ChatMessage{UserMessage("hello")}, true); !errors.Is(err, ErrTemplate) {
		t.Fatalf("ApplyChatTemplateWith(\"\") = %v, want ErrTemplate", err)
	}
}

func TestChatTemplateAfterClose(t *testing.T) {
	m := &Model{}
	m.closed.Store(true)

	if got := m.ChatTemplate(""); got != "" {
		t.Fatalf("ChatTemplate on a closed model = %q, want an empty string", got)
	}
	if _, err := m.ApplyChatTemplate([]ChatMessage{UserMessage("hello")}, true); !errors.Is(err, ErrClosed) {
		t.Fatalf("ApplyChatTemplate on a closed model = %v, want ErrClosed", err)
	}
}
