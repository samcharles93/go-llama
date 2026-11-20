//go:build integration

package llama

import (
	"os"
	"strings"
	"testing"
)

// These tests need a model with a chat template, so they run under the
// integration tag and skip unless GO_LLAMA_TEST_MODEL is set.

func chatModel(t *testing.T) *Model {
	t.Helper()
	path := os.Getenv("GO_LLAMA_TEST_MODEL")
	if path == "" {
		t.Skip("GO_LLAMA_TEST_MODEL is not set")
	}
	// GPULayers 0 keeps the test on the CPU so parallel lanes do not contend for
	// the card.
	m, err := LoadModel(path, ModelParams{GPULayers: 0})
	if err != nil {
		t.Fatalf("LoadModel(%s): %v", path, err)
	}
	t.Cleanup(func() { _ = m.Close() })
	return m
}

func TestChatTemplateRendersInstructModel(t *testing.T) {
	m := chatModel(t)

	tmpl := m.ChatTemplate("")
	if tmpl == "" {
		t.Fatalf("model %s carries no chat template; the integration test needs an instruct model", m.Path())
	}
	if got := m.ChatTemplate("no-such-template"); got != "" {
		t.Fatalf("ChatTemplate for an absent name = %q, want an empty string", got)
	}

	messages := []ChatMessage{SystemMessage("You are terse."), UserMessage("Name a colour.")}
	withoutAssistant, err := m.ApplyChatTemplate(messages, false)
	if err != nil {
		t.Fatalf("ApplyChatTemplate(addAssistant=false): %v", err)
	}
	withAssistant, err := m.ApplyChatTemplate(messages, true)
	if err != nil {
		t.Fatalf("ApplyChatTemplate(addAssistant=true): %v", err)
	}

	var last int
	for _, want := range []string{"You are terse.", "Name a colour."} {
		i := strings.Index(withoutAssistant, want)
		if i < 0 {
			t.Fatalf("rendered prompt does not contain %q: %q", want, withoutAssistant)
		}
		if i < last {
			t.Fatalf("rendered prompt puts %q out of order: %q", want, withoutAssistant)
		}
		last = i
	}
	// llama_chat_apply_template appends the assistant's opening marker to the
	// same buffer it filled for the finished conversation.
	if !strings.HasPrefix(withAssistant, withoutAssistant) {
		t.Fatalf("addAssistant=true does not extend the finished prompt:\nwithout: %q\nwith:    %q", withoutAssistant, withAssistant)
	}
	if withAssistant == withoutAssistant {
		t.Fatalf("addAssistant=true rendered the same prompt as addAssistant=false: %q", withAssistant)
	}

	// ApplyChatTemplate has to use the template from the GGUF file, which
	// llama.cpp would otherwise replace with its built-in chatml template.
	explicit, err := m.ApplyChatTemplateWith(tmpl, messages, true)
	if err != nil {
		t.Fatalf("ApplyChatTemplateWith(GGUF template): %v", err)
	}
	if explicit != withAssistant {
		t.Fatalf("ApplyChatTemplate did not use the GGUF template:\nfrom GGUF: %q\napplied:   %q", explicit, withAssistant)
	}
}
