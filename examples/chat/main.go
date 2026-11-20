// Command chat is an interactive terminal chat loop over one GGUF model.
//
//	go run ./examples/chat -model models/model.gguf
//
// Replies stream token by token and the model's own chat template is applied to
// the whole history on every turn, so llama.cpp's prefix cache keeps the turn
// cheap. Ctrl-C cancels the reply that is running and keeps the partial text in
// the history, so the next message picks up from it. Ctrl-D or /exit leaves.
package main

import (
	"bufio"
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"

	"go-llama/pkg/llama"
)

const defaultSystem = "You are a helpful assistant."

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "path to a GGUF model (required)")
	system := flag.String("system", defaultSystem, "system prompt")
	temp := flag.Float64("temp", 0.7, "sampling temperature (0 selects greedy decoding)")
	topP := flag.Float64("top-p", 0.95, "nucleus sampling probability")
	maxTokens := flag.Int("max-tokens", 256, "maximum tokens per reply")
	ctxSize := flag.Uint("ctx", 0, "KV cache size in tokens (0 = the smaller of the model window and 8192)")
	ngl := flag.Int("ngl", 0, "GPU layers to offload (0 keeps the model on the CPU)")
	raw := flag.Bool("raw", false, "skip the chat template and send User:/Assistant: lines instead")
	flag.Parse()

	if *modelPath == "" {
		return errors.New("-model is required")
	}

	llama.Init()
	defer llama.Shutdown()
	llama.SetLogLevel(llama.LogLevelError)

	model, err := llama.LoadModel(*modelPath, llama.ModelParams{
		GPULayers: int32(*ngl),
		UseMmap:   true,
	})
	if err != nil {
		return err
	}
	defer func() { _ = model.Close() }()

	fmt.Printf("model: %s\n", model.Description())
	if !*raw && model.ChatTemplate("") == "" {
		fmt.Fprintln(os.Stderr, "note: the model embeds no chat template, using -raw style prompts")
		*raw = true
	}

	ctx, err := model.NewContext(llama.ContextOptions{ContextSize: uint32(*ctxSize)})
	if err != nil {
		return err
	}
	defer func() { _ = ctx.Close() }()

	opts := llama.DefaultCompletionOptions()
	opts.MaxTokens = *maxTokens
	opts.Temperature = float32(*temp)
	opts.TopP = float32(*topP)

	s := &session{
		model:    model,
		ctx:      ctx,
		opts:     opts,
		raw:      *raw,
		system:   llama.SystemMessage(*system),
		messages: []llama.ChatMessage{llama.SystemMessage(*system)},
	}

	// One Ctrl-C cancels the running reply. The handler owns no other state, so
	// a signal that arrives while the prompt waits for input is a no-op.
	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigs)
	go s.replies.watch(sigs)

	fmt.Fprintf(os.Stderr, "commands: /reset, /system <text>, /stats, /exit. Ctrl-C interrupts a reply.\n")

	return s.loop()
}

// session holds the conversation and the context it is generated on.
type session struct {
	model  *llama.Model
	ctx    *llama.Context
	opts   llama.CompletionOptions
	raw    bool
	system llama.ChatMessage

	messages []llama.ChatMessage
	replies  trigger
}

func (s *session) loop() error {
	scanner := bufio.NewScanner(os.Stdin)
	interactive := stdinIsTerminal()

	for {
		if interactive {
			fmt.Print("\n> ")
		}
		if !scanner.Scan() {
			if interactive {
				fmt.Println()
			}
			return scanner.Err()
		}
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "/") {
			quit, err := s.command(line)
			if err != nil {
				fmt.Fprintln(os.Stderr, "error:", err)
			}
			if quit {
				return nil
			}
			continue
		}
		if strings.EqualFold(line, "exit") || strings.EqualFold(line, "quit") {
			return nil
		}

		s.messages = append(s.messages, llama.UserMessage(line))
		if err := s.turn(); err != nil {
			return err
		}
	}
}

// command runs a slash command and reports whether the session should end.
func (s *session) command(line string) (quit bool, err error) {
	cmd, arg, _ := strings.Cut(line, " ")
	switch cmd {
	case "/exit", "/quit":
		return true, nil
	case "/reset":
		s.messages = []llama.ChatMessage{s.system}
		return false, s.ctx.Reset()
	case "/system":
		if strings.TrimSpace(arg) == "" {
			fmt.Fprintf(os.Stderr, "system prompt is %q\n", s.system.Content)
			return false, nil
		}
		s.system = llama.SystemMessage(strings.TrimSpace(arg))
		s.messages = []llama.ChatMessage{s.system}
		fmt.Fprintf(os.Stderr, "system prompt replaced, history reset\n")
		return false, s.ctx.Reset()
	case "/stats":
		s.stats()
		return false, nil
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q, try /reset, /system, /stats or /exit\n", cmd)
		return false, nil
	}
}

func (s *session) stats() {
	m := s.ctx.Metrics()
	fmt.Fprintf(os.Stderr, "context: %d/%d tokens in the KV cache\n", s.ctx.TokenCount(), s.ctx.ContextSize())
	fmt.Fprintf(os.Stderr, "history: %d messages, system prompt %q\n", len(s.messages), s.system.Content)
	fmt.Fprintf(os.Stderr, "model:   %s, %d params, %d-wide embeddings\n",
		s.model.Description(), s.model.Params(), s.model.EmbeddingSize())
	fmt.Fprintf(os.Stderr, "last reply: %d prompt tokens at %.0f tok/s, %d generated at %.1f tok/s\n",
		m.PromptTokens, m.PromptTokensPerSecond(), m.GeneratedTokens, m.TokensPerSecond())
}

// turn renders the history, streams one reply and records it in the history.
func (s *session) turn() error {
	prompt, err := s.buildPrompt()
	if err != nil {
		return err
	}

	// A Ctrl-C reaches this cancel, which stops the iterator at the next token.
	ctx, cancel := context.WithCancel(context.Background())
	s.replies.set(cancel)
	defer func() {
		s.replies.set(nil)
		cancel()
	}()

	fmt.Print("\n")
	var reply strings.Builder
	var streamErr error
	for piece, err := range s.ctx.Stream(ctx, prompt, s.opts) {
		if err != nil {
			streamErr = err
			break
		}
		reply.WriteString(piece)
		fmt.Print(piece)
	}

	if text := reply.String(); text != "" {
		s.messages = append(s.messages, llama.AssistantMessage(text))
	}

	m := s.ctx.Metrics()
	switch {
	case errors.Is(streamErr, context.Canceled):
		fmt.Printf("\n[interrupted, %d tokens, partial reply kept]\n", m.GeneratedTokens)
	case errors.Is(streamErr, llama.ErrContextFull):
		fmt.Printf("\n[context full, %d tokens; /reset to start over]\n", m.GeneratedTokens)
	case streamErr != nil:
		return streamErr
	default:
		fmt.Printf("\n[%s, %d tokens, %.1f tok/s]\n", stopLabel(m, s.opts.MaxTokens), m.GeneratedTokens, m.TokensPerSecond())
	}
	return nil
}

// buildPrompt renders the conversation into the prompt for the next turn.
func (s *session) buildPrompt() (string, error) {
	if !s.raw {
		return s.model.ApplyChatTemplate(s.messages, true)
	}
	var b strings.Builder
	for _, m := range s.messages {
		switch m.Role {
		case "system":
			b.WriteString("System: ")
		case "user":
			b.WriteString("User: ")
		default:
			b.WriteString("Assistant: ")
		}
		b.WriteString(strings.TrimSpace(m.Content))
		b.WriteString("\n")
	}
	b.WriteString("Assistant:")
	return b.String(), nil
}

// stopLabel names why a streamed reply ended. Stream does not return the
// Result that carries StopReason, so the label is rebuilt from the perf
// counters: generate stops at MaxTokens after sampling the cap-th token, which
// leaves GeneratedTokens one short of the cap.
func stopLabel(m llama.Metrics, maxTokens int) string {
	if maxTokens > 0 && m.GeneratedTokens+1 >= maxTokens {
		return "max tokens"
	}
	return "end of reply"
}

func stdinIsTerminal() bool {
	fi, err := os.Stdin.Stat()
	if err != nil {
		return false
	}
	return fi.Mode()&os.ModeCharDevice != 0
}

// trigger routes a signal to the cancel function of the reply that is running.
type trigger struct {
	mu     sync.Mutex
	cancel context.CancelFunc
}

func (t *trigger) set(cancel context.CancelFunc) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.cancel = cancel
}

func (t *trigger) watch(sigs <-chan os.Signal) {
	for range sigs {
		t.mu.Lock()
		cancel := t.cancel
		t.mu.Unlock()
		if cancel != nil {
			cancel()
		}
	}
}
