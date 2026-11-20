// Command chat is an interactive terminal chat loop over a single GGUF model.
//
//	go run ./examples/chat models/model.gguf
//
// Type a message and press enter. Ctrl-C interrupts the reply, Ctrl-D or "exit"
// leaves.
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

	"go-llama/pkg/llama"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	system := flag.String("system", "You are a helpful assistant.", "system prompt")
	maxTokens := flag.Int("max-tokens", 512, "maximum tokens per reply")
	temp := flag.Float64("temp", 0.7, "sampling temperature")
	ctxSize := flag.Uint("ctx", 0, "context size in tokens (0 = model default)")
	flag.Parse()

	if flag.NArg() < 1 {
		return errors.New("usage: chat [flags] <model.gguf>")
	}

	llama.Init()
	defer llama.Shutdown()
	llama.SetLogLevel(llama.LogLevelWarn)

	model, err := llama.LoadModel(flag.Arg(0))
	if err != nil {
		return err
	}
	defer model.Close()
	fmt.Printf("model: %s\n", model.Description())

	ctx, err := model.NewContext(llama.ContextOptions{ContextSize: uint32(*ctxSize)})
	if err != nil {
		return err
	}
	defer ctx.Close()

	messages := []llama.ChatMessage{llama.SystemMessage(*system)}
	opts := llama.DefaultCompletionOptions()
	opts.MaxTokens = *maxTokens
	opts.Temperature = float32(*temp)

	interrupt, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	reader := bufio.NewReader(os.Stdin)

	for {
		fmt.Print("\n> ")
		line, readErr := reader.ReadString('\n')
		line = strings.TrimSpace(line)
		if readErr != nil && line == "" {
			fmt.Println()
			return nil
		}
		if line == "" {
			continue
		}
		if strings.EqualFold(line, "exit") || strings.EqualFold(line, "quit") {
			return nil
		}
		messages = append(messages, llama.UserMessage(line))

		fmt.Print("\n")
		var reply strings.Builder
		res, err := ctx.Chat(interrupt, messages, opts)
		if err != nil && !errors.Is(err, context.Canceled) {
			return err
		}
		if res.Text != "" {
			reply.WriteString(res.Text)
			fmt.Print(res.Text)
		}
		fmt.Printf("\n[%s, %d tokens, %.1f tok/s]", res.StopReason, len(res.Tokens), res.Metrics.TokensPerSecond())
		if errors.Is(err, context.Canceled) {
			// The interrupted reply stays in the history so the model can pick
			// up from it.
			fmt.Print(" (interrupted)")
			interrupt, stop = signal.NotifyContext(context.Background(), os.Interrupt)
			defer stop()
		}
		messages = append(messages, llama.AssistantMessage(reply.String()))
	}
}
