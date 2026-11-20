// Command generate streams one completion and prints throughput afterwards.
//
//	go run ./examples/generate -model models/model.gguf -prompt "Once upon a time"
//
// It is the smallest useful program built on pkg/llama: load, stream, report.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"time"

	"go-llama/pkg/llama"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

func run() error {
	modelPath := flag.String("model", "", "path to a GGUF model")
	prompt := flag.String("prompt", "The capital of France is", "prompt text")
	maxTokens := flag.Int("max-tokens", 256, "maximum tokens to generate")
	temp := flag.Float64("temp", 0, "sampling temperature (0 = greedy)")
	raw := flag.Bool("raw", false, "skip the chat template")
	ctxSize := flag.Uint("ctx", 0, "KV cache size in tokens (0 = the smaller of the model window and 8192)")
	ngl := flag.Int("ngl", -1, "GPU layers to offload (-1 offloads every layer when a GPU is present, 0 keeps the model on the CPU)")
	system := flag.String("system", "You are a helpful assistant.", "system prompt for the chat template")
	flag.Parse()

	if *modelPath == "" {
		return errors.New("-model is required")
	}

	llama.Init()
	defer llama.Shutdown()

	model, err := llama.LoadModel(*modelPath, llama.ModelParams{
		GPULayers: int32(*ngl),
		UseMmap:   true,
	})
	if err != nil {
		return err
	}
	defer func() { _ = model.Close() }()

	ctx, err := model.NewContext(llama.ContextOptions{ContextSize: uint32(*ctxSize)})
	if err != nil {
		return err
	}
	defer func() { _ = ctx.Close() }()

	opts := llama.DefaultCompletionOptions()
	opts.MaxTokens = *maxTokens
	opts.Temperature = float32(*temp)
	opts.Stop = []string{"\n\n\n"}

	text := *prompt
	if !*raw {
		rendered, terr := model.ApplyChatTemplate([]llama.ChatMessage{
			llama.SystemMessage(*system),
			llama.UserMessage(*prompt),
		}, true)
		if terr != nil {
			fmt.Fprintf(os.Stderr, "note: %v\nfalling back to the raw prompt\n", terr)
		} else {
			text = rendered
		}
	}

	interrupt, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()

	ctx.ResetMetrics()
	start := time.Now()
	var genErr error
	for piece, err := range ctx.Stream(interrupt, text, opts) {
		if err != nil {
			genErr = err
			break
		}
		fmt.Print(piece)
	}
	fmt.Println()

	if genErr != nil && !errors.Is(genErr, context.Canceled) {
		return genErr
	}
	m := ctx.Metrics()
	fmt.Fprintf(os.Stderr, "\n[%d tokens in %.2fs, %.1f tok/s]\n",
		m.GeneratedTokens, time.Since(start).Seconds(), m.TokensPerSecond())
	return nil
}
