// Command tokenize prints the token ids and pieces a prompt turns into.
//
//	go run ./examples/tokenize -model models/model.gguf "Hello there"
//	echo "Hello there" | go run ./examples/tokenize -model models/model.gguf
//	go run ./examples/tokenize -model models/model.gguf -chat "Hello there"
//
// With -chat the prompt is first rendered through the model's chat template,
// which is how you see what the template actually sends and where the special
// tokens sit. The piece column is the exact text of each token, so it is also
// the ground truth for choosing stop sequences.
package main

import (
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
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
	modelPath := flag.String("model", "", "path to a GGUF model (required)")
	ngl := flag.Int("ngl", 0, "GPU layers to offload (0 keeps the model on the CPU)")
	noBOS := flag.Bool("no-bos", false, "tokenize without the beginning-of-sequence token")
	chat := flag.Bool("chat", false, "render the prompt through the chat template first")
	system := flag.String("system", "", "system prompt for -chat")
	showTemplate := flag.Bool("show-template", false, "print the model's chat template with -chat")
	flag.Parse()

	if *modelPath == "" {
		return errors.New("-model is required")
	}
	text := strings.TrimSpace(strings.Join(flag.Args(), " "))
	if text == "" {
		stdin, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}
		text = strings.TrimRight(string(stdin), "\n")
	}
	if text == "" {
		return errors.New("no prompt: pass it as arguments or on stdin")
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

	if *chat {
		messages := []llama.ChatMessage{llama.UserMessage(text)}
		if *system != "" {
			messages = append([]llama.ChatMessage{llama.SystemMessage(*system)}, messages...)
		}
		rendered, err := model.ApplyChatTemplate(messages, true)
		if err != nil {
			return err
		}
		if *showTemplate {
			fmt.Printf("template: %q\n", model.ChatTemplate(""))
		}
		fmt.Printf("rendered: %q\n", rendered)
		text = rendered
	}

	vocab := model.Vocab()
	tokens, err := vocab.Tokenize(text, !*noBOS)
	if err != nil {
		return err
	}
	if len(tokens) == 0 {
		return errors.New("the prompt produced no tokens")
	}

	fmt.Printf("model:  %s\n", model.Description())
	fmt.Printf("vocab:  %d tokens, bos %d, eos %d\n", vocab.Size(), vocab.BOS(), vocab.EOS())
	fmt.Printf("prompt: %q\n\n", text)

	// Fixed widths rather than text/tabwriter: right-aligned tabwriter cells put
	// their padding before the cell, which glues the last two columns together.
	fmt.Printf("%5s  %7s  %5s  %-5s  %s\n", "idx", "id", "bytes", "flags", "piece")
	for i, t := range tokens {
		piece, err := vocab.TokenToPiece(t)
		if err != nil {
			return err
		}
		var flags []string
		switch {
		case t == vocab.BOS():
			flags = append(flags, "bos")
		case t == vocab.EOS():
			flags = append(flags, "eos")
		}
		if vocab.IsEOG(t) {
			flags = append(flags, "eog")
		}
		fmt.Printf("%5d  %7d  %5d  %-5s  %q\n", i, int(t), len(piece), strings.Join(flags, ","), piece)
	}

	// Detokenising proves the pieces rejoin into the prompt, which is the check
	// that catches a template that injected tokens the model cannot read back.
	back, err := vocab.Detokenize(tokens)
	if err != nil {
		return err
	}
	fmt.Printf("\n%d tokens, %.2f bytes per token\n", len(tokens), float64(len(text))/float64(len(tokens)))
	if back == text {
		fmt.Println("round trip: detokenised pieces match the prompt")
	} else {
		fmt.Printf("round trip: detokenised pieces differ (special tokens are dropped)\n  %q\n", back)
	}
	return nil
}
