package main

import (
	"bufio"
	"fmt"
	"go-llama/pkg/llama"
	"os"
	"strings"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <model_path>")
		return
	}

	modelPath := os.Args[1]

	llama.Initialize()
	defer llama.Free()

	model, err := llama.LoadModel(modelPath)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer model.Free()

	ctx, err := model.NewContext()
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}
	defer ctx.Free()

	sampler := llama.NewSamplerChain()
	defer sampler.Free()
	sampler.AddTemp(0.8)
	sampler.AddTopK(40)
	sampler.AddTopP(0.9, 1)
	sampler.AddDist(1234)


batch := llama.NewBatch(512, 0, 1)
	defer batch.Free()

	reader := bufio.NewReader(os.Stdin)
	curPos := int32(0)

	fmt.Println("Chat started. Type 'exit' to quit.")

	for {
		fmt.Print("\nUser > ")
		input, _ := reader.ReadString('\n')
		input = strings.TrimSpace(input)

		if input == "exit" {
			break
		}

		// In a real chat model, we'd apply a template here.
		// For this low-level example, we just feed the raw text.
		tokens := model.Tokenize(input, curPos == 0, true)

		// Process input tokens
		batch.Clear()
		for i, t := range tokens {
			batch.Add(t, curPos+int32(i), []int32{0}, i == len(tokens)-1)
		}
		curPos += int32(len(tokens))

		if err := ctx.Decode(batch); err != nil {
			fmt.Printf("\nDecode error: %v\n", err)
			break
		}

		fmt.Print("AI > ")

		// Generate response tokens
		for i := 0; i < 500; i++ {
			newToken := sampler.Sample(ctx, -1)
			sampler.Accept(newToken)

			piece := model.TokenToPiece(newToken)
			fmt.Print(piece)

			// Simple EOS check (7 is <|im_end|> for LFM, 2 is <|endoftext|>) 
			if newToken == 7 || newToken == 2 {
				break
			}

			batch.Clear()
			batch.Add(newToken, curPos, []int32{0}, true)
			curPos++

			if err := ctx.Decode(batch); err != nil {
				fmt.Printf("\nDecode error: %v\n", err)
				break
			}
		}
		fmt.Println()
	}
}
