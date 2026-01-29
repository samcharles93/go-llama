package main

import (
	"fmt"
	"go-llama/pkg/llama"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Usage: go run main.go <model_path> [prompt]")
		return
	}

	modelPath := os.Args[1]
	prompt := "Hello, my name is"
	if len(os.Args) > 2 {
		prompt = os.Args[2]
	}

	llama.Initialize()
	defer llama.Free()

	fmt.Printf("Loading model: %s\n", modelPath)
	model, err := llama.LoadModel(modelPath)
	if err != nil {
		fmt.Printf("Error loading model: %v\n", err)
		return
	}
	defer model.Free()

	ctx, err := model.NewContext()
	if err != nil {
		fmt.Printf("Error creating context: %v\n", err)
		return
	}
	defer ctx.Free()

	fmt.Printf("Tokenizing prompt: %q\n", prompt)
	tokens := model.Tokenize(prompt, true, true)

	// Create a batch

batch := llama.NewBatch(512, 0, 1)
	defer batch.Free()

	// Create a sampler chain
	sampler := llama.NewSamplerChain()
	defer sampler.Free()
	sampler.AddTemp(0.7)
	sampler.AddTopK(40)
	sampler.AddTopP(0.9, 1)
	sampler.AddDist(1234)

	// Add prompt to batch
	for i, token := range tokens {
		batch.Add(token, int32(i), []int32{0}, i == len(tokens)-1)
	}

	fmt.Print(prompt)

	// Decode prompt
	if err := ctx.Decode(batch); err != nil {
		fmt.Printf("Decode error: %v\n", err)
		return
	}

	// Main loop
	curLen := len(tokens)
	maxLen := curLen + 50

	for curLen < maxLen {
		// Sample next token
		newToken := sampler.Sample(ctx, -1)
		sampler.Accept(newToken)

		// Print token
		piece := model.TokenToPiece(newToken)
		fmt.Print(piece)

		// Check for EOS (standard is often 2, but depends on model)
		// reliable way is to check against model vocab, but for now we'll just print.

		// Prepare next batch
		batch.Clear()
		batch.Add(newToken, int32(curLen), []int32{0}, true)
		
		curLen++

		// Decode next token
		if err := ctx.Decode(batch); err != nil {
			fmt.Printf("Decode error: %v\n", err)
			break
		}
	}
	fmt.Println()
}
