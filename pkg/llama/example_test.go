// Examples double as compile-checked documentation for the package. Generating
// text, rendering a chat template and embedding a sentence all need a real GGUF
// file, so each example reads one of the test model variables and says what to
// export when it is missing. None of them declares an Output comment, since the
// text a model produces is not something the test run can assert.
package llama_test

import (
	"context"
	"fmt"
	"math"
	"os"

	"go-llama/pkg/llama"
)

func ExampleContext_Generate() {
	path := os.Getenv("GO_LLAMA_TEST_MODEL")
	if path == "" {
		fmt.Println("set GO_LLAMA_TEST_MODEL to run this example")
		return
	}

	model, err := llama.LoadModel(path, llama.ModelParams{GPULayers: 0})
	if err != nil {
		fmt.Println("load model:", err)
		return
	}
	defer model.Close()

	c, err := model.NewContext()
	if err != nil {
		fmt.Println("new context:", err)
		return
	}
	defer c.Close()

	res, err := c.Generate(context.Background(), "Name one primary colour.", llama.GreedyCompletionOptions(32))
	if err != nil {
		fmt.Println("generate:", err)
		return
	}
	fmt.Println(res.Text)
	fmt.Printf("stopped on %v after %d tokens at %.0f tok/s\n",
		res.StopReason, len(res.Tokens), res.Metrics.TokensPerSecond())
}

func ExampleContext_Stream() {
	path := os.Getenv("GO_LLAMA_TEST_MODEL")
	if path == "" {
		fmt.Println("set GO_LLAMA_TEST_MODEL to run this example")
		return
	}

	model, err := llama.LoadModel(path, llama.ModelParams{GPULayers: 0})
	if err != nil {
		fmt.Println("load model:", err)
		return
	}
	defer model.Close()

	c, err := model.NewContext()
	if err != nil {
		fmt.Println("new context:", err)
		return
	}
	defer c.Close()

	// Breaking out of the loop stops generation and leaves the cache usable.
	// This prints the first few pieces and then leaves the loop.
	pieces := 0
	for piece, err := range c.Stream(context.Background(), "Count to five.", llama.GreedyCompletionOptions(24)) {
		if err != nil {
			fmt.Println("stream:", err)
			return
		}
		fmt.Print(piece)
		if pieces++; pieces == 5 {
			break
		}
	}
	fmt.Println()
}

func ExampleModel_ApplyChatTemplate() {
	path := os.Getenv("GO_LLAMA_TEST_MODEL")
	if path == "" {
		fmt.Println("set GO_LLAMA_TEST_MODEL to run this example")
		return
	}

	model, err := llama.LoadModel(path, llama.ModelParams{GPULayers: 0})
	if err != nil {
		fmt.Println("load model:", err)
		return
	}
	defer model.Close()

	// The template lives in the GGUF metadata; a model without one returns an
	// error from ApplyChatTemplate and "" from ChatTemplate.
	prompt, err := model.ApplyChatTemplate([]llama.ChatMessage{
		llama.SystemMessage("You are terse."),
		llama.UserMessage("Name one colour."),
	}, true)
	if err != nil {
		fmt.Println("apply template:", err)
		return
	}
	fmt.Print(prompt)

	c, err := model.NewContext()
	if err != nil {
		fmt.Println("new context:", err)
		return
	}
	defer c.Close()

	res, err := c.Generate(context.Background(), prompt, llama.GreedyCompletionOptions(32))
	if err != nil {
		fmt.Println("generate:", err)
		return
	}
	fmt.Println(res.Text)
}

func ExampleContext_Embed() {
	path := os.Getenv("GO_LLAMA_TEST_EMBED_MODEL")
	if path == "" {
		fmt.Println("set GO_LLAMA_TEST_EMBED_MODEL to run this example")
		return
	}

	model, err := llama.LoadModel(path, llama.ModelParams{GPULayers: 0})
	if err != nil {
		fmt.Println("load model:", err)
		return
	}
	defer model.Close()

	// Embedding contexts cannot generate, and pooling has to match the mode the
	// model was trained with.
	c, err := model.NewContext(llama.ContextOptions{Embeddings: true, Pooling: llama.PoolingMean})
	if err != nil {
		fmt.Println("new context:", err)
		return
	}
	defer c.Close()

	vec, err := c.Embed(context.Background(), "the cat sat on the mat")
	if err != nil {
		fmt.Println("embed:", err)
		return
	}

	// Embed returns the vector unnormalised; divide by this norm before
	// comparing with cosine similarity.
	var sum float64
	for _, f := range vec {
		sum += float64(f) * float64(f)
	}
	fmt.Printf("%d dimensions, norm %.3f\n", len(vec), math.Sqrt(sum))
}
