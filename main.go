package main

import (
	"fmt"
	"go-llama/pkg/llama"
)

func main() {
	fmt.Println("Initializing llama.cpp backend...")
	llama.Init()
	defer llama.Shutdown()
	fmt.Println("llama.cpp backend initialized successfully!")
}
