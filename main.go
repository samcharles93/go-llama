package main

import (
	"fmt"
	"go-llama/pkg/llama"
)

func main() {
	fmt.Println("Initializing llama.cpp backend...")
	llama.Initialize()
	defer llama.Free()
	fmt.Println("llama.cpp backend initialized successfully!")
}