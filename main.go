// Command go-llama initialises the llama.cpp backend and exits, which is the
// smallest check that the cgo link works.
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
