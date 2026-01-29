package main

import (
	"encoding/json"
	"fmt"
	"go-llama/pkg/llama"
	"log/slog"
	"net/http"
	"os"
)

type ChatRequest struct {
	Messages []llama.ChatMessage `json:"messages"`
	MaxTokens int                 `json:"max_tokens"`
	Temperature float32           `json:"temperature"`
	Stream    bool                `json:"stream"`
}

type ChatResponse struct {
	Content string `json:"content"`
}

type EmbeddingRequest struct {
	Input string `json:"input"`
}

type EmbeddingResponse struct {
	Embedding []float32 `json:"embedding"`
}

var (
	model *llama.Model
	ctx   *llama.Context
)

func main() {
	if len(os.Args) < 2 {
		slog.Error("Usage: ./server <model_path>")
		os.Exit(1)
	}

	llama.Initialize()
	// defer llama.Free() // Server runs forever

	var err error
	slog.Info("Loading model...", "path", os.Args[1])
	model, err = llama.LoadModel(os.Args[1])
	if err != nil {
		slog.Error("Failed to load model", "error", err)
		os.Exit(1)
	}

	params := llama.DefaultContextParams()
	// Enable pooling for embeddings (MEAN = 1)
	params.SetPooling(1) 
	
	ctx, err = model.NewContextWithParams(params)
	if err != nil {
		slog.Error("Failed to create context", "error", err)
		os.Exit(1)
	}

	http.HandleFunc("/v1/chat/completions", handleChat)
	http.HandleFunc("/v1/embeddings", handleEmbeddings)

	slog.Info("Server listening on :8080")
	if err := http.ListenAndServe(":8080", nil); err != nil {
		slog.Error("Server failed", "error", err)
		os.Exit(1)
	}
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req ChatRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	prompt, err := model.ApplyTemplate(req.Messages, true)
	if err != nil {
		http.Error(w, fmt.Sprintf("Template error: %v", err), http.StatusInternalServerError)
		return
	}

	opts := llama.DefaultCompletionOptions()
	if req.MaxTokens > 0 {
		opts.MaxTokens = req.MaxTokens
	}
	if req.Temperature > 0 {
		opts.Temperature = req.Temperature
	}

	if req.Stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		ch, err := ctx.Stream(prompt, opts)
		if err != nil {
			http.Error(w, fmt.Sprintf("Generation error: %v", err), http.StatusInternalServerError)
			return
		}

		for piece := range ch {
			data, _ := json.Marshal(ChatResponse{Content: piece})
			fmt.Fprintf(w, "data: %s\n\n", data)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
		fmt.Fprintf(w, "data: [DONE]\n\n")
		return
	}

	resp, err := ctx.Generate(prompt, opts)
	if err != nil {
		http.Error(w, fmt.Sprintf("Generation error: %v", err), http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(ChatResponse{Content: resp})
}

func handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var req EmbeddingRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Bad request", http.StatusBadRequest)
		return
	}

	emb, err := ctx.Embeddings(req.Input)
	if err != nil {
		http.Error(w, fmt.Sprintf("Embedding error: %v", err), http.StatusInternalServerError)
		return
	}

	json.NewEncoder(w).Encode(EmbeddingResponse{Embedding: emb})
}
