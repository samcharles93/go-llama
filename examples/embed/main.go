// Command embed prints JSON embeddings for text given as arguments or on stdin.
//
//	go run ./examples/embed -model models/embed.gguf "a cat" "a dog"
//	printf 'a cat\na dog\n' | go run ./examples/embed -model models/embed.gguf
//	go run ./examples/embed -model models/embed.gguf -cosine "a cat" "a dog"
//
// With -cosine the two inputs are embedded and only their cosine similarity is
// printed, which is the quickest way to check that a model is loaded with the
// right pooling.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
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
	modelPath := flag.String("model", "", "path to an embedding GGUF model (required)")
	normalize := flag.Bool("normalize", true, "scale every vector to unit length")
	cosine := flag.Bool("cosine", false, "print the cosine similarity of exactly two inputs")
	ngl := flag.Int("ngl", 0, "GPU layers to offload (0 keeps the model on the CPU)")
	ctxSize := flag.Uint("ctx", 2048, "context size for the embedding context")
	poolName := flag.String("pool", "mean", "pooling mode: mean, cls, last or none")
	flag.Parse()

	if *modelPath == "" {
		return errors.New("-model is required")
	}

	pooling, err := parsePooling(*poolName)
	if err != nil {
		return err
	}
	texts, err := inputs(flag.Args())
	if err != nil {
		return err
	}
	if *cosine && len(texts) != 2 {
		return fmt.Errorf("-cosine needs exactly two inputs, got %d", len(texts))
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

	// The pooling mode has to be named: llama.cpp leaves the output buffer empty
	// when the context asks for "unspecified" pooling, and this package returns
	// that empty buffer as a vector of zeros. Mean matches what most sentence
	// encoders, including embeddinggemma, train with.
	ctx, err := model.NewContext(llama.ContextOptions{
		ContextSize: uint32(*ctxSize),
		Embeddings:  true,
		Pooling:     pooling,
	})
	if err != nil {
		return fmt.Errorf("create embedding context (is %s an embedding model?): %w", *modelPath, err)
	}
	defer func() { _ = ctx.Close() }()

	vectors, err := ctx.EmbedBatch(context.Background(), texts)
	if err != nil {
		return err
	}
	for _, vec := range vectors {
		if *normalize {
			normalizeVector(vec)
		}
	}

	enc := json.NewEncoder(os.Stdout)
	if *cosine {
		return enc.Encode(cosineOutput{
			Model:      *modelPath,
			Dim:        len(vectors[0]),
			Normalized: *normalize,
			Pooling:    pooling.String(),
			Cosine:     roundCosine(cosineSimilarity(vectors[0], vectors[1])),
		})
	}

	out := embedOutput{
		Model:      *modelPath,
		Dim:        len(vectors[0]),
		Normalized: *normalize,
		Pooling:    pooling.String(),
		Data:       make([]embedItem, len(texts)),
	}
	for i := range texts {
		vec := make([]float32, len(vectors[i]))
		for j, v := range vectors[i] {
			vec[j] = round(v)
		}
		out.Data[i] = embedItem{Index: i, Text: texts[i], Embedding: vec}
	}
	return enc.Encode(out)
}

type embedOutput struct {
	Model      string      `json:"model"`
	Dim        int         `json:"dim"`
	Normalized bool        `json:"normalized"`
	Pooling    string      `json:"pooling"`
	Data       []embedItem `json:"data"`
}

type embedItem struct {
	Index     int       `json:"index"`
	Text      string    `json:"text"`
	Embedding []float32 `json:"embedding"`
}

type cosineOutput struct {
	Model      string  `json:"model"`
	Dim        int     `json:"dim"`
	Normalized bool    `json:"normalized"`
	Pooling    string  `json:"pooling"`
	Cosine     float64 `json:"cosine"`
}

func parsePooling(name string) (llama.PoolingType, error) {
	switch strings.ToLower(name) {
	case "mean":
		return llama.PoolingMean, nil
	case "cls":
		return llama.PoolingCLS, nil
	case "last":
		return llama.PoolingLast, nil
	case "none":
		return llama.PoolingNone, nil
	default:
		return llama.PoolingUnspecified, fmt.Errorf("unknown pooling mode %q, want mean, cls, last or none", name)
	}
}

// inputs prefers the command line and falls back to one text per stdin line,
// which is what a shell pipeline feeds a batch job.
func inputs(args []string) ([]string, error) {
	var texts []string
	for _, arg := range args {
		if s := strings.TrimSpace(arg); s != "" {
			texts = append(texts, s)
		}
	}
	if len(texts) > 0 {
		return texts, nil
	}
	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		if s := strings.TrimSpace(scanner.Text()); s != "" {
			texts = append(texts, s)
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if len(texts) == 0 {
		return nil, errors.New("no input: pass text as arguments or one text per stdin line")
	}
	return texts, nil
}

func normalizeVector(vec []float32) {
	var sum float64
	for _, v := range vec {
		sum += float64(v) * float64(v)
	}
	norm := math.Sqrt(sum)
	if norm == 0 {
		return
	}
	for i, v := range vec {
		vec[i] = float32(float64(v) / norm)
	}
}

func cosineSimilarity(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		x, y := float64(a[i]), float64(b[i])
		dot += x * y
		na += x * x
		nb += y * y
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

// round keeps the JSON readable: a raw float32 prints up to nine digits and
// nothing downstream needs more than six.
func round(v float32) float32 {
	return float32(math.Round(float64(v)*1e6) / 1e6)
}

func roundCosine(v float64) float64 {
	return math.Round(v*1e6) / 1e6
}
