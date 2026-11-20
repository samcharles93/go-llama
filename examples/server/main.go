// Command server serves one GGUF model over an OpenAI-compatible HTTP API.
//
//	go run ./examples/server -model models/model.gguf
//
//	curl -s localhost:8080/v1/chat/completions \
//	  -d '{"messages":[{"role":"user","content":"Say hi"}]}'
//
// Endpoints: POST /v1/chat/completions, POST /v1/completions,
// POST /v1/embeddings, GET /v1/models, GET /healthz.
//
// Generation runs on a pool of -parallel contexts created from the single
// loaded model. A request that finds every context busy waits for one instead of
// touching a KV cache another request is using; -max-queue caps that wait and
// turns the overflow into 429. SIGINT and SIGTERM drain in-flight requests
// before the process exits.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"strings"
	"syscall"
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
	modelPath := flag.String("model", "", "path to a GGUF model (required)")
	addr := flag.String("addr", "127.0.0.1:8080", "listen address")
	ctxSize := flag.Uint("ctx", 0, "KV cache size per context (0 = the smaller of the model window and 8192)")
	timeout := flag.Duration("timeout", 120*time.Second, "per-request deadline")
	parallel := flag.Int("parallel", 1, "contexts in the generation pool")
	ngl := flag.Int("ngl", 0, "GPU layers to offload (0 keeps the model on the CPU)")
	apiKey := flag.String("api-key", "", "require this bearer token on /v1 routes")
	maxQueue := flag.Int("max-queue", 0, "requests that may wait for a context (0 waits forever)")
	maxBody := flag.Int64("max-body", 1<<20, "largest accepted request body in bytes")
	drain := flag.Duration("drain", 15*time.Second, "how long shutdown waits for in-flight requests")
	embedCtx := flag.Uint("embed-ctx", 2048, "context size for the embeddings context")
	embedPool := flag.String("embed-pool", "mean", "pooling for embeddings: mean, cls, last or none")
	flag.Parse()

	if *modelPath == "" {
		return errors.New("-model is required")
	}
	if *parallel < 1 {
		return fmt.Errorf("-parallel must be at least 1, got %d", *parallel)
	}
	if *maxQueue < 0 || *maxBody < 0 {
		return errors.New("-max-queue and -max-body must not be negative")
	}
	pooling, err := parsePooling(*embedPool)
	if err != nil {
		return err
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
	defer model.Close()

	s := &server{
		model:        model,
		id:           modelID(*modelPath),
		created:      time.Now().Unix(),
		pool:         make(chan *llama.Context, *parallel),
		timeout:      *timeout,
		apiKey:       *apiKey,
		maxQueue:     *maxQueue,
		maxBody:      *maxBody,
		embedPool:    pooling,
		embedCtxSize: uint32(*embedCtx),
		log:          newLogger(),
	}

	// One context per parallel slot, all sharing the model's weights.
	contexts := make([]*llama.Context, 0, *parallel)
	defer func() {
		for _, c := range contexts {
			_ = c.Close()
		}
	}()
	for i := 0; i < *parallel; i++ {
		c, err := model.NewContext(llama.ContextOptions{ContextSize: uint32(*ctxSize)})
		if err != nil {
			return fmt.Errorf("create context %d of %d: %w", i+1, *parallel, err)
		}
		contexts = append(contexts, c)
		s.pool <- c
	}
	defer s.closeEmbedContext()

	srv := &http.Server{
		Addr:              *addr,
		Handler:           s.routes(),
		ReadHeaderTimeout: 10 * time.Second,
		// No WriteTimeout: a streamed reply is written for as long as generation
		// runs, and the per-request timeout bounds that instead.
		BaseContext: func(net.Listener) context.Context { return s.base },
	}

	s.base, s.cancelBase = context.WithCancel(context.Background())

	errc := make(chan error, 1)
	go func() {
		errc <- srv.ListenAndServe()
	}()

	s.log.Info("serving",
		"addr", *addr,
		"model", s.id,
		"description", model.Description(),
		"contexts", *parallel,
		"context_size", contexts[0].ContextSize(),
		"timeout", timeout.String(),
	)

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, os.Interrupt, syscall.SIGTERM)
	defer signal.Stop(sigs)

	select {
	case err := <-errc:
		if errors.Is(err, http.ErrServerClosed) {
			return nil
		}
		return err
	case sig := <-sigs:
		s.log.Info("shutting down", "signal", sig.String(), "drain", drain.String())
	}

	// Shutdown stops accepting connections and waits for in-flight handlers. A
	// handler only returns once generation ends, so the drain deadline has to be
	// able to cut a reply short: cancelling the base context aborts the
	// generations that outlive it.
	drainCtx, cancelDrain := context.WithTimeout(context.Background(), *drain)
	defer cancelDrain()
	if err := srv.Shutdown(drainCtx); err != nil {
		s.log.Warn("drain timed out, closing connections", "err", err.Error())
		s.cancelBase()
		_ = srv.Close()
	}
	return nil
}

func modelID(path string) string {
	name := filepath.Base(path)
	return strings.TrimSuffix(name, filepath.Ext(name))
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
		return llama.PoolingUnspecified, fmt.Errorf("unknown -embed-pool %q, want mean, cls, last or none", name)
	}
}
