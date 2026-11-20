package main

import (
	"context"
	"crypto/rand"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"go-llama/pkg/llama"
)

// errQueueFull is returned by acquire when -max-queue is set and that many
// requests are already waiting for a context.
var errQueueFull = errors.New("every context is busy")

// server owns the loaded model, the context pool and the request state.
type server struct {
	model   *llama.Model
	id      string
	created int64

	// pool holds the contexts. Receiving from it is both the semaphore and the
	// wait queue, so no two requests ever share a KV cache.
	pool     chan *llama.Context
	timeout  time.Duration
	apiKey   string
	maxQueue int
	maxBody  int64
	queued   atomic.Int64

	// Embeddings run on their own model and context: a context created with
	// Embeddings enabled clears its KV cache on every call and cannot serve chat
	// traffic, and a chat model's pooled hidden states are not semantic
	// embeddings.
	embedModel          *llama.Model
	allowChatEmbeddings bool
	embedMu             sync.Mutex
	embedCtx            *llama.Context
	embedErr            error
	embedPool           llama.PoolingType
	embedCtxSize        uint32
	embedBatch          int

	base       context.Context
	cancelBase context.CancelFunc
	log        *slog.Logger
}

func newLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
}

func (s *server) routes() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/chat/completions", s.authed(post(s.handleChat)))
	mux.HandleFunc("/v1/completions", s.authed(post(s.handleCompletions)))
	mux.HandleFunc("/v1/embeddings", s.authed(post(s.handleEmbeddings)))
	mux.HandleFunc("/v1/models", s.authed(get(s.handleModels)))
	mux.HandleFunc("/healthz", get(s.handleHealth))
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		writeError(w, http.StatusNotFound, "not_found", "no route for "+r.URL.Path)
	})
	return s.logged(s.recovered(mux))
}

func post(h http.HandlerFunc) http.HandlerFunc {
	return method(http.MethodPost, h)
}

func get(h http.HandlerFunc) http.HandlerFunc {
	return method(http.MethodGet, h)
}

func method(want string, h http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != want {
			w.Header().Set("Allow", want)
			writeError(w, http.StatusMethodNotAllowed, "method_not_allowed",
				fmt.Sprintf("%s does not accept %s, use %s", r.URL.Path, r.Method, want))
			return
		}
		h(w, r)
	}
}

func (s *server) authed(h http.HandlerFunc) http.HandlerFunc {
	if s.apiKey == "" {
		return h
	}
	const prefix = "bearer "
	return func(w http.ResponseWriter, r *http.Request) {
		auth := r.Header.Get("Authorization")
		if len(auth) <= len(prefix) || !strings.EqualFold(auth[:len(prefix)], prefix) ||
			subtle.ConstantTimeCompare([]byte(auth[len(prefix):]), []byte(s.apiKey)) != 1 {
			w.Header().Set("WWW-Authenticate", `Bearer realm="go-llama"`)
			writeError(w, http.StatusUnauthorized, "invalid_api_key", "missing or wrong bearer token")
			return
		}
		h(w, r)
	}
}

func (s *server) logged(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		rec := &statusRecorder{ResponseWriter: w}
		h.ServeHTTP(rec, r)
		s.log.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"status", rec.status,
			"bytes", rec.written,
			"ms", time.Since(start).Milliseconds(),
		)
	})
}

func (s *server) recovered(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if v := recover(); v != nil {
				s.log.Error("panic serving request", "path", r.URL.Path, "panic", fmt.Sprint(v))
				writeError(w, http.StatusInternalServerError, "internal_error", "the server panicked, see its log")
			}
		}()
		h.ServeHTTP(w, r)
	})
}

// statusRecorder remembers the status code and byte count for the request log.
// It forwards Flush so streamed replies keep working through the wrapper.
type statusRecorder struct {
	http.ResponseWriter
	status  int
	written int64
}

func (r *statusRecorder) WriteHeader(code int) {
	if r.status == 0 {
		r.status = code
	}
	r.ResponseWriter.WriteHeader(code)
}

func (r *statusRecorder) Write(b []byte) (int, error) {
	if r.status == 0 {
		r.status = http.StatusOK
	}
	n, err := r.ResponseWriter.Write(b)
	r.written += int64(n)
	return n, err
}

func (r *statusRecorder) Flush() {
	if f, ok := r.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// acquire takes a context from the pool, waiting for one when the pool is
// empty. A waiter counts against -max-queue so the overflow can be told to come
// back instead of piling up.
func (s *server) acquire(ctx context.Context) (*llama.Context, error) {
	select {
	case c := <-s.pool:
		return c, nil
	default:
	}
	if s.maxQueue > 0 {
		if n := s.queued.Add(1); n > int64(s.maxQueue) {
			s.queued.Add(-1)
			return nil, errQueueFull
		}
		defer s.queued.Add(-1)
	}
	select {
	case c := <-s.pool:
		return c, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func (s *server) release(c *llama.Context) { s.pool <- c }

// generate runs one non-streamed completion on a pooled context.
func (s *server) generate(parent context.Context, prompt string, opts llama.CompletionOptions) (llama.Result, error) {
	ctx, cancel := context.WithTimeout(parent, s.timeout)
	defer cancel()

	c, err := s.acquire(ctx)
	if err != nil {
		return llama.Result{}, err
	}
	defer s.release(c)

	// Each request starts from an empty cache: contexts are shared between
	// unrelated clients, and a leftover prefix would leak one request's history
	// into the next one's token accounting.
	if err := c.Reset(); err != nil {
		return llama.Result{}, err
	}
	return c.Generate(ctx, prompt, opts)
}

// streamScope is an SSE response that has a pooled context attached to it.
type streamScope struct {
	w       http.ResponseWriter
	flusher http.Flusher
	srv     *server
	c       *llama.Context
	ctx     context.Context
	cancel  context.CancelFunc
}

// openStream acquires a context and switches the response to server-sent events.
// When it returns false the error response has already been written.
func (s *server) openStream(w http.ResponseWriter, r *http.Request) (*streamScope, bool) {
	ctx, cancel := context.WithTimeout(r.Context(), s.timeout)

	c, err := s.acquire(ctx)
	if err != nil {
		cancel()
		s.writeErrorFor(w, err)
		return nil, false
	}
	if err := c.Reset(); err != nil {
		cancel()
		s.release(c)
		s.writeErrorFor(w, err)
		return nil, false
	}
	flusher, ok := w.(http.Flusher)
	if !ok {
		cancel()
		s.release(c)
		writeError(w, http.StatusInternalServerError, "stream_unsupported", "this connection cannot stream")
		return nil, false
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	return &streamScope{w: w, flusher: flusher, srv: s, c: c, ctx: ctx, cancel: cancel}, true
}

func (sc *streamScope) close() {
	sc.cancel()
	sc.srv.release(sc.c)
}

// send writes one SSE frame. It reports false once the client has gone away.
func (sc *streamScope) send(frame any) bool {
	data, err := json.Marshal(frame)
	if err != nil {
		return false
	}
	if _, err := fmt.Fprintf(sc.w, "data: %s\n\n", data); err != nil {
		return false
	}
	sc.flusher.Flush()
	return true
}

func (sc *streamScope) done() {
	_, _ = fmt.Fprint(sc.w, "data: [DONE]\n\n")
	sc.flusher.Flush()
}

// decodeJSON reads one request body under the size limit.
func (s *server) decodeJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	r.Body = http.MaxBytesReader(w, r.Body, s.maxBody)
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(v); err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			writeError(w, http.StatusRequestEntityTooLarge, "request_too_large",
				fmt.Sprintf("request body exceeds %d bytes", tooBig.Limit))
			return false
		}
		if errors.Is(err, io.EOF) {
			writeError(w, http.StatusBadRequest, "invalid_json", "request body is empty")
			return false
		}
		writeError(w, http.StatusBadRequest, "invalid_json", "request body is not valid JSON: "+err.Error())
		return false
	}
	if err := dec.Decode(new(json.RawMessage)); !errors.Is(err, io.EOF) {
		writeError(w, http.StatusBadRequest, "invalid_json", "request body must contain exactly one JSON object")
		return false
	}
	return true
}

func (s *server) checkModel(w http.ResponseWriter, requested string) bool {
	if requested == "" || requested == s.id {
		return true
	}
	writeError(w, http.StatusNotFound, "model_not_found",
		fmt.Sprintf("model %q is not loaded, this server serves %q", requested, s.id))
	return false
}

// writeErrorFor turns a pool or generation failure into a JSON error response.
func (s *server) writeErrorFor(w http.ResponseWriter, err error) {
	switch {
	case errors.Is(err, errQueueFull):
		w.Header().Set("Retry-After", "1")
		writeError(w, http.StatusTooManyRequests, "queue_full",
			fmt.Sprintf("all %d contexts are busy and %d requests are already waiting", cap(s.pool), s.maxQueue))
	case errors.Is(err, context.Canceled):
		// The client hung up, so there is nobody left to answer.
	case errors.Is(err, context.DeadlineExceeded):
		writeError(w, http.StatusGatewayTimeout, "timeout",
			"the request exceeded the "+s.timeout.String()+" deadline")
	case errors.Is(err, llama.ErrContextFull):
		writeError(w, http.StatusBadRequest, "context_length_exceeded", err.Error())
	case errors.Is(err, llama.ErrClosed):
		writeError(w, http.StatusInternalServerError, "server_shutting_down", err.Error())
	default:
		writeError(w, http.StatusInternalServerError, "generation_failed", err.Error())
	}
}

func (s *server) handleModels(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, modelList{
		Object: "list",
		Data: []modelCard{{
			ID:      s.id,
			Object:  "model",
			Created: s.created,
			OwnedBy: "go-llama",
		}},
	})
}

func (s *server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	busy := cap(s.pool) - len(s.pool)
	writeJSON(w, http.StatusOK, healthBody{
		Status:   "ok",
		Model:    s.id,
		Contexts: cap(s.pool),
		Busy:     busy,
		Queued:   s.queued.Load(),
	})
}

func (s *server) handleChat(w http.ResponseWriter, r *http.Request) {
	var req chatRequest
	if !s.decodeJSON(w, r, &req) {
		return
	}
	if !s.checkModel(w, req.Model) {
		return
	}
	if req.N != nil && *req.N != 1 {
		writeError(w, http.StatusBadRequest, "unsupported", "only n=1 is supported")
		return
	}
	if len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request", "messages must not be empty")
		return
	}

	messages := make([]llama.ChatMessage, 0, len(req.Messages))
	for i, m := range req.Messages {
		role := strings.ToLower(strings.TrimSpace(m.Role))
		switch role {
		case "system", "user", "assistant":
		default:
			writeError(w, http.StatusBadRequest, "invalid_request",
				fmt.Sprintf("messages[%d].role %q is not supported, use system, user or assistant", i, m.Role))
			return
		}
		if strings.TrimSpace(m.Content) == "" {
			writeError(w, http.StatusBadRequest, "invalid_request",
				fmt.Sprintf("messages[%d].content is empty", i))
			return
		}
		messages = append(messages, llama.ChatMessage{Role: role, Content: m.Content})
	}

	opts, err := s.completionOptions(req.samplingRequest)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	prompt, err := s.model.ApplyChatTemplate(messages, true)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "template_failed",
			"the model has no usable chat template, use /v1/completions: "+err.Error())
		return
	}

	if req.Stream {
		s.streamChat(w, r, prompt, opts)
		return
	}
	res, err := s.generate(r.Context(), prompt, opts)
	if err != nil {
		s.writeErrorFor(w, err)
		return
	}
	writeJSON(w, http.StatusOK, chatResponse{
		ID:      newID("chatcmpl"),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   s.id,
		Choices: []chatChoice{{
			Index:        0,
			Message:      chatReply{Role: "assistant", Content: res.Text},
			FinishReason: stopReasonName(res.StopReason),
		}},
		Usage: usageOf(res),
	})
}

func (s *server) streamChat(w http.ResponseWriter, r *http.Request, prompt string, opts llama.CompletionOptions) {
	sc, ok := s.openStream(w, r)
	if !ok {
		return
	}
	defer sc.close()

	id, created := newID("chatcmpl"), time.Now().Unix()
	header := chatChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: s.id,
		Choices: []chunkChoice{{Index: 0, Delta: chunkContent{Role: "assistant"}}},
	}
	if !sc.send(header) {
		return
	}

	var genErr error
	for piece, err := range sc.c.Stream(sc.ctx, prompt, opts) {
		if err != nil {
			genErr = err
			break
		}
		chunk := chatChunk{
			ID: id, Object: "chat.completion.chunk", Created: created, Model: s.id,
			Choices: []chunkChoice{{Index: 0, Delta: chunkContent{Content: piece}}},
		}
		if !sc.send(chunk) {
			// The client went away, so stop generating instead of streaming into
			// a closed socket.
			return
		}
	}

	finish := streamFinishReason(genErr, sc.c.Metrics(), opts.MaxTokens)
	if genErr != nil {
		s.log.Warn("stream ended early", "model", s.id, "err", genErr.Error(), "finish_reason", finish)
	}
	tail := chatChunk{
		ID: id, Object: "chat.completion.chunk", Created: created, Model: s.id,
		Choices: []chunkChoice{{Index: 0, Delta: chunkContent{}, FinishReason: &finish}},
	}
	if sc.send(tail) {
		sc.done()
	}
}

func (s *server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	var req completionRequest
	if !s.decodeJSON(w, r, &req) {
		return
	}
	if !s.checkModel(w, req.Model) {
		return
	}
	prompt, err := parsePrompt(req.Prompt)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}
	opts, err := s.completionOptions(req.samplingRequest)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	if req.Stream {
		s.streamCompletions(w, r, prompt, opts)
		return
	}
	res, err := s.generate(r.Context(), prompt, opts)
	if err != nil {
		s.writeErrorFor(w, err)
		return
	}
	writeJSON(w, http.StatusOK, completionResponse{
		ID:      newID("cmpl"),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   s.id,
		Choices: []completionChoice{{
			Text:         res.Text,
			Index:        0,
			FinishReason: stopReasonName(res.StopReason),
		}},
		Usage: usageOf(res),
	})
}

func (s *server) streamCompletions(w http.ResponseWriter, r *http.Request, prompt string, opts llama.CompletionOptions) {
	sc, ok := s.openStream(w, r)
	if !ok {
		return
	}
	defer sc.close()

	id, created := newID("cmpl"), time.Now().Unix()
	var genErr error
	for piece, err := range sc.c.Stream(sc.ctx, prompt, opts) {
		if err != nil {
			genErr = err
			break
		}
		chunk := completionChunk{
			ID: id, Object: "text_completion", Created: created, Model: s.id,
			Choices: []completionChunkChoice{{Index: 0, Text: piece}},
		}
		if !sc.send(chunk) {
			return
		}
	}

	finish := streamFinishReason(genErr, sc.c.Metrics(), opts.MaxTokens)
	if genErr != nil {
		s.log.Warn("stream ended early", "model", s.id, "err", genErr.Error(), "finish_reason", finish)
	}
	tail := completionChunk{
		ID: id, Object: "text_completion", Created: created, Model: s.id,
		Choices: []completionChunkChoice{{Index: 0, FinishReason: &finish}},
	}
	if sc.send(tail) {
		sc.done()
	}
}

// embedErr carries the first failure out of the batching loop.
type embedErr struct{ err error }

func (e embedErr) Error() string { return e.err.Error() }
func (e embedErr) Unwrap() error { return e.err }

func (s *server) handleEmbeddings(w http.ResponseWriter, r *http.Request) {
	var req embeddingsRequest
	if !s.decodeJSON(w, r, &req) {
		return
	}
	if !s.checkModel(w, req.Model) {
		return
	}
	inputs, err := parseEmbeddingInput(req.Input)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request", err.Error())
		return
	}

	// Embedding requests never touch the pooled chat contexts: a context with
	// Embeddings enabled clears its cache on every call.
	c, err := s.embeddingContext()
	if err != nil {
		writeError(w, http.StatusBadRequest, "embeddings_unavailable", err.Error())
		return
	}

	ctx, cancel := context.WithTimeout(r.Context(), s.timeout)
	defer cancel()

	// EmbedBatch takes as many texts as the context has sequences, so a large
	// request is embedded in fixed-size batches rather than refused.
	batch := max(s.embedBatch, 1)
	vectors := make([][]float32, 0, len(inputs))
	var batchErr embedErr
	for start := 0; start < len(inputs); start += batch {
		end := start + batch
		if end > len(inputs) {
			end = len(inputs)
		}
		part, perr := c.EmbedBatch(ctx, inputs[start:end])
		if perr != nil {
			batchErr = embedErr{perr}
			break
		}
		vectors = append(vectors, part...)
	}
	if batchErr.err != nil {
		err := batchErr.err
		if errors.Is(err, context.DeadlineExceeded) {
			writeError(w, http.StatusGatewayTimeout, "timeout", "the request exceeded the "+s.timeout.String()+" deadline")
			return
		}
		writeError(w, http.StatusBadRequest, "embeddings_failed", err.Error())
		return
	}

	data := make([]embeddingItem, len(inputs))
	tokens := 0
	for i, text := range inputs {
		if allZero(vectors[i]) {
			writeError(w, http.StatusBadRequest, "embeddings_unavailable",
				"the model returned a zero vector: it is not an embedding model, or -embed-pool does not match it")
			return
		}
		data[i] = embeddingItem{Object: "embedding", Index: i, Embedding: vectors[i]}
		if ids, err := s.embeddingModel().Vocab().Tokenize(text, true); err == nil {
			tokens += len(ids)
		}
	}
	writeJSON(w, http.StatusOK, embeddingsResponse{
		Object: "list",
		Data:   data,
		Model:  s.id,
		Usage:  embeddingUsage{PromptTokens: tokens, TotalTokens: tokens},
	})
}

// embeddingModel reports the model embeddings are read from, which is a dedicated
// embedding model unless the operator opted into the chat model.
func (s *server) embeddingModel() *llama.Model {
	if s.embedModel != nil {
		return s.embedModel
	}
	return s.model
}

func (s *server) embeddingContext() (*llama.Context, error) {
	s.embedMu.Lock()
	defer s.embedMu.Unlock()
	if s.embedCtx != nil {
		return s.embedCtx, nil
	}
	if s.embedErr != nil {
		return nil, s.embedErr
	}
	if s.embedModel == nil && !s.allowChatEmbeddings {
		s.embedErr = errors.New("this server has no embedding model: restart it with -embed-model <path/to/embed.gguf>, or pass -embed-from-chat-model if the chat model really is an embedding model")
		return nil, s.embedErr
	}
	if s.embeddingModel().EmbeddingSize() <= 0 {
		s.embedErr = errors.New("the embedding model has no embedding output")
		return nil, s.embedErr
	}
	// One sequence per text in a batch: pooling reads a vector per sequence, so
	// the context has to have room for every text handed to EmbedBatch at once.
	c, err := s.embeddingModel().NewContext(llama.ContextOptions{
		ContextSize: s.embedCtxSize,
		Sequences:   uint32(max(s.embedBatch, 1)),
		Embeddings:  true,
		Pooling:     s.embedPool,
	})
	if err != nil {
		s.embedErr = err
		return nil, err
	}
	s.embedCtx = c
	return c, nil
}

func (s *server) closeEmbedContext() {
	s.embedMu.Lock()
	defer s.embedMu.Unlock()
	if s.embedCtx != nil {
		_ = s.embedCtx.Close()
		s.embedCtx = nil
	}
}

func (s *server) completionOptions(req samplingRequest) (llama.CompletionOptions, error) {
	opts := llama.DefaultCompletionOptions()
	if req.MaxTokens != nil {
		opts.MaxTokens = *req.MaxTokens
	}
	if req.MaxCompletionTokens != nil {
		opts.MaxTokens = *req.MaxCompletionTokens
	}
	if opts.MaxTokens <= 0 {
		return opts, errors.New("max_tokens must be positive")
	}
	if req.Temperature != nil {
		if *req.Temperature < 0 {
			return opts, errors.New("temperature must not be negative")
		}
		opts.Temperature = float32(*req.Temperature)
	}
	if req.TopP != nil {
		if *req.TopP <= 0 || *req.TopP > 1 {
			return opts, errors.New("top_p must be within (0, 1]")
		}
		opts.TopP = float32(*req.TopP)
	}
	stops, err := parseStop(req.Stop)
	if err != nil {
		return opts, err
	}
	opts.Stop = stops
	return opts, nil
}

func parsePrompt(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", errors.New("prompt is required")
	}
	var prompt string
	if err := json.Unmarshal(raw, &prompt); err != nil {
		return "", errors.New("prompt must be a string")
	}
	if strings.TrimSpace(prompt) == "" {
		return "", errors.New("prompt must not be empty")
	}
	return prompt, nil
}

func parseStop(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		if single == "" {
			return nil, errors.New("stop must not be empty")
		}
		return []string{single}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return nil, errors.New("stop must be a string or an array of strings")
	}
	for _, stop := range many {
		if stop == "" {
			return nil, errors.New("stop must not contain an empty string")
		}
	}
	return many, nil
}

func parseEmbeddingInput(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, errors.New("input is required")
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		if strings.TrimSpace(single) == "" {
			return nil, errors.New("input must not be empty")
		}
		return []string{single}, nil
	}
	var many []string
	if err := json.Unmarshal(raw, &many); err != nil {
		return nil, errors.New("input must be a string or an array of strings")
	}
	if len(many) == 0 {
		return nil, errors.New("input must not be empty")
	}
	for i, text := range many {
		if strings.TrimSpace(text) == "" {
			return nil, fmt.Errorf("input[%d] must not be empty", i)
		}
	}
	return many, nil
}

func allZero(vec []float32) bool {
	for _, v := range vec {
		if v != 0 {
			return false
		}
	}
	return true
}

func usageOf(res llama.Result) usage {
	completion := res.Metrics.GeneratedTokens
	return usage{
		PromptTokens:     res.PromptTokens,
		CompletionTokens: completion,
		TotalTokens:      res.PromptTokens + completion,
	}
}

func stopReasonName(reason llama.StopReason) string {
	switch reason {
	case llama.StopReasonMaxTokens, llama.StopReasonContextFull:
		return "length"
	default:
		return "stop"
	}
}

// streamFinishReason names the end of a streamed reply. Stream does not return
// the Result that carries StopReason, so the token cap is read from the perf
// counters instead: generation samples the cap-th token and stops before
// decoding it, which leaves GeneratedTokens one short of MaxTokens.
func streamFinishReason(err error, m llama.Metrics, maxTokens int) string {
	if err != nil {
		return "stop"
	}
	if maxTokens > 0 && m.GeneratedTokens+1 >= maxTokens {
		return "length"
	}
	return "stop"
}

func newID(prefix string) string {
	var b [12]byte
	if _, err := rand.Read(b[:]); err != nil {
		return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
	}
	return prefix + "-" + hex.EncodeToString(b[:])
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	_ = enc.Encode(v)
}

func writeError(w http.ResponseWriter, status int, code, message string) {
	writeJSON(w, status, errorResponse{Error: apiError{Message: message, Type: statusType(status), Code: code}})
}

// statusType mirrors the OpenAI error types, which clients switch on.
func statusType(status int) string {
	switch {
	case status == http.StatusUnauthorized:
		return "authentication_error"
	case status == http.StatusTooManyRequests:
		return "rate_limit_error"
	case status >= 500:
		return "server_error"
	default:
		return "invalid_request_error"
	}
}
