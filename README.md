# go-llama

Go bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp) through cgo.
Load a GGUF model, generate text as a blocking call or an `iter.Seq2` stream,
apply the model's own chat template, embed text, and read llama.cpp's timings.
The C++ source is a git submodule, built as shared libraries the Go package
links against.

The API surface is small on purpose: `Model` holds weights and vocabulary,
`Context` holds a KV cache, and everything that touches the KV cache goes
through one mutex per context.

## Requirements

- Go 1.25 or newer, with cgo enabled (a `CGO_ENABLED=0` build has nothing to link).
- A C and C++ toolchain (`gcc` and `g++`, or clang), plus `cmake` for the llama.cpp build.
- `git`, to check out the `llama.cpp` submodule.
- Linux. See [Supported platforms](#supported-platforms).
- Optional: an NVIDIA GPU and a CUDA toolkit for GPU offload. The build works
  without one.

## Quickstart

```sh
git clone --recurse-submodules git@github.com:samcharles93/go-llama.git
cd go-llama
make llama
```

`make llama` initialises the submodule and runs `scripts/build-llama.sh`, which
builds CUDA when it finds a usable toolkit and an NVIDIA GPU, and a CPU-only
build otherwise. Force the choice with `GGML_BACKEND=cpu make llama` or
`GGML_BACKEND=cuda make llama`. The shared libraries land in
`llama.cpp/build/bin`, which `pkg/llama/cgo.go` links against with the rpath
baked in, so no `LD_LIBRARY_PATH` is needed.

Then a first program, in `cmd/hello/main.go`:

```go
package main

import (
	"context"
	"fmt"
	"log"
	"os"

	"go-llama/pkg/llama"
)

func main() {
	model, err := llama.LoadModel(os.Args[1])
	if err != nil {
		log.Fatal(err)
	}
	defer model.Close()

	c, err := model.NewContext()
	if err != nil {
		log.Fatal(err)
	}
	defer c.Close()

	res, err := c.Chat(context.Background(), []llama.ChatMessage{
		llama.SystemMessage("You are a concise assistant."),
		llama.UserMessage("Why is the sky blue? One sentence."),
	}, llama.DefaultCompletionOptions())
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(res.Text)
}
```

```sh
go run ./cmd/hello /path/to/model.gguf
```

The module path is `go-llama`, so a program outside this tree needs
`go mod edit -replace go-llama=/path/to/go-llama` plus `go mod edit -require
go-llama@v0.0.0`.

## API tour

The snippets below are functions in `package main`, with `go-llama/pkg/llama`
imported. Each one runs against a real model.

### Loading a model

```go
func loadModel(path string) error {
	model, err := llama.LoadModel(path, llama.ModelParams{
		GPULayers: 0, // 0 = CPU only, -1 = every layer on the GPU
		UseMmap:   true,
	})
	if err != nil {
		return fmt.Errorf("load %s: %w", path, err)
	}
	defer model.Close()

	fmt.Println(model.Description())
	fmt.Printf("%.2fB params, trained context %d, embedding width %d\n",
		float64(model.Params())/1e9, model.ContextSize(), model.EmbeddingSize())
	return nil
}
```

`LoadModel` with no params offloads every layer when a GPU is present, which is
`DefaultModelParams`. A second `ModelParams` argument replaces that wholesale, so
any field you set means setting `UseMmap` too. Failures wrap `llama.ErrModelLoad`
for a missing file, a wrong format, or a corrupt download.

### Generating

```go
func generate(path string) error {
	model, err := llama.LoadModel(path)
	if err != nil {
		return err
	}
	defer model.Close()

	c, err := model.NewContext(llama.ContextOptions{ContextSize: 4096})
	if err != nil {
		return err
	}
	defer c.Close()

	prompt, err := model.ApplyChatTemplate([]llama.ChatMessage{
		llama.SystemMessage("You are a concise assistant."),
		llama.UserMessage("Name three uses for a paperclip."),
	}, true)
	if err != nil {
		return err
	}

	opts := llama.DefaultCompletionOptions()
	opts.MaxTokens = 200
	opts.Stop = []string{"\n\nUser:"}

	res, err := c.Generate(context.Background(), prompt, opts)
	if err != nil {
		return err
	}
	fmt.Println(res.Text)
	fmt.Printf("%s after %d tokens, %.1f tok/s\n", res.StopReason, len(res.Tokens), res.Metrics.TokensPerSecond())
	return nil
}
```

`Result` carries the text, the generated token ids, why generation ended
(`StopReason`), how many prompt tokens came from the KV cache (`ReusedTokens`),
and llama.cpp's timings.

Instruct models expect their template. A bare prompt against one often produces a
single end-of-generation token and nothing else, which is why the snippet renders
first. `Chat` below does the same in one call.

### Streaming

```go
func stream(path string) error {
	model, err := llama.LoadModel(path)
	if err != nil {
		return err
	}
	defer model.Close()

	c, err := model.NewContext()
	if err != nil {
		return err
	}
	defer c.Close()

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	pieces := 0
	it, err := c.ChatStream(ctx, []llama.ChatMessage{
		llama.UserMessage("Count from one to twenty."),
	}, llama.GreedyCompletionOptions(60))
	if err != nil {
		return err
	}
	for piece, err := range it {
		if err != nil {
			return err
		}
		fmt.Print(piece)
		pieces++
		if pieces == 10 {
			break // stops generation and leaves the KV cache consistent
		}
	}
	fmt.Println()
	return nil
}
```

`Stream` and `ChatStream` return `iter.Seq2[string, error]`. Pieces are
token-aligned, so print them as they come and never trim them. A piece never
splits a UTF-8 rune: the stop matcher holds bytes back until the rune completes,
and it also keeps a piece back while it could still be the start of a stop
sequence.

Breaking out of the loop ends generation at the next token: no goroutine is left
behind, the context stays usable, and whatever was generated stays in the KV
cache. An error arrives as the second value of the iteration, so check it before
using the piece.

### Chat

```go
func chat(path string) error {
	model, err := llama.LoadModel(path)
	if err != nil {
		return err
	}
	defer model.Close()

	c, err := model.NewContext()
	if err != nil {
		return err
	}
	defer c.Close()

	messages := []llama.ChatMessage{
		llama.SystemMessage("You answer in one short sentence."),
		llama.UserMessage("What is the capital of Portugal?"),
	}
	res, err := c.Chat(context.Background(), messages, llama.DefaultCompletionOptions())
	if err != nil {
		return err
	}
	fmt.Println(res.Text)

	messages = append(messages, llama.AssistantMessage(res.Text), llama.UserMessage("And of Spain?"))
	res, err = c.Chat(context.Background(), messages, llama.DefaultCompletionOptions())
	if err != nil {
		return err
	}
	fmt.Printf("%s\n(prompt %d tokens, %d served from the KV cache)\n", res.Text, res.PromptTokens, res.ReusedTokens)
	return nil
}
```

`Chat` renders the template stored in the GGUF metadata and leaves the assistant
turn open. `Model.ChatTemplate("")` shows what the file carries;
`ApplyChatTemplateWith` takes an explicit template for models whose embedded one
llama.cpp cannot parse. Failures wrap `llama.ErrTemplate`.

### Embeddings

```go
func embed(path string) error {
	model, err := llama.LoadModel(path)
	if err != nil {
		return err
	}
	defer model.Close()

	c, err := model.NewContext(llama.ContextOptions{
		ContextSize: 512,
		Embeddings:  true,
		Pooling:     llama.PoolingMean,
	})
	if err != nil {
		return err
	}
	defer c.Close()

	vec, err := c.Embed(context.Background(), "the cat sat on the mat")
	if err != nil {
		return err
	}
	vecs, err := c.EmbedBatch(context.Background(), []string{"a dog barked", "a cat sat quietly"})
	if err != nil {
		return err
	}
	fmt.Printf("%d dimensions, batch of %d\n", len(vec), len(vecs))
	fmt.Printf("cosine similarity %.3f\n", cosine(vec, vecs[1]))
	return nil
}

func cosine(a, b []float32) float64 {
	var dot, na, nb float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		na += float64(a[i]) * float64(a[i])
		nb += float64(b[i]) * float64(b[i])
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}
```

`EmbedBatch` decodes one text at a time. Submitting several texts as one
multi-sequence batch looks tempting and is wrong: llama.cpp pools a sequence's
embedding per micro-batch, and sequences of different lengths in one decode come
back pooled with the wrong token set (measured on a Qwen2 0.5B model, one sequence
of three texts was off by a cosine of 0.56 while the other two matched exactly).
Pooled embeddings are computed per micro-batch, so a text longer than
`ContextOptions.UBatchSize` (512 by default) cannot be embedded in one piece: raise
`UBatchSize` for long documents, and `Embed` returns an error naming it otherwise.

An embedding context must set `Embeddings: true`; a chat context cannot be reused
for embeddings. `PoolingMean`, `PoolingCLS` and `PoolingLast` need a model trained
for pooled output, which is what embedding GGUFs are. With `PoolingNone` (the
default) the vector is the final token's hidden state, which is all a
decoder-only model produces. Vectors come back unnormalised, so apply L2
normalisation yourself if your index compares with cosine. Every call clears the
KV cache first, which makes batch embedding of unrelated documents safe.

### Cancellation

Cancelling the Go context stops generation at the next token, and also
interrupts a long prompt evaluation: on the CPU and Metal backends llama.cpp
polls an abort flag mid-decode, and on CUDA the prompt stops at the next batch
boundary (`ContextOptions.BatchSize`). The partial `Result` comes back with an
error satisfying `errors.Is(err, context.Canceled)`, and the context stays
usable.

```go
func cancel(path string) error {
	model, err := llama.LoadModel(path)
	if err != nil {
		return err
	}
	defer model.Close()

	c, err := model.NewContext()
	if err != nil {
		return err
	}
	defer c.Close()

	ctx, cancel := context.WithCancel(context.Background())
	time.AfterFunc(300*time.Millisecond, cancel)
	defer cancel()

	res, err := c.Generate(ctx, "Write a long essay about rivers.", llama.DefaultCompletionOptions())
	switch {
	case errors.Is(err, context.Canceled):
		fmt.Printf("stopped early with %d tokens: %q\n", len(res.Tokens), res.Text)
	case err != nil:
		return err
	default:
		fmt.Printf("finished before the deadline: %s\n", res.StopReason)
	}
	return nil
}
```

Cancelling stops at the next token and returns the partial `Result` alongside an
error that satisfies `errors.Is(err, context.Canceled)` and
`errors.Is(err, llama.ErrDecode)`. A `context.WithTimeout` that expires gives
`context.DeadlineExceeded` instead, since those are distinct sentinel errors.

### Metrics

```go
func metrics(path string) error {
	model, err := llama.LoadModel(path)
	if err != nil {
		return err
	}
	defer model.Close()

	c, err := model.NewContext(llama.ContextOptions{DisablePerf: false})
	if err != nil {
		return err
	}
	defer c.Close()

	c.ResetMetrics()
	start := time.Now()
	if _, err := c.Generate(context.Background(), "Summarise the plot of Hamlet.", llama.DefaultCompletionOptions()); err != nil {
		return err
	}
	m := c.Metrics()
	fmt.Printf("prompt: %d tokens at %.0f tok/s\n", m.PromptTokens, m.PromptTokensPerSecond())
	fmt.Printf("generate: %d tokens at %.0f tok/s\n", m.GeneratedTokens, m.TokensPerSecond())
	fmt.Printf("wall clock: %s\n", time.Since(start).Round(time.Millisecond))
	return nil
}
```

`Metrics()` is cumulative for the context; `ResetMetrics()` zeroes the counters.
Prompt tokens served from the KV cache are not counted as prompt evaluation, so
the prompt rate reads high on a warm cache. llama.cpp disables its perf counters
by default; this package turns them back on unless `DisablePerf` is set.

### Concurrency

```go
func concurrent(path string) error {
	model, err := llama.LoadModel(path)
	if err != nil {
		return err
	}
	defer model.Close()

	prompts := []string{"Say hello.", "Name a fruit.", "Name a colour.", "Count to three."}
	results := make([]string, len(prompts))
	errs := make([]error, len(prompts))

	var wg sync.WaitGroup
	for i, prompt := range prompts {
		wg.Add(1)
		go func() {
			defer wg.Done()
			c, err := model.NewContext()
			if err != nil {
				errs[i] = err
				return
			}
			defer c.Close()
			res, err := c.Chat(context.Background(), []llama.ChatMessage{
				llama.UserMessage(prompt),
			}, llama.GreedyCompletionOptions(32))
			if err != nil {
				errs[i] = err
				return
			}
			results[i] = res.Text
		}()
	}
	wg.Wait()

	for i, err := range errs {
		if err != nil {
			return err
		}
		fmt.Printf("%q -> %q\n", prompts[i], results[i])
	}
	return nil
}
```

One model serves every goroutine; each request gets its own context and its own
KV cache. The weights are the expensive part and are loaded once: a 1.2B
`Q4_K_M` file costs about 0.7 GB whatever the number of contexts, and each
context adds its KV cache on top of that.

### Sessions and long conversations

```go
func resume(path string) error {
	model, err := llama.LoadModel(path)
	if err != nil {
		return err
	}
	defer model.Close()

	c, err := model.NewContext()
	if err != nil {
		return err
	}
	defer c.Close()

	if _, err := c.Generate(context.Background(), "Explain mutexes in one sentence.", llama.GreedyCompletionOptions(24)); err != nil {
		return err
	}

	// Persist the KV cache plus the token history, then restore it later instead
	// of paying for the prompt again.
	state, err := c.SaveState()
	if err != nil {
		return err
	}
	other, err := model.NewContext()
	if err != nil {
		return err
	}
	defer other.Close()
	if err := other.LoadState(state); err != nil {
		return err
	}

	// Trim what a conversation no longer needs.
	if err := other.Truncate(other.TokenCount() / 2); err != nil {
		return err
	}
	fmt.Println(len(other.History()), "tokens still cached")

	// Keep a chat inside the window: FitMessages drops the oldest turns and never
	// drops the system prompt or the newest message.
	fitted, err := model.FitMessages([]llama.ChatMessage{
		llama.SystemMessage("Answer briefly."),
		llama.UserMessage("A question with a long history behind it."),
	}, c.ContextSize()-256)
	if err != nil {
		return err
	}
	fmt.Println(len(fitted), "messages fit")
	return nil
}
```

`Model.CountTokens` reports how many tokens a text occupies, which is what a
server needs before it decides a conversation no longer fits.
`Model.Metadata` returns the GGUF key/value pairs (architecture, quantisation,
trained context), and `Model.IsHybrid` / `IsRecurrent` report whether the model can
trim its own state at all.

## KV cache reuse

A context keeps the tokens currently in its KV cache and, on each call, compares
them with the new prompt. The longest common prefix stays where it is, anything
after it is dropped from the cache, and only the tail is evaluated. A chat turn
therefore costs the tokens you appended, not the whole conversation.

Measured on the model below, second turn of a two-turn conversation:
353 of 370 prompt tokens reused, 17 evaluated.

Consequences worth knowing:

- `Result.ReusedTokens` tells you what the prefix match bought you.
- `Context.TokenCount()` reports how many tokens are in the cache.
- `Reset()` drops the cache, which is how you start a fresh conversation on a
  reused context.
- An interrupted generation leaves the tokens it produced in the cache, so the
  next turn can continue from them.
- Embedding contexts clear the cache on every `Embed` call.

## Threading and locking

- A `Model` is immutable after loading and safe for concurrent use. Load one, open
  as many contexts as you need.
- A `Context` serialises generation behind its own mutex, because one KV cache
  serves one decode at a time. Two goroutines calling `Generate` on the same
  context are serialised, but they share one KV cache and one token history, so
  each call sees whatever the other left behind. One context per conversation,
  or per in-flight request.
- Zero `Threads` and `ThreadsBatch` mean `llama.DefaultThreads()`, one thread
  per physical performance core, rather than `runtime.NumCPU()`: SMT siblings
  and E-cores slow ggml down because every step waits for its slowest thread.
- Parallelism means several contexts over one model. Each context allocates its
  own KV cache, which is where memory goes.
- `Init` is idempotent and `LoadModel` calls it for you. `Shutdown` is a no-op
  while any model is open. Processes that never call either are fine, since exit
  does the same work.
- llama.cpp logs from C threads, so the log bridge locks while it assembles
  fragments. `SetLogLevel` and `SetLogger` replace the destination;
  `FlushLogs()` writes a held-back partial line before exit.
- The sampler chain is built per generation from `CompletionOptions`, and a
  grammar is parsed each time it is used rather than cached.

## Performance

Measured on this machine, both backends, one run each:

| Backend | Prompt eval | Generation |
| --- | --- | --- |
| GPU, every layer offloaded | 2483 tok/s | 190 tok/s |
| CPU only (`GPULayers: 0`) | 682 tok/s | 38 tok/s |

- Model: `LFM2.5-1.2B-Instruct-Q4_K_M.gguf`, 1.17B params, 0.7 GB.
- Hardware: i5-12500H (12 cores, 16 threads), RTX 3050 Laptop with 4 GB, 62 GB RAM.
- Build: CUDA 13.3, `CMAKE_CUDA_ARCHITECTURES=86`, `Release`, shared libraries.
- Workload: greedy decoding, 98-token prompt, 255 tokens generated,
  `ContextSize: 4096`, mmap on.
- Counters: `Result.Metrics` from `llama_perf_context`, cleared before the run.

The gap is generation, not prompt processing: 190 against 38 tok/s. Layer
offload is worth having even on a 4 GB card.

`make bench` runs the package benchmarks and `make test-integration` runs the
end-to-end tests; both need a model.

## Supported platforms

Linux is the tested target: x86_64, glibc, gcc or clang, CUDA 13.3 or a CPU-only
build. The rpath flag in `pkg/llama/cgo.go` and the `-lstdc++` link assume an ELF
toolchain.

macOS and Windows are untested and need changes to that flag block (`.dylib`
against `.so`, `libc++` instead of `libstdc++`, no rpath). ARM and other
architectures are untested; `GGML_NATIVE=ON` means the llama.cpp build targets
whatever CPU compiles it, so a binary built on one machine is not portable to
another.

Go 1.25 is the floor because the module says `go 1.25.5`. The features it leans
on (`iter.Seq2`, `min` and `max`, range over integers) would compile from 1.23,
so the module directive is the constraint.

## Known limits

- LoRA adapters are applied to a context rather than a model (`Model.LoadLora`,
  `LoraAdapter.Apply`), so a server can serve one base model with a different
  adapter per request. Adapters are freed with the model, not individually.
- No multimodal input. The `mtmd` library is built but not bound, so image and
  audio models load as text-only at best.
- No function calling. Chat templates render, but tool schemas, tool calls and
  tool results are not parsed or validated.
- Grammars are supported through `CompletionOptions.Grammar` and parsed on every
  generation, not cached between calls. The grammar must be satisfiable by the
  model's own vocabulary: if no token can continue the grammar, llama.cpp throws
  a C++ exception that aborts the process rather than returning an error, so
  write grammars that allow the tokenizer's surface forms (an optional leading
  space is the usual fix).
- One sequence per context by default. `ContextOptions.Sequences` raises that for
  hand-built batches through `Context.Decode`, but `Generate` and `Stream` use
  sequence 0.
- The default context is capped at `MaxDefaultContextSize` (8192 tokens) instead
  of the model's trained window, which can be 128k and cost gigabytes of KV
  cache. Set `ContextSize` to go beyond it.
- No speculative decoding and no token healing: one decode call per generated
  token.

## Build flags

| Variable | Effect |
| --- | --- |
| `GGML_BACKEND` | `auto` (default), `cuda` or `cpu`. `auto` picks CUDA when `scripts/cuda-toolkit.sh` finds a working toolkit and an NVIDIA GPU is present. |
| `CUDA_ARCH` | Pins `CMAKE_CUDA_ARCHITECTURES`, e.g. `CUDA_ARCH=86`. Defaults to the compute capability `nvidia-smi` reports. |
| `CUDA_HOME`, `CUDA_PATH` | Preferred toolkit directories, tried before `/usr/local/cuda*`. |
| `FORCE_TOOLKIT_PROBE` | `1` re-probes the CUDA toolkits instead of using the cached answer for this llama.cpp checkout. |
| `JOBS` | Parallel build jobs, defaulting to `nproc`. |
| `CGO_CFLAGS`, `CGO_LDFLAGS` | Override the include paths and link flags in `pkg/llama/cgo.go`, for an installed llama.cpp instead of the submodule. |
| `TEST_MODEL`, `EMBED_MODEL` | GGUF files for `make test-integration` and `make bench`. Both default to `models/*.gguf`. |
| `GO_LLAMA_LOG_LEVEL` | `debug`, `info`, `warn` (default) or `error` for llama.cpp's own logging. `llama.SetLogLevel` overrides it at runtime. |

The targets that matter here:

| Target | What it does |
| --- | --- |
| `make llama` | Builds llama.cpp into `llama.cpp/build/bin`. |
| `make build` | Builds the package and the examples. |
| `make test` | Unit tests, no model file needed. |
| `make test-integration` | End-to-end tests, needs `TEST_MODEL`. |
| `make test-model` | Downloads the small GGUFs the integration tests use. |
| `make bench` | Benchmarks, needs `TEST_MODEL`. |
| `make check` | `fmt-check`, `vet`, `lint` and `test`. |
| `make fmt`, `make fmt-check` | gofumpt write and check. |

CI (`.github/workflows/ci.yml`) builds llama.cpp CPU-only with the build directory
cached against the submodule commit, then checks formatting, runs `go vet` and
`go test -race ./...`. A second job downloads the integration models and runs the
tests behind the `integration` tag.

## Troubleshooting

Build failures caused by CUDA toolkits, stale CMake caches, library layout and
VRAM limits are covered in [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
