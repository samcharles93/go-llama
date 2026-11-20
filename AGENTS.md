# go-llama

Go bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp), via cgo. The
llama.cpp source is a git submodule; the Go package links its shared libraries.

## Layout

```
pkg/llama/            the library
  cgo.go              the only place cgo compile/link flags are set
  llama.go            backend lifecycle, Model, Context
  generate.go         Generate, Stream (iter.Seq2), KV-cache reuse
  batch.go            capacity-checked token batches
  sampler.go          sampler chains built from CompletionOptions
  stop.go             stop-sequence matcher with holdback
  vocab.go            tokenize, detokenize, token to piece
  chat.go             chat templates
  embeddings.go       pooled embeddings
  metrics.go          llama.cpp perf counters
  logging.go          llama.cpp log bridge into slog
  errors.go           sentinel errors and wrapping
  options.go          ModelParams, ContextOptions, CompletionOptions
examples/             runnable programs (chat, server, embed)
scripts/              build and test-model helpers
llama.cpp/            submodule, do not edit
```

## Build and test

```
make llama              # build llama.cpp shared libraries (CUDA when usable, else CPU)
make build              # go build ./...
make test               # unit tests, no model file needed
make test-integration   # needs TEST_MODEL=/path/to/model.gguf
make bench              # needs TEST_MODEL
make check              # fmt check, vet, lint, tests
```

`GGML_BACKEND=cpu make llama` forces a CPU build. `CUDA_ARCH=86` pins the
architecture instead of detecting the GPU.

## Invariants

- **Backend lifecycle.** `llama_backend_init` and `llama_backend_free` must
  bracket every model. `Init` is idempotent and `Shutdown` is a no-op while a
  model is open. Never call the C functions directly.
- **One cgo flag block.** Flags live in `cgo.go` only. llama.cpp writes shared
  libraries to `llama.cpp/build/bin` (older releases used `build/src` and
  `build/ggml/src`), and the rpath is baked in so no `LD_LIBRARY_PATH` is
  needed.
- **Locking.** A `Context` serialises generation behind its own mutex, because a
  KV cache serves one decode at a time. Parallelism comes from several contexts
  sharing one `Model`. Never call into a context without holding `c.mu`.
- **KV reuse.** `generate` aligns the cache with the prompt via `syncKV`, keeping
  the longest common prefix, so a chat turn only evaluates the new tail.
- **Errors.** Every failure wraps a sentinel from `errors.go`, so callers can use
  `errors.Is`. Cancellation surfaces as an error satisfying
  `errors.Is(err, context.Canceled)` while still returning the partial `Result`.
- **Logging.** llama.cpp calls the log bridge from C threads. The bridge
  assembles continuation fragments (`GGML_LOG_LEVEL_CONT`) and maps levels in the
  documented order: 1 debug, 2 info, 3 warn, 4 error.
- **Streaming.** `Stream` returns `iter.Seq2[string, error]` rather than a
  channel: breaking out of the loop stops generation with no goroutine to leak.

## Conventions

- gofumpt formatting, tests table-driven where it helps.
- Comments explain why, not what. No em dashes in code, comments or docs.
- Errors are wrapped, never returned bare from inside the package.

## Integration tests

`make test-integration` needs a real GGUF file:

```
export GO_LLAMA_TEST_MODEL=/path/to/instruct-model.gguf
export GO_LLAMA_TEST_EMBED_MODEL=/path/to/embedding-model.gguf   # optional
```

`scripts/fetch-test-models.sh` downloads small models into `models/` for this.
Integration tests skip themselves when the variables are unset.

## Upstream gotchas

- CUDA builds fail with "namespace cuda has no member make_strided_iterator" on
  some CCCL releases: `ggml/src/ggml-cuda/argsort.cu` gates the symbol on CCCL
  >= 3.1 without including `<cuda/iterator>`. `scripts/cuda-toolkit.sh` probes
  each installed toolkit by compiling that construct, so the build picks a
  toolkit that works instead of the newest one.
- A stale `llama.cpp/build/CMakeCache.txt` records the CUDA compiler that
  configured it. `scripts/build-llama.sh` clears the cache when the toolchain
  changed, rather than failing with a misleading missing-header error.
