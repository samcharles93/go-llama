# go-llama

Go bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp), via cgo. The
llama.cpp source is a git submodule; the Go package links its shared libraries.

## Layout

```
pkg/llama/            the library
  cgo.go              the only place cgo compile/link flags are set
  llama.go            backend lifecycle, Model, Context
  abort.go            C abort flag so cancellation interrupts llama_decode
  threads*.go         DefaultThreads: one thread per physical performance core
  generate.go         Generate, Stream (iter.Seq2), KV-cache reuse
  batch.go            capacity-checked token batches
  sampler.go          sampler chains built from CompletionOptions, sampling
  stop.go             stop-sequence matcher with holdback
  session.go          SaveState/LoadState, Truncate, History
  chatfit.go          drop old turns to fit a token budget
  metadata.go         GGUF key/value metadata
  lora.go             LoRA adapters
  cgo.go              the only cgo flag block
  vocab.go            tokenize, detokenize, token to piece
  chat.go             chat templates
  embeddings.go       pooled embeddings
  metrics.go          llama.cpp perf counters
  logging.go          llama.cpp log bridge into slog
  options.go          ModelParams, ContextOptions, CompletionOptions
  errors.go           sentinel errors and wrapping
  doc.go              package documentation
examples/             runnable programs (chat, server, embed, generate, tokenize)
scripts/              build-llama.sh, cuda-toolkit.sh, fetch-test-models.sh
docs/                 troubleshooting reference
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
- **Cancellation reaches llama.cpp.** Each context owns a C atomic flag polled
  by ggml's abort callback, raised by `watchCancel` while a generation or embed
  runs. CPU and Metal poll it mid-graph; CUDA does not, so `syncKV` also checks
  the Go context between prompt batches. A failed or aborted decode leaves
  processed micro-batches in the cache, so callers `rollback` to the history.
- **Threads.** Zero `Threads`/`ThreadsBatch` means `DefaultThreads`, never
  llama.cpp's fixed 4. `runtime.NumCPU` oversubscribes SMT and E-cores.
- **Flash attention** is a tri-state whose zero value is auto, because
  llama.cpp's default is already auto and a bool could not turn it off.
- **Errors.** Every failure wraps a sentinel from `errors.go`, so callers can use
  `errors.Is`. Cancellation surfaces as an error satisfying
  `errors.Is(err, context.Canceled)` while still returning the partial `Result`.
- **Logging.** llama.cpp calls the log bridge from C threads. The bridge
  assembles continuation fragments (`GGML_LOG_LEVEL_CONT`) and maps levels in the
  documented order: 1 debug, 2 info, 3 warn, 4 error.
- **Streaming.** `Stream` returns `iter.Seq2[string, error]` rather than a
  channel: breaking out of the loop stops generation with no goroutine to leak.
- **Sampling is apply, select, accept.** `llama_sampler_sample` in this pinned
  llama.cpp applies the chain, selects a token and *accepts* it in one call. Using
  it and then calling `Sampler.Accept` accepts every token twice, which
  double-counts penalties and pushes a grammar sampler past the end of its
  grammar, where llama.cpp throws a C++ exception that aborts the process.
  `Sampler.Sample` therefore applies the chain itself through
  `llama_sampler_apply` and leaves accepting to the caller. The generation loop
  accepts only tokens it keeps, never the stop token.
- **Examples default to GPU offload.** `-ngl` defaults to -1 (every layer) to match
  `DefaultModelParams`; pass `-ngl 0` for a CPU run.
- **Grammars.** A GBNF grammar masks logits. It is only safe when the model's
  vocabulary can spell what the grammar allows; a grammar that no token can
  satisfy makes llama.cpp abort rather than return an error, so grammars in tests
  stay permissive.

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
