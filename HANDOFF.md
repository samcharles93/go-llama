# Handoff: llama.cpp bump to b12603 for LFM3 support

Uncommitted work in progress. Delete this file once the work lands.

## Goal

`examples/chat` failed on `LFM3-1.7B-Q8_0.gguf` with "unknown model
architecture: 'lfm3'". The pinned llama.cpp (b7871, 2026-01-29) predates LFM3.
The fix is bumping the submodule to b12603 (2026-11-21), which has
`LLM_ARCH_LFM3`, and porting the Go package to the changed C API.

## State

- **Submodule:** `llama.cpp` is checked out at `b12603` and the new pointer is
  **staged** (`git add llama.cpp`). It has to stay staged or committed:
  `make llama` runs `git submodule update`, which otherwise resets the checkout
  to the old pointer. That reset silently cost one rebuild in this session.
- **Build:** `llama.cpp/build/bin` holds b12603 libraries (`libllama.so.1.0.0`).
  `go build ./...` passes, with deprecation warnings only (see below).
- **LFM3 works:** `examples/generate` and `examples/chat` load it, generate at
  about 90 tok/s on the GPU, and keep context across chat turns.
- **Unit tests:** `go test -race ./...` passes.
- **Lint:** `golangci-lint` reported one gofumpt issue in `sampler.go`.
  `gofumpt -w` has been run since, but the linter has not been re-run.
- **Integration tests: crash.** See the open bug below.

## C API changes ported (uncommitted)

| Change in b12603 | Port |
| --- | --- |
| `use_mmap`/`use_mlock` replaced by `load_mode` enum | `ModelParams.loadMode()` in `llama.go` maps the two bools; public API unchanged |
| `llama_set_adapter_lora`, `llama_rm_adapter_lora` and `llama_clear_adapter_lora` removed for `llama_set_adapters_lora(ctx, adapters, n, scales)` (whole list) | `Context.loras []loraSlot` tracks the list; `Apply`, `Remove` and `RemoveAllLoras` push the full list through `setLoras` in `lora.go` |
| `llama_sampler_init_penalties` gained a leading `n_vocab` | passes `vocabSize(vocab)`; `AddPenalties` (no model) passes 0, which only sizes llama.cpp's GPU-side sampling path |
| `llama_sampler_init_dry` dropped its `n_ctx_train` argument | removed |
| Penalties and DRY now clamp a negative window to 0, which **disables** the stage | `resolveWindow` in `sampler.go` resolves the documented `-1` ("whole context") to `n_ctx_train`, or `MaxDefaultContextSize` without a model. Without it, `RepeatLastN: -1` silently turns penalties off |

Fixes to existing bugs found along the way:

- `addDry` passed `opts.repeatLastN()` as DRY's window, so `DryPenaltyLastN`
  was ignored. It now uses the new `opts.dryPenaltyLastN()` (0 means 64, -1 the
  whole context).
- `RemoveAllLoras` and `Context.Close` never released the adapters' `applied`
  counts, so `LoraAdapter.Close` could refuse forever. `releaseLoras` fixes
  this. It runs without `c.mu`, because `Apply` takes `a.mu` before `c.mu`.
- `Applied()` now counts contexts, not `Apply` calls.

Tests: `pkg/llama/upstream_api_test.go` (load mode mapping, `resolveWindow`, the
whole-context penalty staying enabled, `dryPenaltyLastN`, `releaseLoras`).
`TestWholeContextPenaltyStaysEnabled` was confirmed to fail without
`resolveWindow` (it builds `?penalties`, llama.cpp's name for a disabled stage).
The DRY integration test comment now says the same about `?dry`.

## Open bug: embeddings crash under the integration suite

```
llama.cpp/ggml/src/ggml.c:3370: GGML_ASSERT(ggml_can_mul_mat(a, b)) failed
```

- It hits `TestEmbedBatchAgreesWithSingleEmbed`
  (`embeddings_integration_test.go:60`) with both embedding models (nomic BERT
  and embeddinggemma), and aborts the whole test binary. The tests before it
  pass, including `TestEmbedVectorMatchesModelSize`.
- Ruled out: `Sequences` > 1 on its own, and `ModelParams{GPULayers: 0}`
  (`UseMmap` false, which maps to `LLAMA_LOAD_MODE_NONE`). A standalone repro
  with those settings embeds fine.
- Not yet tried: running that test alone
  (`-run 'TestEmbedBatchAgreesWithSingleEmbed$'`). If it passes alone, the crash
  depends on state left by an earlier test (shared model, the backend, or the
  abort flag). That points at a context or model lifecycle interaction with the
  new llama.cpp. Also not yet tried: `Sequences: 5` with five texts (the repro
  only used 2), and a symbolised backtrace (`gdb --args` on the test binary;
  the attached backtrace printed only `??`).
- The embed tool in b12603 is no longer called `llama-embedding`
  (`ls llama.cpp/build/bin | grep -i embed`), useful for checking whether the
  crash happens upstream too.
- `pkg/llama/zz_repro_integration_test.go` is a throwaway repro
  (`TestReproEmbedSeqs`). **Delete it before committing.**

## Still to do

1. Fix the embeddings crash, then run both integration pairs with `-race`:
   - `models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf` with `models/nomic-embed-text-v1.5-Q8_0.gguf`
     (from `scripts/fetch-test-models.sh`)
   - `/work/models/gguf/liquidai/LFM2.5-1.2B-Instruct/LFM2.5-1.2B-Instruct-Q4_K_M.gguf`
     with `/work/models/gguf/google/embeddinggemma-300M/embeddinggemma-300M-Q8_0.gguf`
   - Before the bump these gave 105 passes on Qwen. The integration runs here
     reached 22 and 23 passes before the crash.
2. Consider adding an LFM3 smoke test, or noting LFM3 in the README.
3. Deprecations still in use, all working but noisy at build time:
   `llama_decode` (use `llama_process` with `LLAMA_PROCESS_TYPE_DECODE`), and
   `llama_batch_init`/`llama_batch_free` (use `llama_batch_ext_*`). Porting
   `batch.go` and `decode` to the new batch API is a separate change.
4. `make check`, then commit (submodule bump plus port as one commit, so every
   commit builds). Pushing and tagging are the user's call. The last tag is
   `v0.1.0`.
5. Unrelated, flagged earlier: `go.mod` says `module go-llama`, so tags cannot
   be fetched with `go get github.com/samcharles93/go-llama`.
