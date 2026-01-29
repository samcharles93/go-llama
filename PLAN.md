# Project Plan: Go-Llama (Native Bindings)

This document outlines the roadmap for developing `go-llama`, a Go wrapper for the `llama.cpp` library. The goal is to provide a robust, idiomatic Go API for local LLM inference, enabling Go developers to easily integrate powerful AI features into their applications.

## 1. Core Implementation Status

**Current State:**
- [x] Basic project structure (`pkg/llama`)
- [x] CGO compilation and linking against `llama.cpp`
- [x] Backend initialization (`Initialize`, `Free`)
- [x] Model loading (`LoadModel`, `Free`)
- [x] Context creation (`NewContextWithParams`, `Free`)
- [x] Tokenization (`Tokenize`, `TokenToPiece`)
- [x] **Decoding & Batching:** `Decode`, `Batch` wrapper implemented.
- [x] **Sampling:** `Sampler` chain implemented.
- [x] **Embeddings:** `Embeddings()` method supported.
- [x] **Chat Templates:** `ApplyTemplate()` for prompt formatting.
- [x] **Logging:** `slog` integration with `llama.cpp` log redirection.

**Immediate Next Steps (Phase 1 - Complete):**
- [x] Decoding wrapper.
- [x] Batch Management.
- [x] Sampling API.
- [x] End-to-End Inference (`examples/simple`).

## 2. Feature Expansion Roadmap

### Phase 2: High-Level API & Developer Experience
*Goal: Make it "one-liner" easy to run an LLM.*

- [x] **Simple Generation:** `ctx.Generate(prompt, opts)`
- [x] **Streaming:** `ctx.Stream(prompt, opts)`
- [x] **Configuration:** `CompletionOptions`, `ContextParams`.
- [ ] **Structured Logging:**
    - [x] Integrate `log/slog`.
    - [x] Output to `$XDG_CONFIG_DIR/.go-llama/` + stdout/stderr.
    - [x] Redirect native `llama.cpp` logs into the structured logger.

### Phase 3: Advanced Capabilities
- **Grammar/Structured Output:** Bind `llama_grammar` to force JSON output (crucial for tool use).
- **Multimodal (Vision):** Support LLaVA/CLIP models (`llama_model_quantize` bindings).
- **LoRA Adapters:** Runtime loading of LoRA adapters (`llama_model_apply_lora_from_file`).
- **Function Calling:** Higher-level abstraction for tool definition and parsing.

### Phase 4: Performance & Optimization
- **Batched Inference:** Process multiple distinct sequences in parallel (server scenario).
- **State Management:** Save/Load context state (KV cache) to disk.
- **GPU Offloading Control:** Expose `n_gpu_layers` configuration.
- **Memory Tuning:**
    - **Go 1.26+ Features:** Experiment with new GC tuning knobs (e.g., `GOGC` dynamic adjustment, arena allocation) to minimize latency spikes during token generation.
    - **Zero-Copy:** Ensure tensor data moving between C and Go uses unsafe pointers to avoid copying large embedding vectors.

## 3. Long-Term: CGO Removal & Pure Go

*Goal: Reduce dependency on the C toolchain and improve portability.*

1.  **Static Linking (Intermediate):**
    -   Configure `llama.cpp` CMake to build static libraries (`.a`) instead of shared (`.so`).
    -   Update CGO LDFLAGS to link statically (`-static`, `-l:libllama.a`).
    -   *Benefit:* Single-binary distribution (Linux/macOS).

2.  **Pure Go GGUF Loader:**
    -   Implement a native Go GGUF parser (read headers, metadata, and tensor mappings).
    -   *Status:* Feasible, many existing Go libraries do this.

2.  **Pure Go Inference (Tinygrad/GGML-port):**
    -   Port the `ggml` tensor graph evaluation engine to Go.
    -   *Challenge:* Performance. Go's auto-vectorization is not as mature as C++ SIMD intrinsics (AVX2/AVX512/AMX).
    -   *Strategy:* Use Go assembly (`asm`) for hot paths (MatMul, Softmax) or target WebAssembly (WASI) as a portable intermediate layer.

3.  **Intermediate Step (Wasm):**
    -   Compile `llama.cpp` to Wasm/WASI.
    -   Run inside a Go Wasm runtime (like `wazero`).
    -   *Benefit:* Safe, sandboxed, no CGO.
    -   *Drawback:* Performance penalty (10-50% slower than native).

## 4. Maintenance & Upstream Sync Strategy

- **Submodule Management:**
    - Track `llama.cpp` as a git submodule.
    - Update policy: Weekly or bi-weekly.
- **Versioning:**
    - Tag releases matching `llama.cpp` stable points.

## 5. Required Codebase Elements

- `pkg/llama/logging.go`: Setup `slog` and C callbacks.
- `pkg/llama/options.go`: Configuration structs.
- `examples/server`: REST API server.

---
*Last Updated: Jan 23, 2026*