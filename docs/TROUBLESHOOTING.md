# Troubleshooting

Build failures, runtime limits and silent test skips, with what each one means
and how to get past it.

## `cuda_runtime.h: No such file or directory` during `make llama`

**Symptom.** CMake stops during compiler identification or the first CUDA
compile and points at a header that is not there. The message names a toolkit
directory that no longer exists, or one that exists without headers.

**Cause.** `llama.cpp/build/CMakeCache.txt` records `CMAKE_CUDA_COMPILER` from
the last configure. Remove or upgrade that toolkit and the cache still points at
it, so CMake compiles a probe with a compiler whose include paths are gone. A
half-installed toolkit, `nvcc` present but `targets/x86_64-linux/include`
missing, produces the same message.

**Fix.** `scripts/build-llama.sh` compares the cache with the requested
toolchain and clears `llama.cpp/build` when they differ, so `make llama` repairs
it on its own. By hand:

```sh
rm -rf llama.cpp/build
make llama
```

To re-probe the toolkits instead of using the cached answer for this checkout:

```sh
FORCE_TOOLKIT_PROBE=1 make llama
```

The probe result is cached in `.cache/toolkit-<llama.cpp sha>`, so deleting
`.cache/` is enough if you only want a new probe.

## `namespace cuda has no member make_strided_iterator` on CUDA 13.4 and 13.5

**Symptom.** `nvcc` fails on `ggml/src/ggml-cuda/argsort.cu` with a missing
member error. Nothing else in the build fails.

**Cause.** `argsort.cu` includes `<cub/cub.cuh>` only, then gates
`cuda::make_strided_iterator` on `CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION
>= 1`. CCCL 3.4 and 3.5 ship that symbol in `<cuda/iterator>`, which the file
never includes, so the gate turns the call on and the include is missing. CCCL
3.3 and older take the fallback path and compile.

**Fix.** Build with a toolkit whose CCCL works. `scripts/cuda-toolkit.sh`
compiles exactly that construct against each candidate toolkit and returns the
first one that succeeds, which is how `make llama` ends up on CUDA 13.3 here:

```sh
CUDA_HOME=/usr/local/cuda-13.3 make llama
```

Do not patch the file: `llama.cpp` is a submodule and the patch disappears on the
next update.

## `gcc versions later than 15 are not supported`

**Symptom.** `nvcc` rejects the host compiler before compiling any kernel.

**Cause.** The toolkit's host compiler check refuses a gcc release newer than
the one it knows about. CUDA 13.2 refuses gcc 16, which can easily be the
system default on a rolling distribution.

**Fix.** `scripts/cuda-toolkit.sh` probes host compilers in order (system
default, the three newest `/usr/bin/g++-*`, then `clang++`) and reports the one
that works. `scripts/build-llama.sh` passes it as `CMAKE_CUDA_HOST_COMPILER`
together with matching `CMAKE_C_COMPILER` and `CMAKE_CXX_COMPILER`, because
CMake identifies the CUDA compiler before any CUDA flag applies. Building by
hand:

```sh
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-15 -DCMAKE_CXX_COMPILER=/usr/bin/g++-15
```

## `cannot find -lllama` or `libllama.so: cannot open shared object file`

**Symptom.** `go build ./...` fails at the link step looking for `-lllama`,
`-lggml`, `-lggml-base` or `-lggml-cpu`, or a built binary fails at startup
looking for one of the `.so` files.

**Cause.** The link flags in `pkg/llama/cgo.go` point at
`llama.cpp/build/bin`, which is where a `BUILD_SHARED_LIBS=ON` build puts them.
Older llama.cpp releases wrote `build/src` and `build/ggml/src`, and a
hand-configured build can land anywhere. A partial build leaves some of the four
libraries missing.

**Fix.** Check what the build produced:

```sh
ls llama.cpp/build/bin/libllama.so llama.cpp/build/bin/libggml*.so
```

`make llama` asserts all four exist before it returns. For a llama.cpp installed
elsewhere, point the linker at it and keep the rpath so no `LD_LIBRARY_PATH` is
needed:

```sh
CGO_LDFLAGS="-L/opt/llama/lib -Wl,-rpath,/opt/llama/lib -lllama -lggml -lggml-base -lggml-cpu -lstdc++ -lm" \
CGO_CFLAGS="-I/opt/llama/include" \
go build ./...
```

## `n_ctx_seq (8192) < n_ctx_train (128000) -- the full capacity of the model will not be utilized`

**Symptom.** A log line when a context is created, seen at debug level:

```
level=DEBUG msg="llama_context: n_ctx_seq (8192) < n_ctx_train (128000) -- the full capacity of the model will not be utilized" source=llama.cpp
```

**Cause.** Two defaults collide. llama.cpp defaults `n_ctx` to 512 tokens, which
silently truncates every conversation; the model's trained window can be 128k,
which costs gigabytes of KV cache. `llama.DefaultContextOptions` therefore caps
the default at `MaxDefaultContextSize` (8192) and llama.cpp notes that the
model's window is not fully used. The log bridge reports this notice at debug
rather than warn, since the cap is a deliberate choice.

**Fix.** Leave it alone for chat, or ask for the window you need:

```go
c, err := model.NewContext(llama.ContextOptions{ContextSize: 32768})
fmt.Println(c.ContextSize()) // what llama.cpp actually allocated
```

Raise the window until the KV cache stops fitting, then back off, or lower
`GPULayers` so the cache and the weights are not competing for the same VRAM.
Generation that runs out reports `StopReasonContextFull` or wraps
`llama.ErrContextFull`.

## GPU offload runs out of VRAM

**Symptom.** One of:

```
llama: load model: /path/model.gguf: llama.cpp could not load the file (wrong format, corrupt, or unsupported architecture)
llama: create context: llama.cpp could not allocate a 32768 token context for /path/model.gguf (out of memory?); lower ContextOptions.ContextSize
llama: decode: llama_decode could not allocate a compute buffer
```

**Cause.** `DefaultModelParams` sets `GPULayers: -1`, which offloads every layer.
Weights plus the KV cache plus the compute buffer all have to fit in VRAM once
anything is offloaded. Large contexts multiply KV cache; a 4 GB card runs out
well before a 24 GB one does.

**Fix.** Offload fewer layers, or none:

```go
model, err := llama.LoadModel(path, llama.ModelParams{
	GPULayers: 20, // 0 keeps the model on the CPU, -1 offloads everything
	UseMmap:   true,
})
```

`ModelParams` replaces the defaults wholesale, so set `UseMmap` explicitly.
Lowering `ContextSize` frees KV cache, and `nvidia-smi` shows what is actually
resident. Hybrid models with mostly convolution layers keep a small KV cache, so
context size may not be the thing that does not fit.

## Integration tests skip instead of running

**Symptom.** `make test-integration` returns in seconds, or
`go test -tags=integration ./...` reports `--- SKIP` for every end-to-end test.

**Cause.** The integration tests skip unless `GO_LLAMA_TEST_MODEL` points at a
GGUF file; embedding tests additionally want `GO_LLAMA_TEST_EMBED_MODEL`. The
variables are unset in a fresh shell, and `make test-integration` only sets them
from `TEST_MODEL`.

**Fix.**

```sh
make test-model                    # downloads small GGUFs into models/
TEST_MODEL=models/<chat>.gguf make test-integration
```

or set the variables yourself, which is what the tests read:

```sh
GO_LLAMA_TEST_MODEL=/path/chat.gguf \
GO_LLAMA_TEST_EMBED_MODEL=/path/embed.gguf \
go test -race -tags=integration -timeout 20m ./...
```

`make test-integration` refuses to start without `TEST_MODEL` and prints the
command to fix it. Both variables need absolute paths.

## Generation stops after one token against an instruct model

**Symptom.** `Generate` returns `StopReasonEOS` after a single token, and the
text is empty. The same prompt works with a completion model.

**Cause.** Instruct models are trained to receive their chat template. A bare
prompt that does not look like a user turn gets an immediate end-of-generation
token.

**Fix.** Render the template, or use `Chat`:

```go
prompt, err := model.ApplyChatTemplate([]llama.ChatMessage{
	llama.UserMessage("Name three uses for a paperclip."),
}, true)
res, err := c.Generate(ctx, prompt, opts)
```

If the template itself is the problem, `llama.ErrTemplate` says so and
`ApplyChatTemplateWith` takes an explicit one.

## `error obtaining VCS status` with a symlinked submodule

**Symptom.** `go build`, `go test` and even `git status` fail with
`error obtaining VCS status: exit status 128` plus
`error: expected submodule path 'llama.cpp' not to be a symbolic link`.

**Cause.** The build stamps the binary with VCS metadata, and git refuses to read
a submodule that is a symlink. Setups that share one llama.cpp checkout across
worktrees do this deliberately.

**Fix.** Turn off stamping for the command, and stage explicit paths rather than
the whole tree:

```sh
GOFLAGS=-buildvcs=false go build ./...
GOFLAGS=-buildvcs=false go test -race ./...
git add README.md docs/
```

`make build` and `make test` call `go` without that variable, so export it in the
shell instead: `export GOFLAGS=-buildvcs=false`.
