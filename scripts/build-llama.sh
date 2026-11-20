#!/usr/bin/env bash
# Configure and build llama.cpp as the shared libraries pkg/llama links against.
#
# A plain `cmake .. && make` in a reused build directory keeps whatever CUDA
# compiler CMakeCache.txt recorded, even after that toolkit is removed or loses
# its headers, and then fails during compiler identification with a misleading
# "cuda_runtime.h: No such file". This script validates the toolchain, clears a
# stale cache, and keeps the CUDA and host compilers consistent.
#
# Backend: GGML_BACKEND=cuda|cpu|auto (default auto: CUDA when scripts/
# cuda-toolkit.sh finds a working toolkit and an NVIDIA GPU is present).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_dir="$repo_root/llama.cpp"
build_dir="$llama_dir/build"
bin_dir="$build_dir/bin"
cache_dir="$repo_root/.cache"
backend="${GGML_BACKEND:-auto}"
jobs="${JOBS:-$(nproc)}"

log() { printf '\033[1;34m[build]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[build]\033[0m %s\n' "$*" >&2; exit 1; }

[[ -f "$llama_dir/CMakeLists.txt" ]] || die "llama.cpp submodule missing; run 'make update-submodules'"

gpu_arch=""
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '. ' || true)"
fi

# Probing toolkits costs about a minute, so remember the answer until the
# llama.cpp checkout changes.
toolkit=""; ccbin=""
cache_file="$cache_dir/toolkit-$(git -C "$llama_dir" rev-parse --short HEAD 2>/dev/null || echo unknown)"
if [[ -f "$cache_file" && "${FORCE_TOOLKIT_PROBE:-0}" != 1 ]]; then
    # shellcheck disable=SC1090
    source "$cache_file"
fi
if [[ -z "$toolkit" ]]; then
    if out="$(bash "$repo_root/scripts/cuda-toolkit.sh")"; then
        eval "$out"
        mkdir -p "$cache_dir"
        printf 'toolkit=%q\nccbin=%q\n' "$toolkit" "$ccbin" > "$cache_file"
    fi
fi

if [[ "$backend" == auto ]]; then
    if [[ -n "$toolkit" && -n "$gpu_arch" ]]; then backend=cuda; else backend=cpu; fi
fi

cmake_args=(
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=ON
    -DLLAMA_BUILD_TESTS=OFF
    -DLLAMA_BUILD_EXAMPLES=OFF
    -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_SERVER=ON
    -DGGML_NATIVE=ON
)

case "$backend" in
    cuda)
        [[ -n "$toolkit" ]] || die "GGML_BACKEND=cuda but no usable CUDA toolkit was found"
        [[ -n "$gpu_arch" ]] || die "GGML_BACKEND=cuda but no NVIDIA GPU was detected"
        cuda_arch="${CUDA_ARCH:-$gpu_arch}"
        # CMake's CUDA compiler identification runs before any CUDA flag applies,
        # so the host compiler has to be selected here rather than in a flag.
        if [[ -n "$ccbin" ]]; then
            case "$(basename "$ccbin")" in
                clang++*) cmake_args+=(-DCMAKE_CUDA_HOST_COMPILER="$ccbin" -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER="$ccbin") ;;
                *)        cmake_args+=(-DCMAKE_CUDA_HOST_COMPILER="$ccbin" -DCMAKE_C_COMPILER="$(dirname "$ccbin")/gcc-${ccbin##*-}" -DCMAKE_CXX_COMPILER="$ccbin") ;;
            esac
        fi
        cmake_args+=(
            -DGGML_CUDA=ON
            -DCMAKE_CUDA_COMPILER="$toolkit/bin/nvcc"
            -DCMAKE_CUDA_ARCHITECTURES="$cuda_arch"
            -DGGML_CUDA_FA=ON
            -DCUDAToolkit_ROOT="$toolkit"
        )
        log "CUDA $toolkit${ccbin:+, host compiler $ccbin}, arch $cuda_arch"
        ;;
    cpu)
        log "CPU-only build"
        ;;
    *)
        die "GGML_BACKEND must be cuda, cpu or auto (got '$backend')"
        ;;
esac

if [[ -f "$build_dir/CMakeCache.txt" ]]; then
    want_cuda=OFF; [[ "$backend" == cuda ]] && want_cuda=ON
    cached="$(sed -n -e 's/^GGML_CUDA:BOOL=//p' -e 's/^CMAKE_CUDA_COMPILER:FILEPATH=//p' -e 's/^CMAKE_CXX_COMPILER:FILEPATH=//p' -e 's/^CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=//p' "$build_dir/CMakeCache.txt" | tr '\n' ' ')"
    want="$want_cuda ${toolkit:+$toolkit/bin/nvcc} ${ccbin} ${cuda_arch:-}"
    if [[ "$cached" != "$want" ]]; then
        log "build cache does not match the requested toolchain, clearing $build_dir"
        log "  cached: $cached"
        log "  wanted: $want"
        rm -rf "$build_dir"
    fi
fi

mkdir -p "$build_dir"
log "configuring"
cmake -S "$llama_dir" -B "$build_dir" "${cmake_args[@]}"
log "compiling with $jobs jobs"
cmake --build "$build_dir" -j "$jobs"

for lib in libllama.so libggml.so libggml-base.so libggml-cpu.so; do
    [[ -e "$bin_dir/$lib" ]] || die "expected $bin_dir/$lib after the build"
done
log "ok: shared libraries in $bin_dir"
