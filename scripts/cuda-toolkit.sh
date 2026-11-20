#!/usr/bin/env bash
# Choose a CUDA toolkit this llama.cpp checkout can actually build with, and
# report the host compiler nvcc needs to go with it.
#
# Output (stdout, one shell assignment per line, only on success):
#   toolkit=/usr/local/cuda-13.3
#   ccbin=/usr/bin/g++-15     (empty when the system default is fine)
#
# Two things disqualify a toolkit, and both have bitten this repo:
#
#  1. It is incomplete. Helix has carried half-installed toolkits (13.1 ships
#     nvcc but not cuda_runtime.h), and CMake then fails during compiler
#     identification with a misleading "cuda_runtime.h: No such file".
#  2. Its CCCL release does not compile llama.cpp's CUDA kernels. argsort.cu
#     gates cuda::make_strided_iterator on CCCL >= 3.1 but never includes
#     <cuda/iterator>, so CCCL 3.4 and 3.5 fail with "namespace cuda has no
#     member make_strided_iterator", while 13.2 is refused by gcc 16 ("gcc
#     versions later than 15 are not supported").
#
# Guessing version numbers rots, so probe: compile the same construct with each
# candidate toolkit, trying host compilers until one works.
#
# Exit 1 (with a warning on stderr) when nothing works; callers then build CPU.
set -uo pipefail

read -r -d '' PROBE_SRC <<'EOF'
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 1)
void probe() { auto it = cuda::make_strided_iterator(cuda::make_counting_iterator(0), 4); (void) it; }
#else
void probe() {}
#endif
EOF

candidates() {
    [[ -n "${CUDA_HOME:-}" ]] && echo "$CUDA_HOME"
    [[ -n "${CUDA_PATH:-}" ]] && echo "$CUDA_PATH"
    echo /usr/local/cuda
    compgen -G "/usr/local/cuda-*" | sort -Vr
}

host_compilers() {
    echo ""                                        # system default
    compgen -G "/usr/bin/g++-*" | sort -Vr | head -3
    echo "clang++"
}

probe() {
    local home="$1" ccbin="$2" tmp
    tmp="$(mktemp -d)"
    printf '%s\n' "$PROBE_SRC" > "$tmp/probe.cu"
    local -a flags=(-std=c++17 -c)
    [[ -n "$ccbin" ]] && flags+=(-ccbin "$ccbin")
    if "$home/bin/nvcc" "${flags[@]}" -arch=native "$tmp/probe.cu" -o "$tmp/probe.o" >/dev/null 2>&1; then
        rm -rf "$tmp"; return 0
    fi
    if "$home/bin/nvcc" "${flags[@]}" -arch=sm_75 "$tmp/probe.cu" -o "$tmp/probe.o" >/dev/null 2>&1; then
        rm -rf "$tmp"; return 0
    fi
    rm -rf "$tmp"; return 1
}

while read -r dir; do
    [[ -n "$dir" && -d "$dir" ]] || continue
    dir="$(readlink -f "$dir")"
    [[ -x "$dir/bin/nvcc" ]] || continue
    [[ -f "$dir/targets/x86_64-linux/include/cuda_runtime.h" ]] || continue
    while read -r ccbin; do
        if probe "$dir" "$ccbin"; then
            echo "toolkit=$dir"
            echo "ccbin=$(command -v "$ccbin" 2>/dev/null || true)"
            exit 0
        fi
    done < <(host_compilers)
done < <(candidates | awk '!seen[$0]++')

echo "warning: no usable CUDA toolkit found (incomplete, or CCCL incompatible with this llama.cpp); build will use the CPU backend" >&2
exit 1
