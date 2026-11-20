#!/usr/bin/env bash
# Download the small GGUF models the integration tests and benchmarks use.
#
# Files land in models/, which .gitignore excludes. A file that is already on
# disk with a valid GGUF header is left alone, so re-running the script is cheap
# and an interrupted download resumes instead of starting over.
#
# Override the destination with MODELS_DIR=/some/dir.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
models_dir="${MODELS_DIR:-$repo_root/models}"

log() { printf '\033[1;34m[models]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[models]\033[0m %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl is required"

# name|url. The name is the file inside models/; the URL is the resolve link of
# the Hugging Face repository that publishes the file.
models=(
    "stories260K.gguf|https://huggingface.co/ggml-org/models-moved/resolve/main/tinyllamas/stories260K.gguf"
    "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf|https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
    "nomic-embed-text-v1.5-Q8_0.gguf|https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf"
)

# A GGUF file starts with the ASCII magic "GGUF". Checking it catches an HTML
# error page or a truncated file saved under the right name.
has_gguf_magic() {
    [[ "$(head -c 4 "$1" 2>/dev/null || true)" == "GGUF" ]]
}

fetch() {
    local name="$1" url="$2"
    local dest="$models_dir/$name" part="$models_dir/$name.part"

    if [[ -f "$dest" ]]; then
        if has_gguf_magic "$dest"; then
            log "have $name ($(du -h "$dest" | cut -f1))"
            return 0
        fi
        log "discarding $name: no GGUF header"
        rm -f "$dest"
    fi

    log "downloading $name"
    # -C - resumes a partial download, -# shows a progress bar, and the retries
    # cover the redirect chain to the CDN that Hugging Face serves from.
    if ! curl -fL --retry 3 --retry-delay 2 -C - -# -o "$part" "$url"; then
        rm -f "$part"
        die "download failed: $url"
    fi
    if ! has_gguf_magic "$part"; then
        rm -f "$part"
        die "$name does not start with the GGUF magic bytes; the URL may have moved or need auth"
    fi
    mv "$part" "$dest"
    log "saved $name ($(du -h "$dest" | cut -f1))"
}

mkdir -p "$models_dir"
for entry in "${models[@]}"; do
    fetch "${entry%%|*}" "${entry#*|}"
done

# Print absolute paths so the exports work from any directory.
models_dir="$(cd "$models_dir" && pwd)"
printf '\nModels are ready. Export these to run the integration tests and benchmarks:\n\n'
printf 'export GO_LLAMA_TEST_MODEL=%s/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf\n' "$models_dir"
printf 'export GO_LLAMA_TEST_EMBED_MODEL=%s/nomic-embed-text-v1.5-Q8_0.gguf\n' "$models_dir"
printf '\nstories260K.gguf in the same directory is a 1 MB model for quick smoke tests.\n'
