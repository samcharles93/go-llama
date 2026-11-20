SHELL := /usr/bin/env bash
.DEFAULT_GOAL := help

LLAMA_DIR   := llama.cpp
BUILD_DIR   := $(LLAMA_DIR)/build
LIB_DIR     := $(BUILD_DIR)/bin
JOBS        ?= $(shell nproc)
GGML_BACKEND ?= auto

# A model is only needed for the integration and benchmark targets.
TEST_MODEL ?= $(if $(wildcard models/*.gguf),$(firstword $(wildcard models/*.gguf)),)
EMBED_MODEL ?= $(if $(wildcard models/*embed*.gguf),$(firstword $(wildcard models/*embed*.gguf)),)

export GGML_BACKEND JOBS

.PHONY: help
help: ## Show this help
	@awk 'BEGIN {FS = ":.*## "; printf "\nUsage: make \033[36m<target>\033[0m\n\n"} \
		/^[a-zA-Z_-]+:.*## / {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo

.PHONY: all
all: build test ## Build the package and run the fast test suite

.PHONY: update-submodules
update-submodules: ## Fetch the llama.cpp submodule
	git submodule update --init --recursive

.PHONY: llama
llama: update-submodules ## Build llama.cpp shared libraries (GGML_BACKEND=cuda|cpu|auto)
	./scripts/build-llama.sh

.PHONY: build
build: llama ## Compile the Go package and examples
	go build ./...

.PHONY: test
test: llama ## Fast unit tests (no model required)
	go test -race ./...

.PHONY: test-integration
test-integration: llama ## End-to-end tests; set TEST_MODEL=/path/to/model.gguf
	@test -n "$(TEST_MODEL)" || { echo "set TEST_MODEL=/path/to/model.gguf (or run 'make test-model')"; exit 1; }
	GO_LLAMA_TEST_MODEL=$(abspath $(TEST_MODEL)) \
	GO_LLAMA_TEST_EMBED_MODEL=$(abspath $(EMBED_MODEL)) \
	go test -race -tags=integration -v -timeout 20m ./...

.PHONY: test-model
test-model: ## Download small GGUF models used by the integration tests
	./scripts/fetch-test-models.sh

.PHONY: bench
bench: llama ## Run generation benchmarks; set TEST_MODEL=/path/to/model.gguf
	@test -n "$(TEST_MODEL)" || { echo "set TEST_MODEL=/path/to/model.gguf"; exit 1; }
	GO_LLAMA_TEST_MODEL=$(abspath $(TEST_MODEL)) go test -run '^$$' -bench . -benchmem ./pkg/llama/

.PHONY: fmt
fmt: ## Format all Go code
	gofumpt -l -w .

.PHONY: fmt-check
fmt-check: ## Fail if any Go file is unformatted
	@out="$$(gofumpt -l .)"; if [ -n "$$out" ]; then echo "unformatted:"; echo "$$out"; exit 1; fi

.PHONY: vet
vet: llama ## Run go vet
	go vet ./...

.PHONY: lint
lint: ## Run static analysis (golangci-lint if installed, else staticcheck)
	@if command -v golangci-lint >/dev/null 2>&1; then golangci-lint run ./...; \
	elif command -v staticcheck >/dev/null 2>&1; then staticcheck ./...; \
	else echo "install golangci-lint or staticcheck"; fi

.PHONY: check
check: fmt-check vet lint test ## Everything CI runs

.PHONY: clean
clean: ## Remove build output
	rm -rf $(BUILD_DIR) bin
	go clean -cache -testcache 2>/dev/null || true
