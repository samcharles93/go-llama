package llama

import (
	"fmt"
	"runtime"
)

// PoolingType selects how token embeddings are reduced to one vector per
// sequence. Only consulted for contexts created with Embeddings enabled.
type PoolingType int32

// Pooling modes, mirroring enum llama_pooling_type.
const (
	PoolingUnspecified PoolingType = -1
	PoolingNone        PoolingType = 0
	PoolingMean        PoolingType = 1
	PoolingCLS         PoolingType = 2
	PoolingLast        PoolingType = 3
	PoolingRank        PoolingType = 4
)

func (p PoolingType) String() string {
	switch p {
	case PoolingNone:
		return "none"
	case PoolingMean:
		return "mean"
	case PoolingCLS:
		return "cls"
	case PoolingLast:
		return "last"
	case PoolingRank:
		return "rank"
	default:
		return "unspecified"
	}
}

// ModelParams controls how a model is loaded. The zero value means "let
// llama.cpp decide", which loads every layer onto the GPU when one is
// available. Use DefaultModelParams to get that explicitly.
type ModelParams struct {
	// GPULayers is the number of layers to offload. Negative offloads all of
	// them, 0 keeps the model on the CPU.
	GPULayers int32
	// MainGPU picks the device used when the model is not split.
	MainGPU int32
	// UseMmap memory-maps the weights instead of reading them into anonymous
	// memory. Leave it on unless the model lives on a slow network share.
	UseMmap bool
	// UseMlock locks the weights in RAM so they are never paged out. Needs
	// RLIMIT_MEMLOCK, so it is off by default.
	UseMlock bool
	// VocabOnly loads just the tokenizer, which is enough for tokenize tests
	// and costs no weight memory.
	VocabOnly bool
}

// DefaultModelParams mirrors llama_model_default_params.
func DefaultModelParams() ModelParams {
	return ModelParams{GPULayers: -1, UseMmap: true}
}

func (p ModelParams) validate() error {
	if p.GPULayers < -1 {
		return fmt.Errorf("%w: GPULayers %d is below -1", ErrModelLoad, p.GPULayers)
	}
	return nil
}

// ContextOptions controls context creation. Sizes are in tokens; zero means
// "use the model's trained value".
type ContextOptions struct {
	// ContextSize is the KV cache window, n_ctx. Generation stops with
	// ErrContextFull when the conversation outgrows it.
	ContextSize uint32
	// BatchSize is the largest batch handed to llama_decode. A prompt longer
	// than this is decoded in several chunks.
	BatchSize uint32
	// Sequences is the number of concurrent sequences the KV cache holds.
	Sequences uint32
	// Threads defaults to the number of CPUs nvcc and ggml were tuned for; 0
	// leaves it to llama.cpp.
	Threads int32
	// ThreadsBatch is the thread count for batch and prompt processing.
	ThreadsBatch int32
	// Embeddings turns on the embedding output path. Chat and embeddings
	// cannot share a context.
	Embeddings bool
	// Pooling selects the embedding reduction mode.
	Pooling PoolingType
	// FlashAttention enables flash attention where the backend supports it.
	FlashAttention bool
	// DisablePerf turns off llama.cpp's per-context timing counters, which
	// costs a clock read per decode. Leave it off so Metrics is populated:
	// llama_context_default_params sets no_perf true, so this package sets it
	// back to false unless asked otherwise.
	DisablePerf bool
}

// DefaultContextOptions returns a chat-ready context: the model's trained
// window capped at MaxDefaultContextSize, threads matched to the machine, flash
// attention on, and timings enabled.
func DefaultContextOptions() ContextOptions {
	return ContextOptions{
		Sequences:      1,
		Pooling:        PoolingUnspecified,
		FlashAttention: true,
		Threads:        int32(runtime.NumCPU()),
		ThreadsBatch:   int32(runtime.NumCPU()),
	}
}

// MaxDefaultContextSize caps the KV cache a zero-value ContextOptions asks for.
// llama_context_default_params defaults n_ctx to 512, which silently truncates
// every conversation, while the model's trained window can be 128k and cost
// gigabytes of KV cache. 8192 fits comfortably on a 4 GB card and is far more
// than a chat turn needs; set ContextSize explicitly to go beyond it.
const MaxDefaultContextSize uint32 = 8192

// resolveContextSize picks the KV cache size, honouring an explicit request and
// otherwise taking the smaller of the model's trained window and
// MaxDefaultContextSize.
func (o ContextOptions) resolveContextSize(trained int) uint32 {
	if o.ContextSize > 0 {
		return o.ContextSize
	}
	if trained > 0 && uint32(trained) < MaxDefaultContextSize {
		return uint32(trained)
	}
	return MaxDefaultContextSize
}

// CompletionOptions is the sampling and stopping configuration for one
// generation. The zero value is greedy decoding that runs until the model emits
// end of generation; use DefaultCompletionOptions for something chat-like.
type CompletionOptions struct {
	// MaxTokens caps generated tokens. Non-positive means "until the model
	// stops or the context fills".
	MaxTokens int
	// Temperature scales the distribution. 0 selects greedy decoding and
	// disables TopK, TopP, MinP, TypicalP and the random seed.
	Temperature float32
	// TopK keeps the k most likely tokens. 0 disables it.
	TopK int32
	// TopP keeps the smallest set of tokens whose probability sums to p. 0
	// disables it.
	TopP float32
	// MinP keeps tokens at least p times as likely as the best one. 0
	// disables it.
	MinP float32
	// TypicalP applies locally typical sampling. 0 disables it.
	TypicalP float32
	// RepeatPenalty penalises tokens already seen in the last RepeatLastN
	// positions. 1 disables it, values above 1 discourage repetition.
	RepeatPenalty float32
	// RepeatLastN is the repetition window. 0 follows RepeatPenalty's default
	// of 64, -1 uses the whole context.
	RepeatLastN int32
	// PresencePenalty and FrequencyPenalty are additive penalties over the
	// repetition window, as in the OpenAI API.
	PresencePenalty  float32
	FrequencyPenalty float32
	// Seed seeds the sampler chain. Sampling is deterministic for a given
	// seed, prompt and options.
	Seed uint32
	// Stop halts generation when any of these strings appears in the output.
	// The stop text is not returned.
	Stop []string
	// StopTokens halts generation on these token ids in addition to the
	// model's end-of-generation tokens.
	StopTokens []Token
	// AddBOS prepends the model's beginning-of-sequence token when tokenizing
	// the prompt. Defaults to true.
	AddBOS bool
	// Grammar constrains output to a GBNF grammar. Empty means unconstrained.
	// The grammar is parsed when generation starts and is not cached.
	Grammar string
	// IgnoreEOS keeps generating past end-of-generation tokens. Useful for
	// completion models; rarely wanted for chat.
	IgnoreEOS bool
}

// DefaultCompletionOptions returns sampling settings suitable for chat: a
// little creativity, repetition suppressed, and no runaway output.
func DefaultCompletionOptions() CompletionOptions {
	return CompletionOptions{
		MaxTokens:     512,
		Temperature:   0.7,
		TopK:          40,
		TopP:          0.95,
		MinP:          0.05,
		RepeatPenalty: 1.1,
		RepeatLastN:   64,
		Seed:          0xFFFFFFFF,
		AddBOS:        true,
	}
}

// GreedyCompletionOptions returns deterministic decoding, which is what the
// tests use.
func GreedyCompletionOptions(maxTokens int) CompletionOptions {
	return CompletionOptions{MaxTokens: maxTokens, AddBOS: true}
}

// greedy reports whether the options select deterministic decoding.
func (o CompletionOptions) greedy() bool { return o.Temperature <= 0 }

func (o CompletionOptions) validate() error {
	switch {
	case o.TopK < 0:
		return fmt.Errorf("llama: TopK %d is negative", o.TopK)
	case o.TopP < 0 || o.TopP > 1:
		return fmt.Errorf("llama: TopP %v is outside [0,1]", o.TopP)
	case o.MinP < 0 || o.MinP > 1:
		return fmt.Errorf("llama: MinP %v is outside [0,1]", o.MinP)
	case o.TypicalP < 0 || o.TypicalP > 1:
		return fmt.Errorf("llama: TypicalP %v is outside [0,1]", o.TypicalP)
	case o.RepeatPenalty < 0:
		return fmt.Errorf("llama: RepeatPenalty %v is negative", o.RepeatPenalty)
	case o.RepeatLastN < -1:
		return fmt.Errorf("llama: RepeatLastN %d is below -1", o.RepeatLastN)
	}
	for _, s := range o.Stop {
		if s == "" {
			return fmt.Errorf("llama: empty stop sequence")
		}
	}
	return nil
}
