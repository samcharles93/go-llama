package llama

import (
	"fmt"
	"math"
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
	// UBatchSize is the micro-batch: the largest slab of tokens evaluated in one
	// compute graph. It bounds the memory a decode needs, and llama.cpp caps it
	// at 512 by default.
	//
	// It also bounds what pooled embeddings can see: a sequence's pooled vector
	// is computed per micro-batch, so a text longer than UBatchSize cannot be
	// embedded in one piece. Raise it when embedding long documents and watch the
	// memory cost.
	UBatchSize uint32
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
	// disables every stochastic stage: TopK, TopP, MinP, TypicalP, TopNSigma,
	// XTC, mirostat and the random seed.
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
	// TopNSigma keeps tokens whose logit is within n standard deviations of
	// the best one. 0 disables it.
	TopNSigma float32
	// XTCProbability is the chance that XTC drops the most likely tokens and
	// picks from the tail, which breaks the repetitive loops greedy decoding
	// falls into. 0 disables it. XTC draws random numbers, so it needs Seed.
	XTCProbability float32
	// XTCThreshold is the probability at or above which a token counts as an
	// XTC candidate. Above 0.5 the stage is skipped, the same cut-off
	// llama.cpp applies; 0 makes every token above zero probability eligible.
	XTCThreshold float32
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
	// DryMultiplier turns on DRY, which penalises tokens that would extend a
	// sequence already present in the window. 0 disables it; DRY needs the
	// model vocabulary to tokenise its breakers.
	DryMultiplier float32
	// DryBase is the exponential growth rate of the DRY penalty. 0 uses
	// llama.cpp's default of 1.75.
	DryBase float32
	// DryAllowedLength is how long a repeated sequence may get before DRY
	// penalises it. 0 uses the default of 2.
	DryAllowedLength int32
	// DryPenaltyLastN is DRY's window, following RepeatLastN: 0 uses the
	// default of 64, -1 uses the whole context.
	DryPenaltyLastN int32
	// DrySequenceBreakers are the strings that restart DRY's repetition count,
	// for example "\n" so that indentation shared with an earlier line is not
	// penalised. Empty leaves llama.cpp with no breakers, so DRY never fires.
	DrySequenceBreakers []string
	// MirostatTau and MirostatEta turn on mirostat v2, which picks tokens to
	// hold the output near a target surprise level instead of sampling from
	// the truncated distribution. Tau is the target cross-entropy and eta the
	// learning rate; set both, because one without the other is almost always
	// a mistake. The truncation stages still run first.
	MirostatTau float32
	MirostatEta float32
	// Seed seeds the sampler chain. Sampling is deterministic for a given
	// seed, prompt and options; SeedRandom asks for a fresh seed each time.
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

// SeedRandom asks llama.cpp for a random seed on every generation. It is
// llama.cpp's LLAMA_DEFAULT_SEED, so a chain built with it is not reproducible.
const SeedRandom uint32 = 0xFFFFFFFF

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
		Seed:          SeedRandom,
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

// penaltiesEnabled reports whether the penalties stage has anything to do.
// RepeatPenalty of exactly 1 is the identity divisor, so it alone means off.
func (o CompletionOptions) penaltiesEnabled() bool {
	repeat := o.RepeatPenalty > 0 && o.RepeatPenalty != 1
	return repeat || o.PresencePenalty != 0 || o.FrequencyPenalty != 0
}

// repeatLastN resolves the repetition window: 0 means the documented default of
// 64 rather than an empty window, and -1 means the whole context, which
// llama.cpp resolves per generation.
func (o CompletionOptions) repeatLastN() int32 {
	if o.RepeatLastN == 0 {
		return 64
	}
	return o.RepeatLastN
}

// xtcEnabled reports whether XTC would change anything. llama.cpp turns XTC
// into a no-op when the probability is zero or the threshold is above 0.5, so
// the stage is left out rather than added dead.
func (o CompletionOptions) xtcEnabled() bool {
	return o.XTCProbability > 0 && o.XTCThreshold <= 0.5
}

// mirostatEnabled reports whether mirostat v2 replaces the ordinary picker.
func (o CompletionOptions) mirostatEnabled() bool {
	return o.MirostatTau > 0 && o.MirostatEta > 0
}

// dryEnabled reports whether DRY should be built. DryMultiplier is the switch;
// the other DRY fields only tune a stage that is already on.
func (o CompletionOptions) dryEnabled() bool { return o.DryMultiplier != 0 }

// dryBase resolves DRY's base, defaulting to llama.cpp's 1.75.
func (o CompletionOptions) dryBase() float32 {
	if o.DryBase == 0 {
		return 1.75
	}
	return o.DryBase
}

// dryAllowedLength resolves how long a repeat may get before DRY penalises it,
// defaulting to llama.cpp's 2.
func (o CompletionOptions) dryAllowedLength() int32 {
	if o.DryAllowedLength == 0 {
		return 2
	}
	return o.DryAllowedLength
}

// validate rejects option sets that would make llama.cpp sample nonsense. NaN
// and infinite values are caught here because every range comparison against
// them is false, so they would otherwise reach a sampler and poison the logits
// for the rest of the generation.
func (o CompletionOptions) validate() error {
	reals := []struct {
		name string
		v    float32
	}{
		{"Temperature", o.Temperature},
		{"TopP", o.TopP},
		{"MinP", o.MinP},
		{"TypicalP", o.TypicalP},
		{"TopNSigma", o.TopNSigma},
		{"XTCProbability", o.XTCProbability},
		{"XTCThreshold", o.XTCThreshold},
		{"RepeatPenalty", o.RepeatPenalty},
		{"PresencePenalty", o.PresencePenalty},
		{"FrequencyPenalty", o.FrequencyPenalty},
		{"DryMultiplier", o.DryMultiplier},
		{"DryBase", o.DryBase},
		{"MirostatTau", o.MirostatTau},
		{"MirostatEta", o.MirostatEta},
	}
	for _, f := range reals {
		if math.IsNaN(float64(f.v)) || math.IsInf(float64(f.v), 0) {
			return errorf(ErrDecode, "%s %v is not a finite number", f.name, f.v)
		}
	}

	switch {
	case o.TopK < 0:
		return errorf(ErrDecode, "TopK %d is negative", o.TopK)
	case o.TopP < 0 || o.TopP > 1:
		return errorf(ErrDecode, "TopP %v is outside [0,1]", o.TopP)
	case o.MinP < 0 || o.MinP > 1:
		return errorf(ErrDecode, "MinP %v is outside [0,1]", o.MinP)
	case o.TypicalP < 0 || o.TypicalP > 1:
		return errorf(ErrDecode, "TypicalP %v is outside [0,1]", o.TypicalP)
	case o.TopNSigma < 0:
		return errorf(ErrDecode, "TopNSigma %v is negative", o.TopNSigma)
	case o.XTCProbability < 0 || o.XTCProbability > 1:
		return errorf(ErrDecode, "XTCProbability %v is outside [0,1]", o.XTCProbability)
	case o.XTCThreshold < 0 || o.XTCThreshold > 1:
		return errorf(ErrDecode, "XTCThreshold %v is outside [0,1]", o.XTCThreshold)
	case o.RepeatPenalty < 0:
		return errorf(ErrDecode, "RepeatPenalty %v is negative", o.RepeatPenalty)
	case o.RepeatLastN < -1:
		return errorf(ErrDecode, "RepeatLastN %d is below -1", o.RepeatLastN)
	case o.DryMultiplier < 0:
		return errorf(ErrDecode, "DryMultiplier %v is negative", o.DryMultiplier)
	case o.DryBase < 0 || (o.DryBase > 0 && o.DryBase < 1):
		return errorf(ErrDecode, "DryBase %v must be at least 1 or 0 for the default", o.DryBase)
	case o.DryAllowedLength < 0:
		return errorf(ErrDecode, "DryAllowedLength %d is negative", o.DryAllowedLength)
	case o.DryPenaltyLastN < -1:
		return errorf(ErrDecode, "DryPenaltyLastN %d is below -1", o.DryPenaltyLastN)
	case o.MirostatTau < 0 || o.MirostatEta < 0:
		return errorf(ErrDecode, "MirostatTau %v and MirostatEta %v are negative", o.MirostatTau, o.MirostatEta)
	case (o.MirostatTau > 0) != (o.MirostatEta > 0):
		return errorf(ErrDecode, "MirostatTau %v and MirostatEta %v must both be set to use mirostat", o.MirostatTau, o.MirostatEta)
	}
	for _, s := range o.Stop {
		if s == "" {
			return errorf(ErrDecode, "empty stop sequence")
		}
	}
	for _, b := range o.DrySequenceBreakers {
		if b == "" {
			return errorf(ErrDecode, "empty DRY sequence breaker")
		}
	}
	return nil
}
