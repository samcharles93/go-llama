package llama

type CompletionOptions struct {
	MaxTokens    int
	Temperature  float32
	TopK         int32
	TopP         float32
	Seed         uint32
	StopPatterns []string
}

func DefaultCompletionOptions() CompletionOptions {
	return CompletionOptions{
		MaxTokens:   512,
		Temperature: 0.8,
		TopK:        40,
		TopP:        0.9,
		Seed:        1234,
	}
}
