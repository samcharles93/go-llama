package main

import "encoding/json"

// The shapes here follow the OpenAI HTTP API closely enough that the official
// client libraries can parse them. Fields the server does not implement (tools,
// logprobs, n>1, ...) are simply absent from the request structs, so clients
// sending them get the default behaviour instead of an error.

type samplingRequest struct {
	// max_tokens and the newer max_completion_tokens both cap the reply.
	MaxTokens           *int     `json:"max_tokens"`
	MaxCompletionTokens *int     `json:"max_completion_tokens"`
	Temperature         *float64 `json:"temperature"`
	TopP                *float64 `json:"top_p"`
	// stop is either a string or an array of strings in the API.
	Stop json.RawMessage `json:"stop"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
	N        *int          `json:"n"`
	samplingRequest
}

type chatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []chatChoice `json:"choices"`
	Usage   usage        `json:"usage"`
}

type chatChoice struct {
	Index        int       `json:"index"`
	Message      chatReply `json:"message"`
	FinishReason string    `json:"finish_reason"`
}

type chatReply struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatChunk struct {
	ID      string        `json:"id"`
	Object  string        `json:"object"`
	Created int64         `json:"created"`
	Model   string        `json:"model"`
	Choices []chunkChoice `json:"choices"`
}

type chunkChoice struct {
	Index int `json:"index"`
	// Delta carries the role on the first frame and the text after that.
	Delta chunkContent `json:"delta"`
	// FinishReason is null until the closing frame, which is why it is a
	// pointer to a string.
	FinishReason *string `json:"finish_reason"`
}

type chunkContent struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

type completionRequest struct {
	Model  string          `json:"model"`
	Prompt json.RawMessage `json:"prompt"`
	Stream bool            `json:"stream"`
	samplingRequest
}

type completionResponse struct {
	ID      string             `json:"id"`
	Object  string             `json:"object"`
	Created int64              `json:"created"`
	Model   string             `json:"model"`
	Choices []completionChoice `json:"choices"`
	Usage   usage              `json:"usage"`
}

type completionChoice struct {
	Text         string `json:"text"`
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason"`
}

type completionChunk struct {
	ID      string                  `json:"id"`
	Object  string                  `json:"object"`
	Created int64                   `json:"created"`
	Model   string                  `json:"model"`
	Choices []completionChunkChoice `json:"choices"`
}

type completionChunkChoice struct {
	Text         string  `json:"text"`
	Index        int     `json:"index"`
	FinishReason *string `json:"finish_reason"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

type embeddingsRequest struct {
	Model string          `json:"model"`
	Input json.RawMessage `json:"input"`
	// encoding_format is accepted and ignored: the vectors are always floats.
	EncodingFormat string `json:"encoding_format"`
}

type embeddingsResponse struct {
	Object string          `json:"object"`
	Data   []embeddingItem `json:"data"`
	Model  string          `json:"model"`
	Usage  embeddingUsage  `json:"usage"`
}

type embeddingItem struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

type embeddingUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
}

type modelList struct {
	Object string      `json:"object"`
	Data   []modelCard `json:"data"`
}

type modelCard struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

type healthBody struct {
	Status   string `json:"status"`
	Model    string `json:"model"`
	Contexts int    `json:"contexts"`
	Busy     int    `json:"busy"`
	Queued   int64  `json:"queued"`
}

type errorResponse struct {
	Error apiError `json:"error"`
}

type apiError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code"`
}
