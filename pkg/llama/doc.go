// Package llama binds llama.cpp: it loads GGUF models, generates text, streams
// tokens, applies chat templates and computes embeddings.
//
// A Model owns the weights and the vocabulary. A Context owns a KV cache and is
// safe for concurrent use, serialising generation internally. Contexts share one
// Model, which is how a server serves parallel requests from one copy of the
// weights.
//
//	llama.Init()
//	defer llama.Shutdown()
//	model, err := llama.LoadModel("model.gguf")
//	defer model.Close()
//	c, err := model.NewContext()
//	defer c.Close()
//	res, err := c.Generate(ctx, "Once upon a time", llama.DefaultCompletionOptions())
package llama
