# go-llama examples

Five runnable programs built on `go-llama/pkg/llama`. Every one of them uses the
standard library only and needs a GGUF file; the flags all include `-ngl`, which
defaults to 0 so a run stays on the CPU unless you ask for the GPU.

```
export GOFLAGS=-buildvcs=false
MODEL=/path/to/instruct-model.gguf
EMBED_MODEL=/path/to/embedding-model.gguf
```

## chat

Interactive chat that keeps the conversation, applies the model's own chat
template, and streams the reply token by token.

```
go run ./examples/chat -model $MODEL -max-tokens 128
```

What it demonstrates: `Stream` over an `iter.Seq2`, the KV cache staying warm
across turns so only the new message is evaluated, and context cancellation (one
Ctrl-C cancels the reply in flight and keeps the partial text in the history, so
the next message continues from where it stopped). Slash commands are `/reset`,
`/system <text>`, `/stats` and `/exit`; `exit` and `quit` work without the slash
too.

Flags: `-model` (required), `-system`, `-temp`, `-top-p`, `-max-tokens`, `-ctx`,
`-ngl`, `-raw` (skip the chat template and send `User:` / `Assistant:` lines
instead, for base models).

```
printf 'hi\nexit\n' | go run ./examples/chat -model $MODEL -max-tokens 16
```

The reply and the `[end of reply, 9 tokens, 41.7 tok/s]` trailer go to stdout;
hints and errors go to stderr. The `> ` prompt is printed only when stdin is a
terminal, so a piped run stays clean.

## generate

One completion, streamed, with throughput afterwards. The smallest useful program
in the set.

```
go run ./examples/generate -model $MODEL -prompt "Name one colour." -max-tokens 16
```

What it demonstrates: `Stream`, `Metrics().TokensPerSecond()`, and applying the
chat template to a single message pair instead of a whole history.

Flags: `-model` (required), `-prompt`, `-max-tokens`, `-temp`, `-system`, `-ctx`,
`-ngl`, `-raw`.

## server

OpenAI-compatible HTTP server in front of one loaded model.

```
go run ./examples/server -model $MODEL -addr 127.0.0.1:8080 -parallel 2
```

```
curl -s localhost:8080/v1/chat/completions \
  -d '{"messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'

curl -sN localhost:8080/v1/chat/completions \
  -d '{"stream":true,"messages":[{"role":"user","content":"Say hi"}],"max_tokens":16}'
```

What it demonstrates:

- A pool of `-parallel` contexts created from the one loaded model. Receiving
  from the pool channel is the semaphore, so a request that finds every context
  busy waits instead of sharing a KV cache with another request.
- Server-sent events shaped like OpenAI's: one `data: {json}` frame per token,
  a closing frame with `finish_reason`, then `data: [DONE]`.
- Graceful shutdown. SIGINT or SIGTERM stops accepting connections and waits up
  to `-drain` for in-flight replies; the base context is cancelled when the
  drain expires, which aborts a generation that is still running.
- Per-request deadlines derived from the request context, and a request body
  size limit, so a slow client cannot pin a context forever.
- JSON errors with real status codes: 400 for a malformed request, 404 for an
  unknown model, 405 for a wrong method, 413 for an oversized body, 429 when
  `-max-queue` is set and every waiter slot is taken, 500 for a generation
  failure, 504 when the request deadline expires.

Routes: `POST /v1/chat/completions`, `POST /v1/completions`,
`POST /v1/embeddings`, `GET /v1/models`, `GET /healthz`.

Flags: `-model` (required), `-addr`, `-ctx`, `-timeout`, `-parallel`, `-ngl`,
`-api-key` (require `Authorization: Bearer <key>` on `/v1` routes), `-max-queue`
(waiters allowed before 429; 0 waits forever), `-max-body`, `-drain`,
`-embed-model` (a GGUF embedding model; enables `/v1/embeddings`),
`-embed-from-chat-model` (opt in to embedding with the chat model),
`-embed-ctx`, `-embed-batch` (texts per embedding batch), `-embed-pool`.

Embeddings run on a model and context of their own, never on the pooled chat
contexts, because a context created with `Embeddings` enabled clears its KV cache
on every call. Pooling a chat model's hidden states yields vectors that are
numerically fine and semantically useless, so `/v1/embeddings` is opt-in:

```
go run ./examples/server -model $CHAT_MODEL -embed-model $EMBED_MODEL
curl -s localhost:8080/v1/embeddings -d '{"input":["a cat","a kitten"]}'
```

Without `-embed-model` (or the explicit `-embed-from-chat-model`) the endpoint
answers 400 explaining what to pass. `-embed-pool` must match the model (mean for
encoder models such as embeddinggemma, cls for BERT-style ones); the pooling mode
cannot be read back from the model through this package, so it is a flag.
`-embed-batch` is both the number of texts embedded per batch and the sequence
count of the embedding context.

## embed

JSON embeddings for text arguments or stdin lines, plus cosine similarity.

```
go run ./examples/embed -model $EMBED_MODEL "a cat sits on the mat" "a kitten sits on the rug"

printf 'the quick brown fox\nthe lazy dog sleeps\n' \
  | go run ./examples/embed -model $EMBED_MODEL

go run ./examples/embed -model $EMBED_MODEL -cosine "a cat sits on the mat" "quantum chromodynamics"
```

What it demonstrates: `EmbedBatch` on a context created with `Embeddings`
enabled, L2 normalisation, and a similarity check that shows whether the pooling
mode matches the model (related sentences score around 0.86 with mean pooling,
unrelated ones around 0.60).

Flags: `-model` (required), `-normalize` (default true), `-cosine`, `-ctx`,
`-pool` (`mean`, `cls`, `last` or `none`), `-ngl`.

## tokenize

Token ids and pieces for a prompt, which is how you debug a chat template or pick
a stop sequence.

```
go run ./examples/tokenize -model $MODEL "Hello there"
go run ./examples/tokenize -model $MODEL -chat -system "You are terse." "Hi"
printf 'export const x = 1;\n' | go run ./examples/tokenize -model $MODEL
```

What it demonstrates: `Vocab.Tokenize`, `TokenToPiece` and `Detokenize`, the
`bos`/`eos`/`eog` flags that tell you which tokens end generation, and
`ApplyChatTemplate` so the tokenized text is the prompt the model actually sees.

Flags: `-model` (required), `-no-bos`, `-chat`, `-system`, `-show-template`,
`-ngl`.

```
  idx       id  bytes  flags  piece
    0        1     15  bos    "<|startoftext|>"
    1        6     12         "<|im_start|>"
    2    24131      6         "system"
```

## Notes on the library

- `Stream` returns only text pieces, so a streamed reply cannot report the exact
  `StopReason` the way `Generate` does. `chat` and `server` rebuild it from the
  perf counters: an error from the iterator means the reply was cancelled, and
  `Metrics().GeneratedTokens+1 == MaxTokens` means the token cap stopped it.
- A context created without an explicit `Pooling` mode returns a vector of
  zeros, because llama.cpp only fills the pooled output buffer when a pooling
  mode is selected. Both embedding examples therefore pass `-pool`.
