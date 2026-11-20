package llama

import "context"

// FitMessages drops the oldest non-system turns until the rendered prompt fits in
// budget tokens, which is what a chat server has to do before it hands a long
// conversation to a context that cannot hold it.
//
// System messages are always kept: dropping them would change the model's
// instructions rather than shorten the history. When even the system messages and
// the newest turn do not fit, the newest turn is returned anyway and the caller
// gets an ErrContextFull error, because silently truncating the user's current
// message is worse than failing the request.
//
// The message slice is not modified.
func (m *Model) FitMessages(messages []ChatMessage, budget int) ([]ChatMessage, error) {
	if m == nil || m.closed.Load() {
		return nil, ErrClosed
	}
	if budget <= 0 {
		return nil, errorf(ErrContextFull, "budget must be positive, got %d", budget)
	}
	if len(messages) == 0 {
		return nil, nil
	}

	var system, turns []ChatMessage
	for _, msg := range messages {
		if msg.Role == "system" {
			system = append(system, msg)
			continue
		}
		turns = append(turns, msg)
	}

	fits := func(candidate []ChatMessage) (bool, error) {
		prompt, err := m.ApplyChatTemplate(candidate, true)
		if err != nil {
			return false, err
		}
		n, err := m.CountTokens(prompt, true)
		if err != nil {
			return false, err
		}
		return n <= budget, nil
	}

	for drop := 0; drop <= len(turns); drop++ {
		candidate := make([]ChatMessage, 0, len(system)+len(turns)-drop)
		candidate = append(candidate, system...)
		candidate = append(candidate, turns[drop:]...)

		ok, err := fits(candidate)
		if err != nil {
			return nil, err
		}
		if ok {
			return candidate, nil
		}
	}

	// Nothing fit. Report how far over budget the irreducible part is.
	candidate := make([]ChatMessage, 0, len(system)+1)
	candidate = append(candidate, system...)
	if len(turns) > 0 {
		candidate = append(candidate, turns[len(turns)-1])
	}
	prompt, err := m.ApplyChatTemplate(candidate, true)
	if err != nil {
		return candidate, err
	}
	n, err := m.CountTokens(prompt, true)
	if err != nil {
		return candidate, err
	}
	return candidate, errorf(ErrContextFull, "the system prompt and newest turn need %d tokens, budget is %d; raise ContextSize or shorten the message", n, budget)
}

// ChatFitting renders messages through FitMessages and generates a reply, so the
// caller does not have to know the context size. reserve is the number of tokens
// generation may use.
func (c *Context) ChatFitting(ctx context.Context, messages []ChatMessage, opts CompletionOptions, reserve int) (Result, error) {
	if c == nil || c.closed.Load() || c.model == nil {
		return Result{}, ErrClosed
	}
	if reserve < 0 {
		reserve = 0
	}
	fitted, err := c.model.FitMessages(messages, c.nCtx-reserve)
	if err != nil {
		return Result{}, err
	}
	return c.Chat(ctx, fitted, opts)
}
