package llama

/*
#include "llama.h"
*/
import "C"

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"unsafe"
)

// stateMagic identifies a serialised context state produced by this package. It
// wraps llama.cpp's own state blob together with the token history this package
// tracks, because llama.cpp's blob restores the KV cache but not the prompt that
// produced it.
const stateMagic = "GOLMSTATE\x01"

// History returns the tokens currently held in the KV cache, oldest first. The
// slice is a copy, so the caller may retain it.
func (c *Context) History() []Token {
	if c == nil || c.closed.Load() {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	out := make([]Token, len(c.history))
	copy(out, c.history)
	return out
}

// Truncate drops everything after the first keep tokens, which is how a caller
// slides a conversation window forward without paying to re-evaluate the part it
// keeps.
//
// A keep value at or beyond the current length does nothing.
func (c *Context) Truncate(keep int) error {
	if c == nil || c.closed.Load() {
		return ErrClosed
	}
	if keep < 0 {
		return errorf(ErrDecode, "Truncate(%d): keep must not be negative", keep)
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ptr == nil {
		return ErrClosed
	}
	if keep >= len(c.history) {
		return nil
	}
	C.llama_memory_seq_rm(C.llama_get_memory(c.ptr), 0, C.llama_pos(keep), -1)
	c.history = c.history[:keep]
	c.logitsValid = false
	return nil
}

// SaveState serialises the KV cache and the token history so a later process,
// or a later context, can resume the conversation without re-evaluating the
// prompt. Use LoadState to restore it.
//
// The state is tied to the model that produced it; loading it into a context
// from a different model or a differently configured context is undefined.
func (c *Context) SaveState() ([]byte, error) {
	if c == nil || c.closed.Load() {
		return nil, ErrClosed
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ptr == nil {
		return nil, ErrClosed
	}

	size := int(C.llama_state_get_size(c.ptr))
	if size <= 0 {
		return nil, errorf(ErrDecode, "llama.cpp reported no serialisable state")
	}
	state := make([]byte, size)
	written := int(C.llama_state_get_data(c.ptr, (*C.uint8_t)(unsafe.Pointer(&state[0])), C.size_t(size)))
	if written <= 0 {
		return nil, errorf(ErrDecode, "llama_state_get_data wrote no bytes")
	}
	state = state[:written]

	var buf bytes.Buffer
	buf.WriteString(stateMagic)
	if err := binary.Write(&buf, binary.LittleEndian, uint32(len(c.history))); err != nil {
		return nil, wrap(ErrDecode, "encode history length", err)
	}
	for _, t := range c.history {
		if err := binary.Write(&buf, binary.LittleEndian, int32(t)); err != nil {
			return nil, wrap(ErrDecode, "encode history", err)
		}
	}
	if err := binary.Write(&buf, binary.LittleEndian, uint64(len(state))); err != nil {
		return nil, wrap(ErrDecode, "encode state length", err)
	}
	buf.Write(state)
	return buf.Bytes(), nil
}

// SaveStateTo writes SaveState's bytes to w.
func (c *Context) SaveStateTo(w io.Writer) error {
	data, err := c.SaveState()
	if err != nil {
		return err
	}
	_, err = w.Write(data)
	return err
}

// LoadState restores a state produced by SaveState, replacing the context's KV
// cache and history.
func (c *Context) LoadState(data []byte) error {
	if c == nil || c.closed.Load() {
		return ErrClosed
	}
	if !bytes.HasPrefix(data, []byte(stateMagic)) {
		return errors.New("llama: not a go-llama context state")
	}
	r := bytes.NewReader(data[len(stateMagic):])

	var nHistory uint32
	if err := binary.Read(r, binary.LittleEndian, &nHistory); err != nil {
		return wrap(ErrDecode, "decode history length", err)
	}
	if int64(nHistory)*4 > int64(r.Len()) {
		return fmt.Errorf("llama: context state claims %d history tokens but only %d bytes remain", nHistory, r.Len())
	}
	history := make([]Token, nHistory)
	for i := range history {
		var t int32
		if err := binary.Read(r, binary.LittleEndian, &t); err != nil {
			return wrap(ErrDecode, "decode history", err)
		}
		history[i] = Token(t)
	}

	var nState uint64
	if err := binary.Read(r, binary.LittleEndian, &nState); err != nil {
		return wrap(ErrDecode, "decode state length", err)
	}
	if nState != uint64(r.Len()) {
		return fmt.Errorf("llama: context state declares %d bytes but %d remain", nState, r.Len())
	}
	state := make([]byte, nState)
	if _, err := io.ReadFull(r, state); err != nil {
		return wrap(ErrDecode, "read state", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	if c.ptr == nil {
		return ErrClosed
	}
	n := int(C.llama_state_set_data(c.ptr, (*C.uint8_t)(unsafe.Pointer(&state[0])), C.size_t(len(state))))
	if n != len(state) {
		return errorf(ErrDecode, "llama_state_set_data accepted %d of %d bytes", n, len(state))
	}
	c.history = history
	// A saved state carries the KV cache, not the output logits.
	c.logitsValid = false
	return nil
}

// LoadStateFrom reads a state produced by SaveStateTo.
func (c *Context) LoadStateFrom(r io.Reader) error {
	data, err := io.ReadAll(r)
	if err != nil {
		return wrap(ErrDecode, "read state", err)
	}
	return c.LoadState(data)
}
