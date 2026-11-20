package llama

/*
#include "llama.h"
#include <stdlib.h>
*/
import "C"

import (
	"errors"
	"sync/atomic"
	"unsafe"
)

// ErrBatchFull is returned by Batch.Add when the batch is already at capacity.
// It replaces the buffer overflow the previous implementation walked into.
var ErrBatchFull = errors.New("llama: batch is full")

// SeqID identifies a sequence inside a batch. Generation uses sequence 0.
type SeqID = int32

// Batch is a growable-free, capacity-checked token batch. Create one per
// context and reuse it: llama_batch_init allocates once and Reset rewinds it.
//
// A Batch is not safe for concurrent use; the Context that owns it serialises
// access.
type Batch struct {
	batch  C.struct_llama_batch
	cap    int32
	nSeq   int32
	closed atomic.Bool
}

// NewBatch allocates a batch of up to nTokens tokens, each carrying up to
// nSeqMax sequence ids.
func NewBatch(nTokens, nSeqMax int32) (*Batch, error) {
	if nTokens <= 0 {
		return nil, errors.New("llama: batch needs a positive token capacity")
	}
	if nSeqMax <= 0 {
		nSeqMax = 1
	}
	b := C.llama_batch_init(C.int32_t(nTokens), 0, C.int32_t(nSeqMax))
	if b.token == nil {
		return nil, errors.New("llama: out of memory allocating batch")
	}
	return &Batch{batch: b, cap: nTokens, nSeq: nSeqMax}, nil
}

// Reset rewinds the batch so it can be filled again. It does not free memory.
func (b *Batch) Reset() { b.batch.n_tokens = 0 }

// Len reports how many tokens the batch currently holds.
func (b *Batch) Len() int { return int(b.batch.n_tokens) }

// Cap reports how many tokens the batch can hold.
func (b *Batch) Cap() int { return int(b.cap) }

// Add appends one token. pos is its absolute position in the sequence, logits
// asks the backend for output logits at this position (only the last token of
// a decode needs them), and seqIDs lists the sequences the token belongs to.
func (b *Batch) Add(token Token, pos int32, seqIDs []SeqID, logits bool) error {
	if b.closed.Load() {
		return ErrClosed
	}
	if int(b.batch.n_tokens) >= int(b.cap) {
		return ErrBatchFull
	}
	if len(seqIDs) == 0 {
		return errors.New("llama: token needs at least one sequence id")
	}
	if int32(len(seqIDs)) > b.nSeq {
		return errorf(ErrBatchFull, "token needs %d sequence ids, batch holds %d", len(seqIDs), b.nSeq)
	}

	i := b.batch.n_tokens
	row := func(ptr unsafe.Pointer, elem int) unsafe.Pointer {
		return unsafe.Add(ptr, uintptr(i)*uintptr(elem))
	}

	*(*C.llama_token)(row(unsafe.Pointer(b.batch.token), int(unsafe.Sizeof(C.llama_token(0))))) = C.llama_token(token)
	*(*C.llama_pos)(row(unsafe.Pointer(b.batch.pos), int(unsafe.Sizeof(C.llama_pos(0))))) = C.llama_pos(pos)
	*(*C.int32_t)(row(unsafe.Pointer(b.batch.n_seq_id), int(unsafe.Sizeof(C.int32_t(0))))) = C.int32_t(len(seqIDs))

	// seq_id is an array of pointers to per-token arrays owned by llama.cpp.
	rowPtrs := unsafe.Slice((**C.llama_seq_id)(unsafe.Pointer(b.batch.seq_id)), int(b.cap))
	dst := unsafe.Slice(rowPtrs[i], int(b.nSeq))
	for n, id := range seqIDs {
		dst[n] = C.llama_seq_id(id)
	}

	var flag C.int8_t
	if logits {
		flag = 1
	}
	*(*C.int8_t)(row(unsafe.Pointer(b.batch.logits), int(unsafe.Sizeof(C.int8_t(0))))) = flag

	b.batch.n_tokens++
	return nil
}

// AddOne appends a single token belonging to sequence 0.
func (b *Batch) AddOne(token Token, pos int32, logits bool) error {
	return b.Add(token, pos, []SeqID{0}, logits)
}

// Close frees the batch. It is safe to call more than once.
func (b *Batch) Close() error {
	if b == nil || b.closed.Swap(true) {
		return nil
	}
	C.llama_batch_free(b.batch)
	b.batch = C.struct_llama_batch{}
	b.cap, b.nSeq = 0, 0
	return nil
}

// Free is Close, kept for callers written against the old API.
//
// Deprecated: use Close.
func (b *Batch) Free() { _ = b.Close() }
