package llama

/*
#cgo CFLAGS: -I${SRCDIR}/../../llama.cpp/include -I${SRCDIR}/../../llama.cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../../llama.cpp/build/bin -lllama -lggml -lggml-base -lggml-cpu -Wl,-rpath,${SRCDIR}/../../llama.cpp/build/bin
#include "llama.h"
*/
import "C"
import (
	"unsafe"
)

type Batch struct {
	batch C.struct_llama_batch
}

func NewBatch(nTokens int32, embd int32, nSeqMax int32) *Batch {
	return &Batch{
		batch: C.llama_batch_init(C.int32_t(nTokens), C.int32_t(embd), C.int32_t(nSeqMax)),
	}
}

func (b *Batch) Free() {
	C.llama_batch_free(b.batch)
}

func (b *Batch) Clear() {
	b.batch.n_tokens = 0
}

func (b *Batch) Add(token Token, pos int32, seqIDs []int32, logits bool) {
	idx := b.batch.n_tokens

	// Set token
	tokens := (*[1 << 30]C.llama_token)(unsafe.Pointer(b.batch.token))
	tokens[idx] = C.llama_token(token)

	// Set pos
	positions := (*[1 << 30]C.llama_pos)(unsafe.Pointer(b.batch.pos))
	positions[idx] = C.llama_pos(pos)

	// Set n_seq_id
	nSeqIDs := (*[1 << 30]C.int32_t)(unsafe.Pointer(b.batch.n_seq_id))
	nSeqIDs[idx] = C.int32_t(len(seqIDs))

	// Set seq_id
	seqIDPtrs := (*[1 << 30]*C.llama_seq_id)(unsafe.Pointer(b.batch.seq_id))
	for i, sid := range seqIDs {
		currentSeqIDArr := (*[1 << 30]C.llama_seq_id)(unsafe.Pointer(seqIDPtrs[idx]))
		currentSeqIDArr[i] = C.llama_seq_id(sid)
	}

	// Set logits
	logitsArr := (*[1 << 30]C.int8_t)(unsafe.Pointer(b.batch.logits))
	if logits {
		logitsArr[idx] = 1
	} else {
		logitsArr[idx] = 0
	}

	b.batch.n_tokens++
}
