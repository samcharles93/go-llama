package llama

// Linker and compiler flags for the whole package live here, so there is one
// place to change when llama.cpp moves its output or gains native deps.
//
// llama.cpp is built as shared libraries by scripts/build-llama.sh, which
// installs them into llama.cpp/build/bin (the layout of BUILD_SHARED_LIBS=ON;
// older releases used build/src and build/ggml/src instead). The rpath is baked
// in, so binaries run without LD_LIBRARY_PATH.
//
// Override for an installed llama.cpp by exporting CGO_CFLAGS and CGO_LDFLAGS.

/*
#cgo CFLAGS: -I${SRCDIR}/../../llama.cpp/include
#cgo CFLAGS: -I${SRCDIR}/../../llama.cpp/ggml/include
#cgo LDFLAGS: -L${SRCDIR}/../../llama.cpp/build/bin
#cgo LDFLAGS: -Wl,-rpath,${SRCDIR}/../../llama.cpp/build/bin
#cgo LDFLAGS: -lllama -lggml -lggml-base -lggml-cpu
#cgo LDFLAGS: -lstdc++ -lm
*/
import "C"
