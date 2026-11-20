package llama

/*
#include "llama.h"
*/
import "C"
import (
	"io"
	"log/slog"
	"os"
	"strings"
	"sync"
	"unsafe"
)

// Logger receives llama.cpp's log output. It defaults to a warning-level
// handler on stderr, because llama.cpp's info output (model metadata, backend
// selection) is noise for most programs.
//
// Set it before loading a model to see that output; see SetLogLevel.
// Log levels for SetLogLevel, mirroring slog.
const (
	LogLevelDebug = slog.LevelDebug
	LogLevelInfo  = slog.LevelInfo
	LogLevelWarn  = slog.LevelWarn
	LogLevelError = slog.LevelError
)

var Logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: defaultLogLevel()}))

// defaultLogLevel reads GO_LLAMA_LOG_LEVEL so a deployment can turn llama.cpp's
// chatter up or down without a code change. Unrecognised values keep the
// warning default.
func defaultLogLevel() slog.Level {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("GO_LLAMA_LOG_LEVEL"))) {
	case "debug":
		return slog.LevelDebug
	case "info":
		return slog.LevelInfo
	case "warn", "warning":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelWarn
	}
}

// llama.cpp's log severities, as declared in ggml.h.
const (
	levelNone  = 0
	levelDebug = 1
	levelInfo  = 2
	levelWarn  = 3
	levelError = 4
	levelCont  = 5
)

var (
	logMu      sync.Mutex
	pending    strings.Builder
	pendingLvl int
	havePend   bool
)

// SetLogger replaces the logger llama.cpp writes to. Passing nil silences it.
func SetLogger(l *slog.Logger) {
	logMu.Lock()
	defer logMu.Unlock()
	if l == nil {
		Logger = slog.New(slog.NewTextHandler(io.Discard, nil))
		return
	}
	Logger = l
}

// SetLogLevel sets the minimum severity the default logger reports. Use
// slog.LevelDebug to see everything llama.cpp emits, including per-tensor
// backend placement.
func SetLogLevel(level slog.Level) {
	logMu.Lock()
	defer logMu.Unlock()
	Logger = slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))
}

// goLogBridge receives llama.cpp's log callback. It runs on whichever thread is
// inside llama.cpp, which is not necessarily a Go thread, so everything it
// touches must be locked.
//
// llama.cpp emits text, not lines: one logical message often arrives as several
// calls, with every call but the last tagged GGML_LOG_LEVEL_CONT. The
// previous implementation logged each fragment as its own entry, which split
// sentences across records, and inverted the level mapping so errors were logged
// at debug.
//
//export goLogBridge
func goLogBridge(level C.int, text *C.char, _ unsafe.Pointer) {
	if text == nil {
		return
	}
	handleLogFragment(int(level), C.GoString(text))
}

// handleLogFragment assembles one llama.cpp log fragment. It is split out of the
// cgo bridge so the level mapping and the continuation rules are testable
// without a live backend.
func handleLogFragment(level int, msg string) {
	if msg == "" {
		return
	}

	logMu.Lock()
	defer logMu.Unlock()

	lvl := level
	if lvl != levelCont {
		flushPendingLocked()
		pendingLvl = lvl
	}
	pending.WriteString(msg)
	if !strings.HasSuffix(msg, "\n") {
		// No newline yet, so the message is still being written. A following
		// fragment with a level of its own flushes it, and FlushLogs covers the
		// case where the last message never gets one.
		havePend = true
		return
	}
	flushPendingLocked()
}

// flushPendingLocked writes the assembled message to the logger. Callers hold
// logMu.
func flushPendingLocked() {
	if !havePend && pending.Len() == 0 {
		return
	}
	msg := pending.String()
	pending.Reset()
	havePend = false
	emitLog(pendingLvl, msg)
}

// emitLog writes one assembled message. Callers hold logMu.
func emitLog(level int, raw string) {
	body := strings.TrimRight(raw, "\n")
	if body == "" {
		return
	}
	// This package caps the KV cache at MaxDefaultContextSize by design, so
	// llama.cpp's "n_ctx_train not fully utilized" notice fires on almost every
	// context. Report it once per call at debug rather than warning about a
	// decision the caller made.
	if strings.Contains(body, "n_ctx_train") && strings.Contains(body, "not be utilized") {
		level = 1
	}
	attrs := []any{"source", "llama.cpp"}
	switch level {
	case levelDebug:
		Logger.Debug(body, attrs...)
	case levelInfo:
		Logger.Info(body, attrs...)
	case levelWarn:
		Logger.Warn(body, attrs...)
	case levelError:
		Logger.Error(body, attrs...)
	default:
		Logger.Info(body, attrs...)
	}
}

// FlushLogs writes any partially assembled llama.cpp message. Call it before
// exiting when the last log line matters.
func FlushLogs() {
	logMu.Lock()
	defer logMu.Unlock()
	flushPendingLocked()
}
