package llama

import "C"
import (
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
)

var Logger *slog.Logger

func init() {
	SetupLogger()
}

func SetupLogger() {
	configDir, err := os.UserConfigDir()
	if err != nil {
		configDir = ".go-llama"
	} else {
		configDir = filepath.Join(configDir, ".go-llama")
	}
	_ = os.MkdirAll(configDir, 0755)
	logPath := filepath.Join(configDir, "app.log")

	f, err := os.OpenFile(logPath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
	var w io.Writer = os.Stderr
	if err == nil {
		w = io.MultiWriter(os.Stderr, f)
	}

	opts := &slog.HandlerOptions{
		Level: slog.LevelDebug,
	}
	Logger = slog.New(slog.NewTextHandler(w, opts))
	slog.SetDefault(Logger)
}

//export GoLogCallback
func GoLogCallback(level int, text *C.char) {
	if Logger == nil {
		return
	}

	msg := C.GoString(text)
	msg = strings.TrimSpace(msg)
	if msg == "" {
		return
	}

	// Map llama.cpp levels
	// 1=Error, 2=Warn, 3=Info, 4=Debug usually.
	switch level {
	case 1: // GGML_LOG_LEVEL_ERROR
		Logger.Error(msg, "source", "llama.cpp")
	case 2: // GGML_LOG_LEVEL_WARN
		Logger.Warn(msg, "source", "llama.cpp")
	case 3: // GGML_LOG_LEVEL_INFO
		Logger.Info(msg, "source", "llama.cpp")
	case 4: // GGML_LOG_LEVEL_DEBUG
		Logger.Debug(msg, "source", "llama.cpp")
	default:
		Logger.Info(msg, "source", "llama.cpp", "level", level)
	}
}
