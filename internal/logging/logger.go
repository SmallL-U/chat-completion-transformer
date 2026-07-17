// Package logging configures application logging.
package logging

import (
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"
	"gopkg.in/natefinch/lumberjack.v2"
)

const (
	defaultFilename  = "logs/server.log"
	defaultMaxSizeMB = 100
	maxLogFiles      = 5
)

// Logger writes human-readable logs to stdout and JSON logs to a rotating
// file.
type Logger struct {
	*zap.Logger
	file *lumberjack.Logger
}

// New creates the application logger with the default file rotation policy.
func New() (*Logger, error) {
	if err := prepareLogFile(defaultFilename); err != nil {
		return nil, err
	}

	return newLogger(defaultFilename, noSyncWriter{Writer: os.Stdout}), nil
}

// Close flushes zap and closes the active log file.
func (logger *Logger) Close() error {
	if logger == nil {
		return nil
	}

	return errors.Join(logger.Sync(), logger.file.Close())
}

func newLogger(filename string, stdout zapcore.WriteSyncer) *Logger {
	file := &lumberjack.Logger{
		Filename:   filename,
		MaxSize:    defaultMaxSizeMB,
		MaxBackups: maxLogFiles - 1, // The active file is not counted as a backup.
	}
	level := zap.LevelEnablerFunc(func(logLevel zapcore.Level) bool {
		return logLevel >= zapcore.InfoLevel
	})
	config := encoderConfig()
	core := zapcore.NewTee(
		zapcore.NewCore(zapcore.NewConsoleEncoder(config), stdout, level),
		zapcore.NewCore(zapcore.NewJSONEncoder(config), zapcore.AddSync(file), level),
	)

	return &Logger{
		Logger: zap.New(core, zap.AddCaller()),
		file:   file,
	}
}

func prepareLogFile(filename string) error {
	if err := os.MkdirAll(filepath.Dir(filename), 0o755); err != nil {
		return fmt.Errorf("create log directory: %w", err)
	}

	file, err := os.OpenFile(filename, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("open log file: %w", err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("close log file: %w", err)
	}
	return nil
}

func encoderConfig() zapcore.EncoderConfig {
	return zapcore.EncoderConfig{
		TimeKey:        "time",
		LevelKey:       "level",
		NameKey:        "logger",
		CallerKey:      "caller",
		MessageKey:     "message",
		StacktraceKey:  "stacktrace",
		LineEnding:     zapcore.DefaultLineEnding,
		EncodeLevel:    zapcore.LowercaseLevelEncoder,
		EncodeTime:     zapcore.ISO8601TimeEncoder,
		EncodeDuration: zapcore.StringDurationEncoder,
		EncodeCaller:   zapcore.ShortCallerEncoder,
	}
}

type noSyncWriter struct {
	io.Writer
}

func (noSyncWriter) Sync() error {
	return nil
}
