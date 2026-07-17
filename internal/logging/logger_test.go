package logging

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"go.uber.org/zap"
)

func TestLoggerUsesConsoleAndJSONEncoders(t *testing.T) {
	var stdout bytes.Buffer
	filename := filepath.Join(t.TempDir(), "server.log")
	logger := newLogger(filename, noSyncWriter{Writer: &stdout})
	logger.Info("server started", zap.String("component", "test"))
	if err := logger.Close(); err != nil {
		t.Fatalf("Close() error = %v", err)
	}

	if json.Valid(bytes.TrimSpace(stdout.Bytes())) {
		t.Fatalf("stdout log is JSON, want console text: %s", stdout.String())
	}
	if !bytes.Contains(stdout.Bytes(), []byte("server started")) {
		t.Fatalf("stdout log does not contain message: %s", stdout.String())
	}

	contents, err := os.ReadFile(filename)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	var entry map[string]any
	if err := json.Unmarshal(bytes.TrimSpace(contents), &entry); err != nil {
		t.Fatalf("file log is not JSON: %v: %s", err, contents)
	}
	if entry["message"] != "server started" || entry["component"] != "test" {
		t.Fatalf("file log entry = %#v", entry)
	}
}

func TestLoggerRotationDefaults(t *testing.T) {
	logger := newLogger(filepath.Join(t.TempDir(), "server.log"), noSyncWriter{Writer: io.Discard})
	t.Cleanup(func() {
		if err := logger.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})

	if logger.file.MaxSize != defaultMaxSizeMB {
		t.Fatalf("MaxSize = %d, want %d", logger.file.MaxSize, defaultMaxSizeMB)
	}
	if logger.file.MaxBackups+1 != maxLogFiles {
		t.Fatalf("total log files = %d, want %d", logger.file.MaxBackups+1, maxLogFiles)
	}
}

func TestLoggerKeepsAtMostFiveFilesAfterRotation(t *testing.T) {
	directory := t.TempDir()
	logger := newLogger(filepath.Join(directory, "server.log"), noSyncWriter{Writer: io.Discard})
	t.Cleanup(func() {
		if err := logger.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	})

	for index := 0; index < maxLogFiles+2; index++ {
		logger.Info("rotate log", zap.Int("index", index))
		if err := logger.file.Rotate(); err != nil {
			t.Fatalf("Rotate() error = %v", err)
		}
		time.Sleep(5 * time.Millisecond)
	}
	logger.Info("active log")

	deadline := time.Now().Add(time.Second)
	for {
		files, err := filepath.Glob(filepath.Join(directory, "server*.log"))
		if err != nil {
			t.Fatalf("Glob() error = %v", err)
		}
		if len(files) <= maxLogFiles {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("log files = %d, want at most %d: %v", len(files), maxLogFiles, files)
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestPrepareLogFileCreatesParentDirectory(t *testing.T) {
	filename := filepath.Join(t.TempDir(), "nested", "server.log")
	if err := prepareLogFile(filename); err != nil {
		t.Fatalf("prepareLogFile() error = %v", err)
	}
	if _, err := os.Stat(filename); err != nil {
		t.Fatalf("Stat() error = %v", err)
	}
}

func TestPrepareLogFileRejectsDirectory(t *testing.T) {
	if err := prepareLogFile(t.TempDir()); err == nil {
		t.Fatal("prepareLogFile() error = nil, want directory error")
	}
}
