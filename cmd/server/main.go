package main

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"chat-completion-transformer/internal/config"
	"chat-completion-transformer/internal/httpapi"
	"chat-completion-transformer/internal/logging"
	"chat-completion-transformer/internal/upstream"
	"chat-completion-transformer/pkg/transformer"

	"github.com/gin-gonic/gin"
	"go.uber.org/zap"
)

func main() {
	os.Exit(realMain())
}

func realMain() int {
	applicationLogger, err := logging.New()
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "initialize logger: %v\n", err)
		return 1
	}
	runErr := run(applicationLogger.Logger)
	if runErr != nil {
		applicationLogger.Error("server stopped with error", zap.Error(runErr))
	}
	closeErr := applicationLogger.Close()
	if closeErr != nil {
		_, _ = fmt.Fprintf(os.Stderr, "close logger: %v\n", closeErr)
	}
	if runErr != nil {
		return 1
	}
	if closeErr != nil {
		return 1
	}
	return 0
}

func run(logger *zap.Logger) error {
	if logger == nil {
		return errors.New("logger is required")
	}

	settings, err := config.Load("config.yml")
	if err != nil {
		return err
	}
	gin.SetMode(settings.Server.GinMode)

	service, err := transformer.New(transformer.Config{
		Mode:                   transformer.Mode(settings.Transformer.Mode),
		InstructionPolicy:      transformer.InstructionPolicy(settings.Transformer.InstructionPolicy),
		AnthropicEndpoint:      transformer.Endpoint(settings.Transformer.AnthropicEndpoint),
		DefaultMaxOutputTokens: settings.Transformer.DefaultMaxOutputTokens,
		MaxSSEEventBytes:       settings.Server.MaxSSEEventBytes,
		Profiles:               settings.Transformer.Profiles,
		Routes:                 settings.Transformer.Routes,
	})
	if err != nil {
		return fmt.Errorf("create transformer: %w", err)
	}

	upstreamClient, err := upstream.New(settings.Gateway, nil)
	if err != nil {
		return fmt.Errorf("create upstream client: %w", err)
	}

	router, err := httpapi.NewRouter(service, upstreamClient, httpapi.Limits{
		MaxBodyBytes:   settings.Server.MaxBodyBytes,
		MaxStreamBytes: settings.Server.MaxStreamBytes,
	}, logger)
	if err != nil {
		return fmt.Errorf("create HTTP router: %w", err)
	}
	serverErrorLogger, err := zap.NewStdLogAt(logger.Named("http.server"), zap.ErrorLevel)
	if err != nil {
		return fmt.Errorf("create HTTP server logger: %w", err)
	}

	server := &http.Server{
		Addr:              settings.Server.Address,
		Handler:           router,
		ReadHeaderTimeout: settings.Server.ReadHeaderTimeout,
		IdleTimeout:       settings.Server.IdleTimeout,
		ErrorLog:          serverErrorLogger,
	}
	signalContext, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()

	serveErrors := make(chan error, 1)
	logger.Info("starting HTTP server", zap.String("address", settings.Server.Address))
	go func() {
		serveErrors <- server.ListenAndServe()
	}()

	select {
	case serveErr := <-serveErrors:
		return normalizeServeError(serveErr)
	case <-signalContext.Done():
	}

	logger.Info("shutting down HTTP server")
	shutdownContext, cancelShutdown := context.WithTimeout(context.WithoutCancel(signalContext), settings.Server.ShutdownTimeout)
	defer cancelShutdown()
	shutdownErr := server.Shutdown(shutdownContext)
	if shutdownErr != nil {
		closeErr := server.Close()
		serveErr := <-serveErrors
		return errors.Join(
			fmt.Errorf("shut down HTTP server: %w", shutdownErr),
			normalizeCloseError(closeErr),
			normalizeServeError(serveErr),
		)
	}

	serveErr := <-serveErrors
	return normalizeServeError(serveErr)
}

func normalizeServeError(err error) error {
	if err == nil || errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return fmt.Errorf("serve HTTP: %w", err)
}

func normalizeCloseError(err error) error {
	if err == nil || errors.Is(err, http.ErrServerClosed) {
		return nil
	}
	return fmt.Errorf("close HTTP server: %w", err)
}
