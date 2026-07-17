package main

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"

	"chat-completion-transformer/internal/config"
	"chat-completion-transformer/internal/httpapi"
	"chat-completion-transformer/pkg/transformer"

	"github.com/gin-gonic/gin"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
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

	router, err := httpapi.NewRouter(service, httpapi.Limits{
		MaxBodyBytes:   settings.Server.MaxBodyBytes,
		MaxStreamBytes: settings.Server.MaxStreamBytes,
	})
	if err != nil {
		return fmt.Errorf("create HTTP router: %w", err)
	}

	server := &http.Server{
		Addr:              settings.Server.Address,
		Handler:           router,
		ReadHeaderTimeout: settings.Server.ReadHeaderTimeout,
		IdleTimeout:       settings.Server.IdleTimeout,
	}
	signalContext, stopSignals := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stopSignals()

	serveErrors := make(chan error, 1)
	go func() {
		serveErrors <- server.ListenAndServe()
	}()

	select {
	case serveErr := <-serveErrors:
		return normalizeServeError(serveErr)
	case <-signalContext.Done():
	}

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
