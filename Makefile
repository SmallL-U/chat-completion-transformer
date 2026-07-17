.PHONY: build run fmt test test-race vet check tidy

build:
	go build ./...

run:
	go run ./cmd/server

fmt:
	go fmt ./...

test:
	go test ./...

test-race:
	go test -race ./...

vet:
	go vet ./...

check: build test vet

tidy:
	go mod tidy
