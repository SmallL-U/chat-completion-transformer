// Package config loads and validates the server configuration.
package config

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"chat-completion-transformer/internal/capabilities"
	"chat-completion-transformer/internal/upstream"

	"github.com/go-viper/mapstructure/v2"
	"github.com/spf13/viper"
)

const (
	defaultPath        = "config.yml"
	// Keep periods literal in model aliases and upstream map keys.
	configKeyDelimiter = "::"
)

var scalarKeys = []string{
	"server::address",
	"server::gin_mode",
	"server::read_header_timeout",
	"server::idle_timeout",
	"server::shutdown_timeout",
	"server::max_body_bytes",
	"server::max_stream_bytes",
	"server::max_sse_event_bytes",
	"gateway::response_header_timeout",
	"transformer::mode",
	"transformer::instruction_policy",
	"transformer::anthropic_endpoint",
}

const defaultMaxOutputTokensEnvironment = "CCT_TRANSFORMER_DEFAULT_MAX_OUTPUT_TOKENS"

type Config struct {
	Server      ServerConfig      `json:"server"`
	Gateway     upstream.Settings `json:"gateway"`
	Transformer TransformerConfig `json:"transformer"`
}

type ServerConfig struct {
	Address           string        `json:"address"`
	GinMode           string        `json:"gin_mode"`
	ReadHeaderTimeout time.Duration `json:"read_header_timeout"`
	IdleTimeout       time.Duration `json:"idle_timeout"`
	ShutdownTimeout   time.Duration `json:"shutdown_timeout"`
	MaxBodyBytes      int64         `json:"max_body_bytes"`
	MaxStreamBytes    int64         `json:"max_stream_bytes"`
	MaxSSEEventBytes  int           `json:"max_sse_event_bytes"`
}

type TransformerConfig struct {
	Mode                   string                    `json:"mode"`
	InstructionPolicy      string                    `json:"instruction_policy"`
	AnthropicEndpoint      capabilities.Endpoint     `json:"anthropic_endpoint"`
	DefaultMaxOutputTokens *int                      `json:"default_max_output_tokens"`
	Profiles               []capabilities.Profile    `json:"profiles"`
	Routes                 []capabilities.ModelRoute `json:"routes"`
}

// Load reads path, applies CCT-prefixed environment overrides, and validates
// the result. An empty path selects config.yml in the working directory.
func Load(path string) (Config, error) {
	if strings.TrimSpace(path) == "" {
		path = defaultPath
	}

	v := viper.NewWithOptions(viper.KeyDelimiter(configKeyDelimiter))
	v.SetConfigFile(path)
	v.SetEnvPrefix("CCT")
	v.SetEnvKeyReplacer(strings.NewReplacer(configKeyDelimiter, "_"))
	v.AllowEmptyEnv(true)

	if err := bindScalarEnvironment(v); err != nil {
		return Config{}, err
	}
	if err := v.ReadInConfig(); err != nil {
		return Config{}, fmt.Errorf("read config %q: %w", path, err)
	}
	if err := applyJSONEnvironment(v); err != nil {
		return Config{}, err
	}

	var result Config
	if err := v.UnmarshalExact(&result, useJSONTags); err != nil {
		return Config{}, fmt.Errorf("decode config %q: %w", path, err)
	}
	if err := applyDefaultMaxOutputTokensEnvironment(&result); err != nil {
		return Config{}, err
	}
	for index := range result.Transformer.Profiles {
		if result.Transformer.Profiles[index].PromptCache.Mode == capabilities.PromptCacheUnset {
			result.Transformer.Profiles[index].PromptCache.Mode = capabilities.PromptCacheNone
		}
	}
	if err := result.Validate(); err != nil {
		return Config{}, fmt.Errorf("validate config %q: %w", path, err)
	}

	return result, nil
}

func applyDefaultMaxOutputTokensEnvironment(config *Config) error {
	raw, ok := os.LookupEnv(defaultMaxOutputTokensEnvironment)
	if !ok {
		return nil
	}

	raw = strings.TrimSpace(raw)
	if raw == "" || strings.EqualFold(raw, "null") {
		config.Transformer.DefaultMaxOutputTokens = nil
		return nil
	}

	value, err := strconv.Atoi(raw)
	if err != nil {
		return fmt.Errorf("decode %s as an integer or null: %w", defaultMaxOutputTokensEnvironment, err)
	}
	config.Transformer.DefaultMaxOutputTokens = &value
	return nil
}

func bindScalarEnvironment(v *viper.Viper) error {
	for _, key := range scalarKeys {
		if err := v.BindEnv(key); err != nil {
			return fmt.Errorf("bind environment for %q: %w", key, err)
		}
	}

	return nil
}

func applyJSONEnvironment(v *viper.Viper) error {
	if err := applyJSONOverride(v, "gateway::upstreams", "CCT_GATEWAY_UPSTREAMS", map[string]upstream.Config{}); err != nil {
		return err
	}
	if err := applyJSONOverride(v, "gateway::routes", "CCT_GATEWAY_ROUTES", map[string]string{}); err != nil {
		return err
	}
	if err := applyJSONOverride(v, "transformer::profiles", "CCT_TRANSFORMER_PROFILES", []capabilities.Profile{}); err != nil {
		return err
	}
	if err := applyJSONOverride(v, "transformer::routes", "CCT_TRANSFORMER_ROUTES", []capabilities.ModelRoute{}); err != nil {
		return err
	}

	return nil
}

func applyJSONOverride[T any](v *viper.Viper, key, environment string, empty T) error {
	raw, ok := os.LookupEnv(environment)
	if !ok {
		return nil
	}
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" || strings.EqualFold(trimmed, "null") {
		v.Set(key, empty)
		return nil
	}

	var value T
	decoder := json.NewDecoder(strings.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&value); err != nil {
		return fmt.Errorf("decode %s as JSON: %w", environment, err)
	}
	if err := ensureJSONEnd(decoder); err != nil {
		return fmt.Errorf("decode %s as JSON: %w", environment, err)
	}

	v.Set(key, value)
	return nil
}

func ensureJSONEnd(decoder *json.Decoder) error {
	var trailing any
	err := decoder.Decode(&trailing)
	if errors.Is(err, io.EOF) {
		return nil
	}
	if err != nil {
		return err
	}

	return errors.New("multiple JSON values")
}

func useJSONTags(config *mapstructure.DecoderConfig) {
	config.TagName = "json"
}
