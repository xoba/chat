package chat

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/sashabaranov/go-openai"
	"golang.org/x/time/rate"
)

type Callback func([]openai.ChatCompletionMessage)

type APIConfig struct {
	Key       string // openai key
	MaxTokens int    // max tokens for each completion request (zero means no limit)
	GPT4      bool   // use GPT-4 instead of GPT-3.5
	RPM       int    // limit on tokens per minute (no good token info with streaming)
	Prompt    string // prompt to use for the first request (optional)
	Callback         // callback for messages sent to openai (optional)
}

type File interface {
	OptionalMetadata() string
	io.ReadCloser
}

// manages a streaming chat via stdin/stdout
func Streaming(config APIConfig, promptFiles ...File) error {
	rpm := rate.NewLimiter(RPMLimit(float64(config.RPM)), 1) // allow only one request at a time
	reader := bufio.NewReader(os.Stdin)
	c, err := newClient(config.Key)
	if err != nil {
		return err
	}
	var messages []openai.ChatCompletionMessage
	add := func(role, text string) {
		messages = append(messages, openai.ChatCompletionMessage{
			Role:    role,
			Content: strings.TrimSpace(text),
		})
	}
	addSystem := func(s string) {
		add(openai.ChatMessageRoleSystem, s)
	}
	addAssistant := func(s string) {
		add(openai.ChatMessageRoleAssistant, s)
	}
	addUser := func(s string) {
		add(openai.ChatMessageRoleUser, s)
	}
	for _, file := range promptFiles {
		if err := func() error {
			defer file.Close()
			w := new(bytes.Buffer)
			if _, err := io.Copy(w, file); err != nil {
				return err
			}
			if meta := file.OptionalMetadata(); meta != "" {
				addSystem(fmt.Sprintf("the following resource has metadata:\n\n%s", meta))
			}
			addSystem(w.String())
			return nil
		}(); err != nil {
			return err
		}
	}
	var init bool
	if len(config.Prompt) > 0 {
		init = true
		addSystem(config.Prompt)
	}
	for {
		if callback := config.Callback; callback != nil {
			callback(messages)
		}
		if init {
			if err := rpm.Wait(context.Background()); err != nil {
				return fmt.Errorf("request limiter failed: %v", err)
			}
			var model string
			if config.GPT4 {
				model = openai.GPT4
			} else {
				model = openai.GPT3Dot5Turbo
			}
			r, err := retry(7, 3*time.Second, func() (*completionResponse, error) {
				return complete(c, model, config.MaxTokens, os.Stdout, messages...)
			})
			if err != nil {
				return fmt.Errorf("gpt can't complete: %v", err)
			}
			switch r.FinishReason {
			case "stop":
				addAssistant(r.Content)
			case "length":
				msg := fmt.Sprintf("\n<token limit of %d for response reached>\n", config.MaxTokens)
				fmt.Print(msg)
				addAssistant(r.Content + msg)
			default:
				return fmt.Errorf("bad finish reason: %q", r.FinishReason)
			}
			fmt.Println()
		}
		init = true
		fmt.Print("> ")
		text, err := reader.ReadString('\n')
		if err == io.EOF {
			fmt.Println()
			return nil
		} else if err != nil {
			return fmt.Errorf("can't read from stdin: %v", err)
		}
		text = strings.TrimSpace(text)
		if len(text) == 0 {
			init = false
			continue
		}
		addUser(text)
	}
}

func RPMLimit(requestsPerMinute float64) rate.Limit {
	return rate.Limit(requestsPerMinute / 60)
}

type client interface {
	CreateChatCompletionStream(context.Context, openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error)
}

type completionResponse struct {
	FinishReason string
	Content      string
}

func complete(c client, model string, maxTokens int, w io.Writer, messages ...openai.ChatCompletionMessage) (*completionResponse, error) {
	resp, err := c.CreateChatCompletionStream(context.Background(), openai.ChatCompletionRequest{
		Model:       model,
		Messages:    messages,
		MaxTokens:   maxTokens,
		Temperature: 0.7,
		TopP:        1,
	})
	if err != nil {
		return nil, err
	}
	defer resp.Close()
	q := new(bytes.Buffer)
	out := io.MultiWriter(w, q)
	var finishReason string
	for {
		t, err := resp.Recv()
		if err == io.EOF {
			break
		} else if err != nil {
			var apiError *openai.APIError
			if errors.As(err, &apiError) {
				p := regexp.MustCompile(`(\d+) tokens`)
				if p.MatchString(apiError.Message) {
					m := p.FindStringSubmatch(apiError.Message)
					tokens, convError := strconv.ParseUint(m[1], 10, 64)
					if convError != nil {
						return nil, convError
					}
					fmt.Printf("total tokens = %d; error = %v\n", tokens, err)
					return nil, err
				} else {
					return nil, err
				}
			}
			return nil, err
		}
		choices := t.Choices
		if len(choices) == 0 {
			return nil, fmt.Errorf("no choices")
		}
		firstChoice := choices[0]
		finishReason = string(firstChoice.FinishReason)
		fmt.Fprint(out, firstChoice.Delta.Content)
	}
	fmt.Fprintln(out)
	return &completionResponse{
		FinishReason: finishReason,
		Content:      q.String(),
	}, nil
}

func newClient(key string) (client, error) {
	return openai.NewClient(key), nil
}

func retry[T any](maxTries int, dt time.Duration, f func() (T, error)) (T, error) {
	var zero T
	start := time.Now()
	var tries int
	for {
		r, err := f()
		if err != nil {
			if tries++; tries == maxTries {
				return zero, fmt.Errorf("%w after %d tries, %v", err, tries, time.Since(start))
			}
			log.Printf("try %d, transient error %v, retrying after %v", tries, err, dt)
			time.Sleep(dt)
			dt *= 2
		} else {
			return r, nil
		}
	}
}
