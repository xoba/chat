package chat

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"log"
	"os"
	"strings"
	"time"
)

type LLMInterface interface {
	// the context window capacity of the LLM
	MaxTokens() int
	// estimates how many tokens are used by the messages
	TokenEstimate(messages []Message) (int, error)
	// streams the response of the LLM to the messages
	Streaming(messages []Message, stream io.Writer) (*Response, error)
}

type Message struct {
	Role    Role
	Content string
}

type Response struct {
	Content      string
	FinishReason FinishReason
}

//go:generate stringer -type=Role
type Role int

const (
	_ Role = iota
	RoleSystem
	RoleUser
	RoleAssistant
)

//go:generate stringer -type=FinishReason
type FinishReason int

const (
	_ FinishReason = iota
	FinishReasonStop
	FinishReasonLength
	FinishReasonUnknown
)

type APIConfig struct {
	LLMInterface
	Prompt string // prompt to use for the first request (optional)
	Print  bool   // whether to print the initial messages to stdout
}

type File interface {
	Metadata() string
	io.ReadCloser
}

// manages a streaming chat via stdin/stdout
func Streaming(config APIConfig, promptFiles ...File) error {
	reader := bufio.NewReader(os.Stdin)
	var messages []Message
	add := func(role Role, text string) {
		messages = append(messages, Message{
			Role:    role,
			Content: strings.TrimSpace(text),
		})
	}
	addAssistant := func(s string) {
		add(RoleAssistant, s)
	}
	addUser := func(s string) {
		add(RoleUser, s)
	}
	addSystem := func(s string) {
		add(RoleSystem, s)
	}
	for _, file := range promptFiles {
		if err := func() error {
			defer file.Close()
			w := new(bytes.Buffer)
			if _, err := io.Copy(w, file); err != nil {
				return err
			}
			if meta := file.Metadata(); len(meta) > 0 {
				addSystem(fmt.Sprintf("the following resource has metadata:\n\n%s\n\nhere's the contents of the resource:\n\n%s", meta, w))
			} else {
				addSystem(w.String())
			}
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
	if config.Print {
		for _, m := range messages {
			fmt.Printf("%s:\n\n%s\n\n", m.Role, m.Content)
		}
	}
	for {
		if init {
			r, err := retry(7, 3*time.Second, func() (*Response, error) {
				return config.Streaming(messages, os.Stdout)
			})
			if err != nil {
				return fmt.Errorf("gpt can't complete: %v", err)
			}
			switch r.FinishReason {
			case FinishReasonStop:
				addAssistant(r.Content)
			case FinishReasonLength:
				msg := fmt.Sprintf("\n<token limit for response reached>\n")
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
