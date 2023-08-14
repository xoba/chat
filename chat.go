package chat

import (
	"bufio"
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"strconv"
	"strings"

	"github.com/sashabaranov/go-openai"
)

func Chat(key string, maxTokens int, promptFiles ...string) error {
	reader := bufio.NewReader(os.Stdin)
	c, err := newClient(key)
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
	for _, filename := range promptFiles {
		prompt, err := os.ReadFile(filename)
		if err != nil {
			return err
		}
		addSystem(fmt.Sprintf("file %q contains:\n\n%s", filename, string(prompt)))
	}
	var init bool
	for {
		if init {
			r, err := complete(c, maxTokens, os.Stdout, messages)
			if err != nil {
				return fmt.Errorf("gpt can't complete: %v", err)
			}
			switch r.FinishReason {
			case "stop":
				addAssistant(r.Content)
			case "length":
				msg := fmt.Sprintf("\n<token limit of %d for response reached>\n", maxTokens)
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

type client interface {
	CreateChatCompletionStream(context.Context, openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error)
}

type completionResponse struct {
	FinishReason string
	Content      string
}

func complete(c client, maxTokens int, w io.Writer, messages []openai.ChatCompletionMessage) (*completionResponse, error) {
	resp, err := c.CreateChatCompletionStream(context.Background(), openai.ChatCompletionRequest{
		Model:       openai.GPT40613,
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
					fmt.Printf("error: %v\n", err)
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
