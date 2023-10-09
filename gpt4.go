package chat

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"regexp"
	"strconv"

	"github.com/sashabaranov/go-openai"
)

func GPT4(c *openai.Client) (Interface, error) {
	return gpt4interface{c: c}, nil
}

type gpt4interface struct {
	c *openai.Client
}

func (i gpt4interface) String() string {
	return "openai.gpt4"
}

func (i gpt4interface) Streaming(messages []Message, stream io.Writer) (*Response, error) {
	var list []openai.ChatCompletionMessage
	for _, m := range messages {
		var role string
		switch m.Role {
		case RoleSystem:
			role = "system"
		case RoleHuman:
			role = "human"
		case RoleAssistant:
			role = "assistant"
		default:
			return nil, fmt.Errorf("unknown role: %v", m.Role)
		}
		list = append(list, openai.ChatCompletionMessage{
			Role:    role,
			Content: m.Content,
		})
	}
	r, err := complete(i.c, openai.GPT4, 0, stream, list...)
	if err != nil {
		return nil, err
	}
	out := Response{
		Content: r.Content,
	}
	switch r.FinishReason {
	case "stop":
		out.FinishReason = FinishReasonStop
	case "length":
		out.FinishReason = FinishReasonLength
	default:
		out.FinishReason = FinishReasonUnknown
	}
	return &out, nil
}

func complete(c client, model string, maxTokens int, stream io.Writer, messages ...openai.ChatCompletionMessage) (*completionResponse, error) {
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
	out := io.MultiWriter(stream, q)
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

type client interface {
	CreateChatCompletionStream(context.Context, openai.ChatCompletionRequest) (*openai.ChatCompletionStream, error)
}

type completionResponse struct {
	FinishReason string
	Content      string
}
