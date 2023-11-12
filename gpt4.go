package chat

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"log"
	"regexp"
	"strconv"
	"strings"

	"github.com/pkoukk/tiktoken-go"
	"github.com/xoba/openai"
)

//go:generate stringer -type=GPT4Mode
type GPT4Mode int

const (
	_ GPT4Mode = iota
	GPT4ModeDefault
	GPT4ModeTurbo
)

func GPT4(m GPT4Mode, c *openai.Client) (LLMInterface, error) {
	return gpt4interface{c: c, m: m}, nil
}

type gpt4interface struct {
	c *openai.Client
	m GPT4Mode
}

func (i gpt4interface) String() string {
	return "openai.gpt4"
}

func (gpt4interface) MaxTokens() int {
	return 128 * 1024
}

// seems to be a precise estimate
func (i gpt4interface) TokenEstimate(messages []Message) (int, error) {
	n, err := numTokensFromMessages(messages, "gpt-4")
	if err != nil {
		return 0, err
	}
	return n, nil
}

func numTokensFromMessages(messages []Message, model string) (int, error) {
	tkm, err := tiktoken.EncodingForModel(model)
	if err != nil {
		return 0, fmt.Errorf("encoding for model: %v", err)
	}
	var tokensPerMessage int
	switch model {
	case "gpt-3.5-turbo-0613",
		"gpt-3.5-turbo-16k-0613",
		"gpt-4-0314",
		"gpt-4-32k-0314",
		"gpt-4-0613",
		"gpt-4-32k-0613":
		tokensPerMessage = 3
	case "gpt-3.5-turbo-0301":
		tokensPerMessage = 4 // every message follows <|start|>{role/name}\n{content}<|end|>\n
	default:
		if strings.Contains(model, "gpt-3.5-turbo") {
			log.Println("warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
			return numTokensFromMessages(messages, "gpt-3.5-turbo-0613")
		} else if strings.Contains(model, "gpt-4") {
			//log.Println("warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
			return numTokensFromMessages(messages, "gpt-4-0613")
		} else {
			return 0, fmt.Errorf("num_tokens_from_messages() is not implemented for model %s. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.", model)
		}
	}
	var numTokens int
	for _, message := range messages {
		numTokens += tokensPerMessage
		numTokens += len(tkm.Encode(message.Content, nil, nil))
		var role string
		switch message.Role {
		case RoleSystem:
			role = "system"
		case RoleUser:
			role = "user"
		case RoleAssistant:
			role = "assistant"
		default:
			return 0, fmt.Errorf("unknown role: %v", message.Role)
		}

		numTokens += len(tkm.Encode(role, nil, nil))
	}
	numTokens += 3 // every reply is primed with <|start|>assistant<|message|>
	return numTokens, nil
}

func (i gpt4interface) Streaming(messages []Message, stream io.Writer) (*Response, error) {
	list, err := openaiMessages(messages)
	if err != nil {
		return nil, err
	}
	var m string
	switch i.m {
	case GPT4ModeDefault:
		m = "gpt-4"
	case GPT4ModeTurbo:
		m = "gpt-4-1106-preview"
	}
	r, err := complete(i.c, m, 0, stream, list...)
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

type completionResponse struct {
	FinishReason string
	Content      string
}
