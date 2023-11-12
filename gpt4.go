package chat

import (
	"fmt"
	"io"
	"log"
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
	switch m {
	case GPT4ModeDefault:
	case GPT4ModeTurbo:
	default:
		return nil, fmt.Errorf("illegal mode: %v", m)
	}
	return gpt4interface{c: c, m: m}, nil
}

type gpt4interface struct {
	c *openai.Client
	m GPT4Mode
}

func (i gpt4interface) String() string {
	return "openai.gpt4"
}

func (i gpt4interface) MaxTokens() int {
	switch i.m {
	case GPT4ModeDefault:
		return 8 * 1024
	case GPT4ModeTurbo:
		return 128 * 1024
	default:
		panic("illegal")
	}
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
	var m2 []openai.Message
	for _, m := range messages {
		var r string
		switch m.Role {
		case RoleSystem:
			r = "system"
		case RoleUser:
			r = "user"
		case RoleAssistant:
			r = "assistant"
		}
		m2 = append(m2, openai.Message{
			Role:    r,
			Content: m.Content,
		})
	}
	var model string
	switch i.m {
	case GPT4ModeDefault:
		model = "gpt-4-0613"
	case GPT4ModeTurbo:
		model = "gpt-4-1106-preview"
	}
	r, err := i.c.Streaming(model, m2, stream)
	if err != nil {
		return nil, err
	}
	var finishReason FinishReason
	switch r.FinishReason {
	case "stop":
		finishReason = FinishReasonStop
	case "length":
		finishReason = FinishReasonLength
	default:
		finishReason = FinishReasonUnknown
	}
	return &Response{
		Content:      r.Content,
		FinishReason: finishReason,
	}, nil
}
