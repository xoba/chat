package chat

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"unicode"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime/types"
)

func Claude2(c *bedrockruntime.Client) (Interface, error) {
	return claudeInterface{c: c}, nil
}

type claudeInterface struct {
	c *bedrockruntime.Client
}

func (i claudeInterface) String() string {
	return "anthopic.claude-v2"
}

func (c claudeInterface) Streaming(messages []Message, stream io.Writer) (*Response, error) {
	const (
		humanPrompt     = "\n\nHuman:"
		assistantPrompt = "\n\nAssistant:"
	)
	prompt := new(bytes.Buffer)
	for _, m := range messages {
		switch m.Role {
		case RoleUser:
			fmt.Fprintf(prompt, "%s %s\n", humanPrompt, m.Content)
		case RoleSystem:
			fmt.Fprintf(prompt, "%s %s\n", humanPrompt, m.Content)
		case RoleAssistant:
			fmt.Fprintf(prompt, "%s %s\n", assistantPrompt, m.Content)
		default:
		}
	}
	fmt.Fprintf(prompt, "%s\n", assistantPrompt)
	bedrockReq := bedrockRequest{
		Prompt:           prompt.String(),
		Max:              1000,
		Temperature:      1,
		TopK:             250,
		TopP:             0.999,
		StopSequences:    []string{humanPrompt},
		AnthropicVersion: "bedrock-2023-05-31",
	}
	body, _ := json.MarshalIndent(bedrockReq, "", "  ")
	resp, err := c.c.InvokeModelWithResponseStream(context.Background(), &bedrockruntime.InvokeModelWithResponseStreamInput{
		Body:        body,
		ModelId:     aws.String("anthropic.claude-v2"),
		Accept:      aws.String("*/*"),
		ContentType: aws.String("application/json"),
	})
	if err != nil {
		return nil, err
	}
	s := resp.GetStream()
	r := s.Reader
	defer r.Close()
	content := new(bytes.Buffer)
	var n int
	const debug = false
	first := true
	for e := range r.Events() {
		n++
		if debug {
			fmt.Printf("<event %d: %T>", n, e)
		}
		switch v := e.(type) {
		case *types.ResponseStreamMemberChunk:
			var br bedrockResponse
			if err := json.Unmarshal(v.Value.Bytes, &br); err != nil {
				return nil, err
			}
			if first {
				// for some unknown reason, claude-v2 always returns a leading space
				br.Completion = strings.TrimLeftFunc(br.Completion, unicode.IsSpace)
				first = false
			}
			fmt.Fprint(stream, br.Completion)
			content.WriteString(br.Completion)
		case *types.UnknownUnionMember:
			return nil, fmt.Errorf("unknown union member: %v", v)
		default:
			return nil, fmt.Errorf("union is nil or unknown type: %T %v", v, v)
		}
	}
	if err := r.Err(); err != nil {
		return nil, err
	}
	if debug {
		fmt.Printf("<%d events done>", n)
	}
	return &Response{
		Content:      strings.TrimSpace(content.String()),
		FinishReason: FinishReasonStop,
	}, nil
}

type bedrockRequest struct {
	Prompt           string   `json:"prompt"`
	Max              int      `json:"max_tokens_to_sample"`
	Temperature      float64  `json:"temperature"`
	TopK             int      `json:"top_k"`
	TopP             float64  `json:"top_p"`
	StopSequences    []string `json:"stop_sequences"`
	AnthropicVersion string   `json:"anthropic_version"`
}

func (r bedrockRequest) String() string {
	buf, _ := json.MarshalIndent(r, "", "  ")
	return string(buf)
}

type bedrockResponse struct {
	Completion string `json:"completion,omitempty"`
	StopReason string `json:"stop_reason,omitempty"`
}

func (r bedrockResponse) String() string {
	buf, _ := json.MarshalIndent(r, "", "  ")
	return string(buf)
}
