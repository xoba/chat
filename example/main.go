package main

import (
	"context"
	"embed"
	"io"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/sashabaranov/go-openai"
	"github.com/xoba/chat"
)

//go:embed *.txt
var vfs embed.FS

func main() {
	if err := Run(); err != nil {
		panic(err)
	}
}

func Run() error {
	buf, err := os.ReadFile("openai_key.txt")
	if err != nil {
		return err
	}
	llm1, err := chat.GPT4(openai.NewClient(strings.TrimSpace(string(buf))))
	if err != nil {
		return err
	}
	br, err := NewBedrockRuntime()
	if err != nil {
		return err
	}
	llm2, err := chat.Claude2(br)
	if err != nil {
		return err
	}
	multi, err := chat.NewMultiInterface(llm1, llm2)
	if err != nil {
		return err
	}
	exampleFile, err := NewEmbeddedFile("example.txt")
	if err != nil {
		return err
	}
	config := chat.APIConfig{
		LLMInterface: multi,
		Prompt:       "please describe the included resources, and provide some notable quotes.",
		Print:        false,
	}
	return chat.Streaming(config, exampleFile)
}

func NewBedrockRuntime() (*bedrockruntime.Client, error) {
	c, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		return nil, err
	}
	return bedrockruntime.NewFromConfig(c), nil
}

func NewEmbeddedFile(name string) (chat.File, error) {
	f, err := vfs.Open(name)
	if err != nil {
		return nil, err
	}
	return file{name: name, ReadCloser: f}, nil
}

type file struct {
	name string
	io.ReadCloser
}

func (f file) Metadata() string {
	return f.name
}
