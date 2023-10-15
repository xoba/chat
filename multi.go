package chat

import (
	"fmt"
	"io"
)

// uses the first smaller capacity one, until tokens exceed its limit, then uses the second one
func NewMultiInterface(firstSmaller, secondLarger LLMInterface) (LLMInterface, error) {
	if firstSmaller.MaxTokens() >= secondLarger.MaxTokens() {
		return nil, fmt.Errorf("first interface should have less capacity than second")
	}
	return multiInterface{firstSmaller, secondLarger}, nil
}

type multiInterface struct {
	first, second LLMInterface
}

func (i multiInterface) MaxTokens() int {
	return i.first.MaxTokens()
}

func (i multiInterface) String() string {
	return fmt.Sprintf("%s / %s", i.first, i.second)
}

func (i multiInterface) TokenEstimate(messages []Message) (int, error) {
	return i.first.TokenEstimate(messages)
}

func (i multiInterface) Streaming(messages []Message, stream io.Writer) (*Response, error) {
	over := func(i LLMInterface) (bool, int, error) {
		n, err := i.TokenEstimate(messages)
		if err != nil {
			return false, 0, err
		}
		const response = 1000
		max := i.MaxTokens()
		return n+response > max, n, nil
	}
	firstOver, n0, err := over(i.first)
	if err != nil {
		return nil, err
	}
	if firstOver {
		secondOver, n1, err := over(i.second)
		if err != nil {
			return nil, err
		}
		if false && secondOver {
			return nil, fmt.Errorf("%d / %d tokens too big for either model", n0, n1)
		}
		return i.second.Streaming(messages, stream)
	}
	return i.first.Streaming(messages, stream)
}
