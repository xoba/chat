module github.com/xoba/chat

go 1.21.4

replace github.com/xoba/openai => ../openai

require (
	github.com/aws/aws-sdk-go-v2 v1.22.1
	github.com/aws/aws-sdk-go-v2/config v1.22.0
	github.com/aws/aws-sdk-go-v2/service/bedrockruntime v1.3.0
	github.com/pkoukk/tiktoken-go v0.1.6
	github.com/xoba/openai v0.0.1
)

require (
	github.com/alecthomas/jsonschema v0.0.0-20220216202328-9eeeec9d044b // indirect
	github.com/aws/aws-sdk-go-v2/aws/protocol/eventstream v1.5.0 // indirect
	github.com/aws/aws-sdk-go-v2/credentials v1.15.1 // indirect
	github.com/aws/aws-sdk-go-v2/feature/ec2/imds v1.14.2 // indirect
	github.com/aws/aws-sdk-go-v2/internal/configsources v1.2.1 // indirect
	github.com/aws/aws-sdk-go-v2/internal/endpoints/v2 v2.5.1 // indirect
	github.com/aws/aws-sdk-go-v2/internal/ini v1.5.0 // indirect
	github.com/aws/aws-sdk-go-v2/service/internal/presigned-url v1.10.1 // indirect
	github.com/aws/aws-sdk-go-v2/service/sso v1.17.0 // indirect
	github.com/aws/aws-sdk-go-v2/service/ssooidc v1.19.0 // indirect
	github.com/aws/aws-sdk-go-v2/service/sts v1.25.0 // indirect
	github.com/aws/smithy-go v1.16.0 // indirect
	github.com/dlclark/regexp2 v1.10.0 // indirect
	github.com/google/uuid v1.4.0 // indirect
	github.com/iancoleman/orderedmap v0.3.0 // indirect
	github.com/vincent-petithory/dataurl v1.0.0 // indirect
	github.com/xoba/open-golang v1.0.0 // indirect
)
