# Access LLM

This Python script provides convenient functions to interact with various Large Language Models (LLMs) via AWS Bedrock (Claude Sonnet 3.5, 3.7, 4) and OpenAI (GPT-3.5-turbo) using both `boto3` and LangChain. It includes robust error handling and retry logic for throttling.

## Features

- Call Claude Sonnet 3.5, 3.7, and 4 via AWS Bedrock using `boto3`
- Call Claude Sonnet 3.5 and 3.7 via AWS Bedrock using LangChain
- Call OpenAI GPT models (default: gpt-3.5-turbo)
- Automatic retry on throttling with exponential backoff
- Environment variable support for credentials and configuration

## Requirements

- Python 3.8+
- AWS credentials with Bedrock access
- OpenAI API key (for OpenAI calls)

Install dependencies:
```bash
pip install boto3 langchain_aws langchain-core openai python-dotenv
```

## Environment Variables

Create a `.env` file in the project root with the following content:

```
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo
```

## Usage

Run the script directly to test all LLM calls with a sample prompt:

```bash
python access_llm.py
```

You can also import and use the functions in your own code:

```python
from access_llm import call_claude_sonnet_35, call_openai_llm

response = call_claude_sonnet_35("Your prompt here")
print(response)
```

## Functions

- `call_claude_sonnet_35(prompt, ...)`  
- `call_claude_sonnet_35_langchain(prompt, ...)`
- `call_claude_sonnet_37(prompt, ...)`
- `call_claude_sonnet_37_langchain(prompt, ...)`
- `call_claude_sonnet_4(prompt, ...)`
- `call_openai_llm(prompt, ...)`

All functions accept optional AWS credentials and region parameters.

## Error Handling

- Throttling and other AWS errors are handled with retries and clear messages.
- Malformed requests and other exceptions are caught and printed.

---

**Note:**  
Make sure your AWS account has access to Bedrock and the Anthropic Claude models.  
For OpenAI, ensure your API key is valid and has quota.
