import boto3
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage
import openai
import dotenv
import os
from botocore.exceptions import ClientError, BotoCoreError
import json
import time

# Load environment variables from .env file
dotenv.load_dotenv()
# Set OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
# Set AWS credentials from environment variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
# Set AWS region from environment variable
aws_region = os.getenv("AWS_REGION", "us-east-1")
# Set AWS Bedrock model ID from environment variable
bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
# Set OpenAI model from environment variable
openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

def retry_on_throttling(func):
    """Decorator to retry function on throttling with exponential backoff."""
    def wrapper(*args, **kwargs):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except ClientError as e:
                if e.response['Error']['Code'] == 'ThrottlingException':
                    wait = 2 ** attempt
                    print(f"Throttled. Waiting {wait} seconds before retrying...")
                    time.sleep(wait)
                else:
                    print(f"ClientError: {e}")
                    break
            except BotoCoreError as e:
                print(f"BotoCoreError: {e}")
                break
            except Exception as e:
                print(f"Unexpected error: {e}")
                break
        print("Max retries exceeded.")
        return None
    return wrapper

@retry_on_throttling
def call_claude_sonnet_35(prompt, region="us-east-1", aws_access_key_id=None, aws_secret_access_key=None):
    """
    Calls the AWS Bedrock Claude Sonnet 3.5 model with the given prompt.
    Optionally accepts AWS access key and secret.
    Returns the model's response as a string.
    """
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_k": 250,
        "top_p": 1,
        "stop_sequences": ["\n\nHuman:"],
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }
        ]
    }
    request = json.dumps(body)
    response = bedrock.invoke_model(
        modelId=model_id,
        body=request
    )
    model_response = json.loads(response["body"].read())
    response_text = model_response["content"][0]["text"]
    return response_text

def call_claude_sonnet_35_langchain(prompt, region="us-east-1", aws_access_key_id=None, aws_secret_access_key=None):
    """
    Calls the AWS Bedrock Claude Sonnet 3.5 model using LangChain.
    Optionally accepts AWS access key and secret.
    Returns the model's response as a string.
    """
    try:
        model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
        chat = ChatBedrock(
            model_id=model_id,
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        response = chat([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print(f"Error calling Claude Sonnet 3.5 (LangChain): {e}")
        return None

@retry_on_throttling
def call_claude_sonnet_37(prompt, region="us-east-1", aws_access_key_id=None, aws_secret_access_key=None):
    """
    Calls the AWS Bedrock Claude Sonnet 3.7 model with the given prompt.
    Optionally accepts AWS access key and secret.
    Returns the model's response as a string.
    """
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1000,
        "temperature": 0.5,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            }
        ],
    }
    request = json.dumps(native_request)
    response = bedrock.invoke_model(modelId=model_id, body=request)
    model_response = json.loads(response["body"].read())
    response_text = model_response["content"][0]["text"]
    return response_text

def call_claude_sonnet_37_langchain(prompt, region="us-east-1", aws_access_key_id=None, aws_secret_access_key=None):
    """
    Calls the AWS Bedrock Claude Sonnet 3.7 model using LangChain.
    Optionally accepts AWS access key and secret.
    Returns the model's response as a string.
    """
    try:
        model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        chat = ChatBedrock(
            model_id=model_id,
            region_name=region,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key
        )
        response = chat([HumanMessage(content=prompt)])
        return response.content
    except Exception as e:
        print(f"Error calling Claude Sonnet 3.7 (LangChain): {e}")
        return None

@retry_on_throttling
def call_claude_sonnet_4(prompt, region="us-east-1", aws_access_key_id=None, aws_secret_access_key=None):
    """
    Calls the AWS Bedrock Claude Sonnet 4 model with the given prompt.
    Optionally accepts AWS access key and secret.
    Returns the model's response as a string.
    """
    bedrock = boto3.client(
        "bedrock-runtime",
        region_name=region,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key
    )
    model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
    native_request  = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 200,
        "top_k": 250,
        "stop_sequences": [],
        "temperature": 1,
        "top_p": 0.999,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    }
    request = json.dumps(native_request)
    response = bedrock.invoke_model(modelId=model_id, body=request)
    model_response = json.loads(response["body"].read())
    response_text = model_response["content"][0]["text"]
    return response_text

def call_openai_llm(prompt, model="gpt-3.5-turbo", api_key=None):
    """
    Calls the OpenAI LLM (default: gpt-3.5-turbo) with the given prompt.
    Optionally accepts an OpenAI API key.
    Returns the model's response as a string.
    """
    try:
        if api_key:
            openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message["content"]
    except Exception as e:
        print(f"Error calling OpenAI LLM: {e}")
        return None

def main():
    prompt = "Hello, how are you?"
    print("Claude Sonnet 3.5 (boto3):")
    print(call_claude_sonnet_35(prompt, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key))
    print("\nClaude Sonnet 3.5 (LangChain):")
    print(call_claude_sonnet_35_langchain(prompt, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key))
    print("\nClaude Sonnet 3.7 (boto3):")
    print(call_claude_sonnet_37(prompt, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key))
    print("\nClaude Sonnet 3.7 (LangChain):")
    print(call_claude_sonnet_37_langchain(prompt, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key))
    print("\nClaude Sonnet 4 (boto3):")
    print(call_claude_sonnet_4(prompt, aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key))
    #print("\nOpenAI LLM:")
    #print(call_openai_llm(prompt, api_key=openai.api_key))

if __name__ == "__main__":
    main()