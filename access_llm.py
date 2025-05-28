import os
import json
import time
import dotenv
import boto3
import openai
from botocore.exceptions import ClientError, BotoCoreError
from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

class LLMClient:
    def __init__(self):
        # --- Environment Setup ---
        dotenv.load_dotenv()
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.bedrock_model_id = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        openai.api_key = self.openai_api_key
        
        # Initialize bedrock client once
        self.bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=self.aws_region,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key
        )

    def retry_on_throttling(self, func):
        """Retry function on throttling with exponential backoff."""
        def wrapper(*args, **kwargs):
            max_retries = 7  # Increased retries
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except ClientError as e:
                    if e.response['Error']['Code'] == 'ThrottlingException':
                        wait = min(2 ** attempt, 60)  # Cap at 60 seconds
                        print(f"Throttled. Waiting {wait} seconds before retrying (attempt {attempt + 1}/{max_retries})...")
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

    def _create_bedrock_payload(self, prompt, max_tokens=512, temperature=0.5):
        """Create standardized bedrock payload."""
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt}]
                }
            ]
        }

    # --- Bedrock Claude Sonnet 3.5 (boto3) ---
    def call_claude_sonnet_35(self, prompt):
        """Call Claude Sonnet 3.5 via Bedrock (boto3)."""
        @self.retry_on_throttling
        def _call(prompt):
            try:
                model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
                body = self._create_bedrock_payload(prompt, max_tokens=512, temperature=0.7)
                request = json.dumps(body)
                response = self.bedrock_client.invoke_model(modelId=model_id, body=request)
                model_response = json.loads(response["body"].read())
                return model_response["content"][0]["text"]
            except Exception as e:
                print(f"Error calling Claude Sonnet 3.5 (boto3): {e}")
                return None
        return _call(prompt)

    # --- Bedrock Claude Sonnet 3.5 (LangChain) ---
    def call_claude_sonnet_35_langchain(self, prompt):
        """Call Claude Sonnet 3.5 via Bedrock (LangChain)."""
        try:
            model_id = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
            chat = ChatBedrock(
                model_id=model_id,
                region_name=self.aws_region,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
            response = chat.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"Error calling Claude Sonnet 3.5 (LangChain): {e}")
            return None

    # --- Bedrock Claude Sonnet 3.7 (boto3) ---
    def call_claude_sonnet_37(self, prompt):
        """Call Claude Sonnet 3.7 via Bedrock (boto3)."""
        @self.retry_on_throttling
        def _call(prompt):
            try:
                model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
                body = self._create_bedrock_payload(prompt, max_tokens=512, temperature=0.5)
                request = json.dumps(body)
                response = self.bedrock_client.invoke_model(modelId=model_id, body=request)
                model_response = json.loads(response["body"].read())
                return model_response["content"][0]["text"]
            except Exception as e:
                print(f"Error calling Claude Sonnet 3.7 (boto3): {e}")
                return None
        return _call(prompt)

    # --- Bedrock Claude Sonnet 3.7 (LangChain) ---
    def call_claude_sonnet_37_langchain(self, prompt):
        """Call Claude Sonnet 3.7 via Bedrock (LangChain)."""
        try:
            model_id = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
            chat = ChatBedrock(
                model_id=model_id,
                region_name=self.aws_region,
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key
            )
            response = chat.invoke([HumanMessage(content=prompt)])
            return response.content
        except Exception as e:
            print(f"Error calling Claude Sonnet 3.7 (LangChain): {e}")
            return None

    # --- Bedrock Claude Sonnet 4 (boto3) ---
    def call_claude_sonnet_4(self, prompt):
        """Call Claude Sonnet 4 via Bedrock (boto3)."""
        @self.retry_on_throttling
        def _call(prompt):
            try:
                model_id = "us.anthropic.claude-sonnet-4-20250514-v1:0"
                body = self._create_bedrock_payload(prompt, max_tokens=256, temperature=0.5)  # Reduced tokens
                request = json.dumps(body)
                response = self.bedrock_client.invoke_model(modelId=model_id, body=request)
                model_response = json.loads(response["body"].read())
                return model_response["content"][0]["text"]
            except Exception as e:
                print(f"Error calling Claude Sonnet 4 (boto3): {e}")
                return None
        return _call(prompt)

    # --- OpenAI LLM ---
    def call_openai_llm(self, prompt, model=None, api_key=None):
        """Call OpenAI LLM (default: gpt-3.5-turbo)."""
        try:
            if api_key:
                openai.api_key = api_key
            model = model or self.openai_model
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512  # Limit tokens
            )
            return response.choices[0].message["content"]
        except Exception as e:
            print(f"Error calling OpenAI LLM: {e}")
            return None

def main():
    client = LLMClient()
    prompt = "Hello, how are you?"
    
    # Add delays between calls to avoid throttling
    print("Claude Sonnet 3.5 (boto3):")
    print(client.call_claude_sonnet_35(prompt))
    time.sleep(2)  # Wait between calls
    
    print("\nClaude Sonnet 3.5 (LangChain):")
    print(client.call_claude_sonnet_35_langchain(prompt))
    time.sleep(2)
    
    print("\nClaude Sonnet 3.7 (boto3):")
    print(client.call_claude_sonnet_37(prompt))
    time.sleep(2)
    
    print("\nClaude Sonnet 3.7 (LangChain):")
    print(client.call_claude_sonnet_37_langchain(prompt))
    time.sleep(2)
    
    print("\nClaude Sonnet 4 (boto3):")
    print(client.call_claude_sonnet_4(prompt))
    # print("\nOpenAI LLM:")
    # print(client.call_openai_llm(prompt))

if __name__ == "__main__":
    main()