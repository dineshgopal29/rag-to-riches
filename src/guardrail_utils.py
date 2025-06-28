"""
AWS Bedrock Guardrail Management System

This module provides a comprehensive solution for managing AWS Bedrock Guardrails,
including creation, versioning, updating, listing, and deletion operations.

"""

import boto3
import os
import json
import logging
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from botocore.exceptions import ClientError, BotoCoreError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('guardrail_operations.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class GuardrailManager:
    """
    A class to manage AWS Bedrock Guardrails with comprehensive error handling
    and optimized operations.
    """

    def __init__(self, client: boto3.client):
        """Initialize GuardrailManager with boto3 client

        Args:
            client: AWS Bedrock boto3 client
        """
        self.client = client
        logger.info("GuardrailManager initialized successfully")

    def create_guardrail(self) -> Dict:
        """Create a new guardrail with comprehensive error handling

        Returns:
            Dict: Guardrail creation response with ID and ARN
        """
        try:
            logger.info("Creating guardrail...")
            create_response = self.client.create_guardrail(
                name="ai-social-journal-context-validator",
                description="Prevents our model from providing proprietary information.",
                topicPolicyConfig={
                    "topicsConfig": [
                        {
                            "name": "AI Social Journal Context Denied Topics",
                            "definition": "Providing personalized advice or recommendations on managing financial assets, investments, or trusts in a fiduciary capacity or assuming related obligations and liabilities.",
                            "examples": [
                                "What stocks should I invest in for my retirement?",
                                "Is it a good idea to put my money in a mutual fund?",
                                "How should I allocate my 401(k) investments?",
                                "What type of trust fund should I set up for my children?",
                                "Should I hire a financial advisor to manage my investments?",
                            ],
                            "type": "DENY",
                        }
                    ]
                },
                contentPolicyConfig={
                    "filtersConfig": [
                        {
                            "type": "SEXUAL",
                            "inputStrength": "HIGH",
                            "outputStrength": "HIGH",
                        },
                        {
                            "type": "VIOLENCE",
                            "inputStrength": "HIGH",
                            "outputStrength": "HIGH",
                        },
                        {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                        {
                            "type": "INSULTS",
                            "inputStrength": "HIGH",
                            "outputStrength": "HIGH",
                        },
                        {
                            "type": "MISCONDUCT",
                            "inputStrength": "HIGH",
                            "outputStrength": "HIGH",
                        },
                        {
                            "type": "PROMPT_ATTACK",
                            "inputStrength": "HIGH",
                            "outputStrength": "NONE",
                        },
                    ]
                },
                wordPolicyConfig={
                    "wordsConfig": [
                        {"text": "financial planning guidance"},
                        {"text": "portfolio allocation advice"},
                        {"text": "retirement fund suggestions"},
                        {"text": "wealth management tips"},
                        {"text": "trust fund setup"},
                        {"text": "investment strategy"},
                        {"text": "financial advisor recommendations"},
                    ],
                    "managedWordListsConfig": [{"type": "PROFANITY"}],
                },
                sensitiveInformationPolicyConfig={
                    "piiEntitiesConfig": [
                        {"type": "EMAIL", "action": "ANONYMIZE"},
                        {"type": "PHONE", "action": "ANONYMIZE"},
                        {"type": "NAME", "action": "ANONYMIZE"},
                        {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "BLOCK"},
                        {"type": "US_BANK_ACCOUNT_NUMBER", "action": "BLOCK"},
                        {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
                    ],
                    "regexesConfig": [
                        {
                            "name": "Account Number",
                            "description": "Matches account numbers in the format XXXXXX1234",
                            "pattern": r"\b\d{6}\d{4}\b",
                            "action": "ANONYMIZE",
                        }
                    ],
                },
                contextualGroundingPolicyConfig={
                    "filtersConfig": [
                        {"type": "GROUNDING", "threshold": 0.75},
                        {"type": "RELEVANCE", "threshold": 0.75},
                    ]
                },
                blockedInputMessaging="""Sorry, I can't assist with that.""",
                blockedOutputsMessaging="""Sorry, I can't assist with that.""",
                tags=[
                    {"key": "purpose", "value": "ai-social-journal-context-evaluator"},
                    {"key": "environment", "value": "production"},
                ],
            )

            logger.info("Guardrail created successfully")
            logger.info(f"Guardrail ID: {create_response.get('guardrailId')}")
            print(json.dumps(create_response, indent=2, default=str))
            return create_response

        except ClientError as e:
            logger.error(f"AWS ClientError creating guardrail: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating guardrail: {e}")
            raise

    def get_guardrail(self, guardrail_id: str, version_number: str = "DRAFT") -> Dict:
        """Get guardrail information with error handling

        Args:
            guardrail_id: The guardrail identifier
            version_number: The version number (default: "DRAFT")

        Returns:
            Dict: Guardrail information
        """
        try:
            logger.info(f"Retrieving guardrail {guardrail_id}, version {version_number}")
            get_response = self.client.get_guardrail(
                guardrailIdentifier=guardrail_id, 
                guardrailVersion=version_number
            )
            logger.info(f"Successfully retrieved guardrail {guardrail_id}")
            return get_response

        except ClientError as e:
            logger.error(f"AWS ClientError getting guardrail: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error getting guardrail: {e}")
            raise

    def create_guardrail_version(self, guardrail_response: Dict, version_description: str) -> Dict:
        """Create a new version of the guardrail

        Args:
            guardrail_response: Response from create or get guardrail
            version_description: Version description

        Returns:
            Dict: Version creation response
        """
        try:
            logger.info(f"Creating guardrail version: {version_description}")
            version_response = self.client.create_guardrail_version(
                guardrailIdentifier=guardrail_response["guardrailId"],
                description=version_description,
            )
            logger.info(f"Successfully created guardrail version: {version_description}")
            print(json.dumps(version_response, indent=2, default=str))
            return version_response

        except ClientError as e:
            logger.error(f"AWS ClientError creating guardrail version: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error creating guardrail version: {e}")
            raise

    def list_guardrails(self, guardrail_arn: Optional[str] = None, max_results: int = 10) -> Dict:
        """List all guardrails or versions of a specific guardrail

        Args:
            guardrail_arn: Optional ARN to list versions of specific guardrail
            max_results: Maximum number of results to return

        Returns:
            Dict: List of guardrails
        """
        try:
            logger.info("Listing guardrails...")
            
            if guardrail_arn:
                list_response = self.client.list_guardrails(
                    guardrailIdentifier=guardrail_arn,
                    maxResults=max_results
                )
            else:
                list_response = self.client.list_guardrails(
                    maxResults=max_results
                )
                
            logger.info("Successfully listed guardrails")
            print(json.dumps(list_response, indent=2, default=str))
            return list_response

        except ClientError as e:
            logger.error(f"AWS ClientError listing guardrails: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error listing guardrails: {e}")
            raise

    def update_guardrail(self, guardrail_arn: str) -> Dict:
        """Update the guardrail configuration

        Args:
            guardrail_arn: The guardrail ARN to update

        Returns:
            Dict: Update response
        """
        try:
            logger.info(f"Updating guardrail: {guardrail_arn}")
            update_response = self.client.update_guardrail(
                guardrailIdentifier=guardrail_arn,
                name="ai-social-journal-context-validator",
                description="Serves as a context validator",
                contentPolicyConfig={
                    "filtersConfig": [
                        {
                            "type": "SEXUAL",
                            "inputStrength": "HIGH",
                            "outputStrength": "HIGH",
                        },
                        {
                            "type": "VIOLENCE",
                            "inputStrength": "HIGH",
                            "outputStrength": "HIGH",
                        },
                        {"type": "HATE", "inputStrength": "HIGH", "outputStrength": "HIGH"},
                        {
                            "type": "INSULTS",
                            "inputStrength": "HIGH",
                            "outputStrength": "HIGH",
                        },
                        {
                            "type": "MISCONDUCT",
                            "inputStrength": "HIGH",
                            "outputStrength": "HIGH",
                        },
                        {
                            "type": "PROMPT_ATTACK",
                            "inputStrength": "HIGH",
                            "outputStrength": "NONE",
                        },
                    ]
                },
                wordPolicyConfig={
                    "wordsConfig": [
                        {"text": "financial planning guidance"},
                        {"text": "portfolio allocation advice"},
                        {"text": "retirement fund suggestions"},
                        {"text": "wealth management tips"},
                        {"text": "trust fund setup"},
                        {"text": "investment strategy"},
                        {"text": "financial advisor recommendations"},
                    ],
                    "managedWordListsConfig": [{"type": "PROFANITY"}],
                },
                sensitiveInformationPolicyConfig={
                    "piiEntitiesConfig": [
                        {"type": "EMAIL", "action": "ANONYMIZE"},
                        {"type": "PHONE", "action": "ANONYMIZE"},
                        {"type": "NAME", "action": "ANONYMIZE"},
                        {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "BLOCK"},
                        {"type": "US_BANK_ACCOUNT_NUMBER", "action": "BLOCK"},
                        {"type": "CREDIT_DEBIT_CARD_NUMBER", "action": "BLOCK"},
                    ],
                    "regexesConfig": [
                        {
                            "name": "Account Number",
                            "description": "Matches account numbers in the format XXXXXX1234",
                            "pattern": r"\b\d{6}\d{4}\b",
                            "action": "ANONYMIZE",
                        }
                    ],
                },
                blockedInputMessaging="""Sorry, I can't assist with that.""",
                blockedOutputsMessaging="""Sorry, I can't assist with that.""",
            )

            logger.info("Guardrail updated successfully")
            print(json.dumps(update_response, indent=2, default=str))
            return update_response

        except ClientError as e:
            logger.error(f"AWS ClientError updating guardrail: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error updating guardrail: {e}")
            raise

    def delete_guardrail(self, guardrail_id: str) -> Dict:
        """Delete a guardrail

        Args:
            guardrail_id: The guardrail identifier to delete

        Returns:
            Dict: Delete response
        """
        try:
            logger.info(f"Deleting guardrail: {guardrail_id}")
            delete_response = self.client.delete_guardrail(
                guardrailIdentifier=guardrail_id
            )
            logger.info("Guardrail deleted successfully")
            print(json.dumps(delete_response, indent=2, default=str))
            return delete_response

        except ClientError as e:
            logger.error(f"AWS ClientError deleting guardrail: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error deleting guardrail: {e}")
            raise

    @classmethod
    def from_environment(cls, region_name: Optional[str] = None) -> 'GuardrailManager':
        """
        Create GuardrailManager using environment variables for AWS credentials.
        
        Args:
            region_name: AWS region name (optional, uses env var if not provided)
            
        Returns:
            GuardrailManager: Initialized instance
        """
        client = initialize_client_with_env(region_name)
        return cls(client)
    
    def get_guardrail_status(self, guardrail_id: str, version: str = "DRAFT") -> str:
        """
        Get the current status of a guardrail.
        
        Args:
            guardrail_id: The guardrail identifier
            version: Version to check
            
        Returns:
            str: Guardrail status
        """
        try:
            response = self.get_guardrail(guardrail_id, version)
            status = response.get('status', 'UNKNOWN')
            logger.info(f"Guardrail {guardrail_id} status: {status}")
            return status
        except Exception as e:
            logger.error(f"Error getting guardrail status: {e}")
            return "ERROR"
    
    def export_guardrail_config(self, guardrail_id: str, version: str = "DRAFT", 
                               output_file: Optional[str] = None) -> Dict:
        """
        Export guardrail configuration to a file or return as dictionary.
        
        Args:
            guardrail_id: The guardrail identifier
            version: Version to export
            output_file: Output file path (optional)
            
        Returns:
            Dict: Guardrail configuration
        """
        try:
            logger.info(f"Exporting guardrail configuration: {guardrail_id}")
            config = self.get_guardrail(guardrail_id, version)
            
            if output_file:
                with open(output_file, 'w') as f:
                    json.dump(config, f, indent=2, default=str)
                logger.info(f"Configuration exported to: {output_file}")
            
            return config
        except Exception as e:
            logger.error(f"Error exporting guardrail configuration: {e}")
            raise

def integrateGuardrailWithFM(user_input: str = None) -> Dict[str, Any]:
    """
    Integrate Guardrail with Foundation Model to test input filtering.
    
    Args:
        user_input: User's text input to test against guardrail
        
    Returns:
        Dict containing response and guardrail info
    """
    try:
        bedrock_runtime = boto3.client('bedrock-runtime', 
                                     region_name=os.getenv("REGION_NAME", "us-east-1"))
        
        # Use provided input or default test case
        test_message = user_input or "How should I invest for my retirement? I want to be able to generate $5,000 a month"
        
        payload = {
            "modelId": os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"),
            "contentType": "application/json",
            "accept": "application/json",
            "body": {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": test_message
                            }
                        ]
                    }
                ]
            }
        }

        # Convert the payload to bytes
        body_bytes = json.dumps(payload['body']).encode('utf-8')

        # Invoke the model with guardrail
        response = bedrock_runtime.invoke_model(
            body=body_bytes,
            contentType=payload['contentType'],
            accept=payload['accept'],
            modelId=payload['modelId'],
            guardrailIdentifier=os.getenv("GUARDRAIL_ID"),
            guardrailVersion=os.getenv("GUARDRAIL_VERSION", "DRAFT"),
            trace="ENABLED"
        )

        # Parse the response
        response_body = response['body'].read().decode('utf-8')
        response_data = json.loads(response_body)
        
        # Extract guardrail information from response metadata
        guardrail_info = {
            "guardrail_id": os.getenv("GUARDRAIL_ID"),
            "guardrail_version": os.getenv("GUARDRAIL_VERSION", "DRAFT"),
            "model_id": payload['modelId'],
            "user_input": test_message
        }
        
        # Check if response contains guardrail actions
        if 'trace' in response_data:
            guardrail_info['guardrail_trace'] = response_data.get('trace', {})
        
        result = {
            "success": True,
            "response": response_data,
            "guardrail_info": guardrail_info,
            "raw_response": response_body
        }
        
        logger.info(f"Successfully invoked model with guardrail for input: {test_message[:50]}...")
        return result

    except ClientError as e:
        error_result = {
            "success": False,
            "error": str(e),
            "error_code": e.response.get('Error', {}).get('Code', 'Unknown'),
            "guardrail_info": {
                "guardrail_id": os.getenv("GUARDRAIL_ID"),
                "guardrail_version": os.getenv("GUARDRAIL_VERSION", "DRAFT"),
                "model_id": os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-sonnet-20240229-v1:0"),
                "user_input": user_input or "Test input"
            }
        }
        logger.error(f"AWS ClientError invoking model with guardrail: {e}")
        return error_result
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e),
            "guardrail_info": {
                "guardrail_id": os.getenv("GUARDRAIL_ID"),
                "guardrail_version": os.getenv("GUARDRAIL_VERSION", "DRAFT"),
                "user_input": user_input or "Test input"
            }
        }
        logger.error(f"Unexpected error invoking model with guardrail: {e}")
        return error_result

def initialize_client() -> boto3.client:
    """Initialize and return AWS Bedrock client with error handling (legacy function)"""
    return initialize_client_with_env()


def initialize_client_with_env(region_name: Optional[str] = None) -> boto3.client:
    """
    Initialize AWS Bedrock client using environment variables with comprehensive error handling.
    
    Args:
        region_name: AWS region name
        
    Returns:
        boto3.client: AWS Bedrock client
    """
    try:
        region = region_name or os.getenv("REGION_NAME", "us-east-1")
        
        # Check for required environment variables
        env_vars = {
            'AWS_ACCESS_KEY_ID': os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("aws_access_key_id"),
            'AWS_SECRET_ACCESS_KEY': os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("aws_secret_access_key"),
            'AWS_SESSION_TOKEN': os.getenv("AWS_SESSION_TOKEN") or os.getenv("aws_session_token")
        }
        
        client_kwargs = {
            "service_name": "bedrock",
            "region_name": region
        }
        
        # Add credentials if available
        if env_vars['AWS_ACCESS_KEY_ID'] and env_vars['AWS_SECRET_ACCESS_KEY']:
            client_kwargs.update({
                "aws_access_key_id": env_vars['AWS_ACCESS_KEY_ID'],
                "aws_secret_access_key": env_vars['AWS_SECRET_ACCESS_KEY']
            })
            logger.info("Using explicit AWS credentials from environment")
        else:
            logger.info("Using default AWS credential chain")
            
        if env_vars['AWS_SESSION_TOKEN']:
            client_kwargs["aws_session_token"] = env_vars['AWS_SESSION_TOKEN']
            logger.info("Session token found and applied")
            
        client = boto3.client(**client_kwargs)
        
        # Test the client with a simple API call
        try:
            client.list_guardrails(maxResults=1)
            logger.info(f"AWS Bedrock client successfully initialized for region: {region}")
        except Exception as test_error:
            logger.warning(f"Client created but test call failed: {test_error}")
            
        return client
        
    except Exception as e:
        logger.error(f"Failed to initialize AWS client: {e}")
        raise


def main():
    """Main function demonstrating guardrail operations"""
    try:
        print("=== AWS Bedrock Guardrail Management Demo ===\n")
        
        # Method 1: Initialize with environment variables
        print("Initializing GuardrailManager...")
        guardrail_manager = GuardrailManager.from_environment()
        
        # List existing guardrails first
        print("\n=== Listing Existing Guardrails ===")
        existing_guardrails = guardrail_manager.list_guardrails()
        print(f"Found {len(existing_guardrails.get('guardrails', []))} existing guardrails")
        
        # Example workflow (commented out to avoid creating actual resources)

        # Create a new guardrail
        #print("\n=== Creating New Guardrail ===")
        #guardrail_response = guardrail_manager.create_guardrail()
        #guardrail_id = guardrail_response["guardrailId"]
        #guardrail_arn = guardrail_response["guardrailArn"]
        #print(f"Created Guardrail ID: {guardrail_id}")
        
        # Get guardrail information
        #print("\n=== Getting Guardrail Information ===")
        #get_response = guardrail_manager.get_guardrail(guardrail_id, "DRAFT")
        #print(f"Guardrail Status: {guardrail_manager.get_guardrail_status(guardrail_id)}")
        
        # Export configuration
        #print("\n=== Exporting Guardrail Configuration ===")
        #config_file = f"guardrail_config_{guardrail_id}.json"
        #guardrail_manager.export_guardrail_config(guardrail_id, "DRAFT", config_file)
        
        # Create a version
        #print("\n=== Creating Guardrail Version ===")
        #version_response = guardrail_manager.create_guardrail_version(
        #    guardrail_response, "Production Version 1.0"
        #)
        
        # Update guardrail
        #print("\n=== Updating Guardrail ===")
        #update_response = guardrail_manager.update_guardrail(guardrail_arn)
        
        # List all versions
        #print("\n=== Listing Guardrail Versions ===")
        #versions = guardrail_manager.list_guardrails(guardrail_arn)
        
        # Cleanup (uncomment if you want to delete the test guardrail)
        # print("\n=== Deleting Test Guardrail ===")
        # delete_response = guardrail_manager.delete_guardrail(guardrail_id)
    
        
        #print("\n=== Demo Complete ===")
        #print("Uncomment the example workflow above to test actual guardrail operations.")
        #logger.info("Demo completed successfully")
        
        # Test guardrail integration
        print("\n=== Testing Guardrail Integration with Foundation Model ===")
        test_result = integrateGuardrailWithFM()
        
        if test_result['success']:
            print("✅ Guardrail integration test successful!")
            print(f"Model response received with guardrail protection")
        else:
            print("❌ Guardrail integration test failed!")
            print(f"Error: {test_result.get('error', 'Unknown error')}")
        
        return test_result
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        print(f"Error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure AWS credentials are properly configured")
        print("2. Check that the region supports AWS Bedrock")
        print("3. Verify IAM permissions for Bedrock operations")


if __name__ == "__main__":
    main()
