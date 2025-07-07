"""
Basic test without external API calls
"""

import os
from dotenv import load_dotenv

load_dotenv()

def test_environment():
    """Test if environment variables are loaded correctly"""
    print("🔍 Testing environment setup...")
    
    aws_key = os.environ.get("aws_access_key_id")
    aws_secret = os.environ.get("aws_secret_access_key")
    region = os.environ.get("REGION_NAME")
    fm_arn = os.environ.get("FM_ARN")
    
    print(f"AWS Access Key: {'✅ Set' if aws_key else '❌ Missing'}")
    print(f"AWS Secret Key: {'✅ Set' if aws_secret else '❌ Missing'}")
    print(f"Region: {region or '❌ Missing'}")
    print(f"FM ARN: {fm_arn or '❌ Missing'}")
    
    if not all([aws_key, aws_secret, region]):
        print("❌ Missing required AWS credentials")
        return False
    
    print("✅ Environment setup looks good")
    return True

def test_imports():
    """Test if all required packages can be imported"""
    print("\n🔍 Testing imports...")
    
    try:
        import boto3
        print("✅ boto3 imported")
    except ImportError as e:
        print(f"❌ boto3 import failed: {e}")
        return False
    
    try:
        from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
        print("✅ langchain_aws imported")
    except ImportError as e:
        print(f"❌ langchain_aws import failed: {e}")
        return False
    
    try:
        from ragas import SingleTurnSample, EvaluationDataset, evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.metrics import Faithfulness
        print("✅ ragas imported")
    except ImportError as e:
        print(f"❌ ragas import failed: {e}")
        return False
    
    print("✅ All imports successful")
    return True

def test_basic_data_structure():
    """Test basic data structures for RAGAS"""
    print("\n🔍 Testing data structures...")
    
    try:
        from ragas import SingleTurnSample
        
        # Create a simple sample
        sample = SingleTurnSample(
            user_input="Test question",
            response="Test response", 
            reference="Test reference",
            retrieved_contexts=["Test context"]
        )
        
        print("✅ SingleTurnSample created successfully")
        print(f"   User input: {sample.user_input}")
        print(f"   Response: {sample.response}")
        print(f"   Reference: {sample.reference}")
        print(f"   Contexts: {sample.retrieved_contexts}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Running basic diagnostics...")
    print("=" * 50)
    
    env_ok = test_environment()
    imports_ok = test_imports()
    data_ok = test_basic_data_structure()
    
    print("\n" + "=" * 50)
    print("📋 DIAGNOSTIC SUMMARY:")
    print(f"Environment: {'✅ OK' if env_ok else '❌ FAILED'}")
    print(f"Imports: {'✅ OK' if imports_ok else '❌ FAILED'}")
    print(f"Data Structures: {'✅ OK' if data_ok else '❌ FAILED'}")
    
    if all([env_ok, imports_ok, data_ok]):
        print("\n🎉 All basic tests passed! NaN issue is likely in model initialization or evaluation.")
    else:
        print("\n❌ Some basic tests failed. Fix these issues first.")
