"""
Simple RAGAS test to debug NaN issues
"""

import os
import json
from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import Faithfulness

load_dotenv()

# Configuration
AWS_CONFIG = {
    "aws_access_key_id": os.environ.get("aws_access_key_id"),
    "aws_secret_access_key": os.environ.get("aws_secret_access_key"), 
    "aws_session_token": os.environ.get("aws_session_token"),
    "region_name": os.environ.get("REGION_NAME", "us-east-1"),
}

MODEL_CONFIG = {
    "llm": os.environ.get("FM_ARN", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
    "embeddings": "amazon.titan-embed-text-v2:0",
    "temperature": 0.4,
}

def test_simple_evaluation():
    print("🧪 Testing simple RAGAS evaluation...")
    
    try:
        # Initialize clients
        print("🔧 Initializing clients...")
        llm = LangchainLLMWrapper(ChatBedrockConverse(
            **AWS_CONFIG,
            base_url=f"https://bedrock-runtime.{AWS_CONFIG['region_name']}.amazonaws.com",
            model=MODEL_CONFIG["llm"],
            temperature=MODEL_CONFIG["temperature"],
        ))
        
        embeddings = LangchainEmbeddingsWrapper(BedrockEmbeddings(
            **AWS_CONFIG,
            model_id=MODEL_CONFIG["embeddings"],
        ))
        
        print("✅ Clients initialized successfully")
        
        # Create simple test sample
        sample = SingleTurnSample(
            user_input="Where is the Eiffel Tower located?",
            response="The Eiffel Tower is located in Paris, France.",
            reference="The Eiffel Tower is located in Paris, the capital city of France.",
            retrieved_contexts=["Paris is the capital of France. The Eiffel Tower is one of the most famous landmarks in Paris."]
        )
        
        dataset = EvaluationDataset(samples=[sample])
        print("✅ Test dataset created with 1 sample")
        
        # Test with just Faithfulness metric
        print("🔍 Testing Faithfulness metric...")
        faithfulness_metric = Faithfulness(llm=llm)
        
        result = evaluate(
            dataset=dataset,
            metrics=[faithfulness_metric],
            llm=llm
        )
        
        print("✅ Evaluation completed!")
        
        # Check results
        df = result.to_pandas()
        print("\n📊 Results:")
        print(df)
        
        # Check for NaN
        if df.isna().any().any():
            print("⚠️  Found NaN values!")
            print("NaN columns:", df.columns[df.isna().any()].tolist())
        else:
            print("✅ No NaN values found")
            
        return True
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_simple_evaluation()
