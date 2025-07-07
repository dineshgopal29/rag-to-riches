"""
RAGAS Evaluation Script for Geography Q&A Dataset
=================================================

This script generates a geography-focused Q&A dataset and evaluates it using RAGAS metrics.
It supports AWS Bedrock models and provides evaluation results in tabular and CSV formats.

Features:
- Uses AWS Bedrock Claude 3.5 Sonnet for LLM evaluation
- Uses Amazon Titan embeddings for semantic similarity
- Evaluates geography knowledge base with 10 predefined questions
- Configurable number of questions to evaluate (default: 5)
- Outputs results in both tabular and CSV format
- Includes metrics: LLM Context Recall, Faithfulness, and Factual Correctness

Usage:
1. Set AWS credentials in environment variables or .env file
2. Run: python rag_eval.py
3. Results will be saved to geography_evaluation_results.csv

Author: RAG Evaluation System
Version: 2.0 (Cleaned and optimized)
"""

# Standard library imports
import os
import traceback

# Third-party imports
from dotenv import load_dotenv

# Langchain imports
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

# RAGAS imports
from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# AWS configuration from environment variables
aws_config = {
    "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
    "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
    "region_name": os.environ.get("REGION_NAME", "us-east-1"),
}


# Model configuration
model_config = {
    "llm": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "embeddings": "amazon.titan-embed-text-v2:0",
    "temperature": 0.4,
}

# Evaluation configuration
EVALUATION_CONFIG = {
    "num_questions": 5,  # Number of questions to evaluate (change as needed)
    "sample_display": 3,  # Number of samples to display
}

# Geography Q&A questions for evaluation
GEOGRAPHY_QUESTIONS = [
    "Where is the Eiffel Tower located?",
    "What is the capital of Japan and what language do they speak?",
    "Which country has Rome as its capital and what is the official language?",
    "What language is spoken in Brazil and what is its capital city?",
    "Tell me about Germany - its capital and main language.",
    "What is the capital of Russia and what language do Russians speak?",
    "Which language is spoken in Spain and what is its capital?",
    "What is the capital of China and what is the primary language there?",
    "Tell me about Egypt - its capital city and official language.",
    "What language do they speak in Sweden and what is its capital?",
]

# ============================================================================
# MODEL INITIALIZATION FUNCTIONS
# ============================================================================

def initialize_llm():
    """Initialize the LLM for RAGAS evaluation."""
    return LangchainLLMWrapper(ChatBedrockConverse(
        aws_access_key_id=aws_config["aws_access_key_id"],
        aws_secret_access_key=aws_config["aws_secret_access_key"],
        region_name=aws_config["region_name"],
        base_url=f"https://bedrock-runtime.{aws_config['region_name']}.amazonaws.com",
        model=model_config["llm"],
        temperature=model_config["temperature"],
    ))

def initialize_embeddings():
    """Initialize the embeddings for RAGAS evaluation."""
    return LangchainEmbeddingsWrapper(BedrockEmbeddings(
        aws_access_key_id=aws_config["aws_access_key_id"],
        aws_secret_access_key=aws_config["aws_secret_access_key"],
        region_name=aws_config["region_name"],
        model_id=model_config["embeddings"],
    ))

# ============================================================================
# GEOGRAPHY KNOWLEDGE BASE  
# ============================================================================

def get_geography_knowledge_base():
    """
    Returns the comprehensive geography knowledge base for Q&A evaluation.
    
    Returns:
        dict: Geography knowledge base with questions as keys and data as values
    """
    return {
        "Where is the Eiffel Tower located?": {
            "user_input": "Where is the Eiffel Tower located?",
            "response": "The Eiffel Tower is located in Paris, France.",
            "reference": "The Eiffel Tower is located in Paris, the capital city of France.",
            "retrieved_contexts": ["Paris is the capital of France. The Eiffel Tower is one of the most famous landmarks in Paris."]
        },
        "What is the capital of Japan and what language do they speak?": {
            "user_input": "What is the capital of Japan and what language do they speak?",
            "response": "Japan's capital is Tokyo, and the primary language spoken is Japanese, which uses multiple writing systems including hiragana, katakana, and kanji.",
            "reference": "Japan is an East Asian country with Tokyo as its capital, where Japanese is the official language using three writing systems, and Tokyo serves as the major political and economic hub.",
            "retrieved_contexts": ["Japan is an island nation in East Asia with Tokyo as its capital city. The official language of Japan is Japanese, which uses three writing systems: hiragana, katakana, and kanji."]
        },
        "Which country has Rome as its capital and what is the official language?": {
            "user_input": "Which country has Rome as its capital and what is the official language?",
            "response": "Italy has Rome as its capital city, and Italian is the official language spoken throughout the country.",
            "reference": "Italy is a Southern European country with Rome as its capital, where Italian is the official language, and Rome is known for its historical significance as the Eternal City.",
            "retrieved_contexts": ["Italy is a country in Southern Europe with Rome as its capital city. The official language of Italy is Italian, which evolved from Latin and is spoken by nearly all residents."]
        },
        "What language is spoken in Brazil and what is its capital city?": {
            "user_input": "What language is spoken in Brazil and what is its capital city?",
            "response": "Brazil's capital is Brasília, and Portuguese is the official language, making Brazil unique as the only Portuguese-speaking nation in South America.",
            "reference": "Brazil is a South American country with Brasília as its capital city, where Portuguese is the official language, and Brasília is notable for its modern planned architecture.",
            "retrieved_contexts": ["Brazil is the largest country in South America with Brasília as its capital city. The official language of Brazil is Portuguese, making it the only Portuguese-speaking country in South America."]
        },
        "Tell me about Germany - its capital and main language.": {
            "user_input": "Tell me about Germany - its capital and main language.",
            "response": "Germany's capital is Berlin, and German is the official language spoken throughout the country and in neighboring regions.",
            "reference": "Germany is a Central European country with Berlin as its capital, where German is the official language, and Berlin became the capital following reunification in 1990.",
            "retrieved_contexts": ["Germany is a country in Central Europe with Berlin as its capital city. The official language of Germany is German, which is also widely spoken in Austria and parts of Switzerland."]
        },
        "What is the capital of Russia and what language do Russians speak?": {
            "user_input": "What is the capital of Russia and what language do Russians speak?",
            "response": "Russia's capital is Moscow, and Russian is the official language, written in Cyrillic script and spoken by most of the population.",
            "reference": "Russia is the world's largest country with Moscow as its capital, where Russian is the official language using Cyrillic alphabet, and Moscow serves as the political and cultural hub.",
            "retrieved_contexts": ["Russia is the largest country in the world by land area, with Moscow as its capital city. The official language of Russia is Russian, which uses the Cyrillic alphabet."]
        },
        "Which language is spoken in Spain and what is its capital?": {
            "user_input": "Which language is spoken in Spain and what is its capital?",
            "response": "Spain's capital is Madrid, and Spanish (Castilian) is the official language spoken throughout the country.",
            "reference": "Spain is a European country with Madrid as its capital, where Spanish is the official language, and Madrid serves as the political and cultural center.",
            "retrieved_contexts": ["Spain is a country in southwestern Europe with Madrid as its capital city. The official language is Spanish, also known as Castilian."]
        },
        "What is the capital of China and what is the primary language there?": {
            "user_input": "What is the capital of China and what is the primary language there?",
            "response": "China's capital is Beijing, and Mandarin Chinese is the primary language spoken by the majority of the population.",
            "reference": "China is an East Asian country with Beijing as its capital, where Mandarin Chinese is the official language and most widely spoken dialect.",
            "retrieved_contexts": ["China is a vast country in East Asia with Beijing as its capital city. Mandarin Chinese is the official language and is spoken by over 900 million people."]
        },
        "Tell me about Egypt - its capital city and official language.": {
            "user_input": "Tell me about Egypt - its capital city and official language.",
            "response": "Egypt's capital is Cairo, and Arabic is the official language spoken throughout the country.",
            "reference": "Egypt is a country in North Africa with Cairo as its capital, where Arabic is the official language, and Cairo is one of the largest cities in the Middle East.",
            "retrieved_contexts": ["Egypt is a country in North Africa with Cairo as its capital city. The official language of Egypt is Arabic, which is used in government, education, and media."]
        },
        "What language do they speak in Sweden and what is its capital?": {
            "user_input": "What language do they speak in Sweden and what is its capital?",
            "response": "Sweden's capital is Stockholm, and Swedish is the official language spoken by the majority of the population.",
            "reference": "Sweden is a Scandinavian country with Stockholm as its capital, where Swedish is the official language, and Stockholm is known for its archipelago and Nobel Prize ceremonies.",
            "retrieved_contexts": ["Sweden is a country in Scandinavia with Stockholm as its capital city. The official language of Sweden is Swedish, which is a North Germanic language."]
        }
    }

# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_dataset(questions_list):
    """Generate a dataset of geography knowledge in dictionary format.
    
    Args:
        questions_list: List of questions to use as prompts
        
    Returns:
        list: Dataset of geography knowledge samples as dictionaries
    """
    geography_data = get_geography_knowledge_base()
    dataset = []
    
    for question in questions_list:
        if question in geography_data:
            data = geography_data[question]
            dataset.append({
                "user_input": data["user_input"],
                "retrieved_contexts": data["retrieved_contexts"],
                "response": data["response"],
                "reference": data["reference"]
            })
        else:
            # Create generic entry for unknown questions
            dataset.append({
                "user_input": question,
                "retrieved_contexts": [f"This question asks about geographical information: '{question}'."],
                "response": "This question requires specific geographical knowledge about countries, capitals, and languages.",
                "reference": "This question seeks information about geographical facts including country capitals and official languages."
            })
    
    return dataset

def print_dataset_sample(num_samples=3):
    """
    Print sample dataset entries in the requested format.
    
    Args:
        num_samples (int): Number of samples to display
    """
    dataset = generate_dataset(GEOGRAPHY_QUESTIONS[:num_samples])
    
    print(f"# Generated Geography Dataset Samples (Top {num_samples}):")
    print("=" * 60)
    
    for i, entry in enumerate(dataset, 1):
        print(f"\n# Sample {i}:")
        print(f'user_input="{entry["user_input"]}",')
        print(f'response="{entry["response"]}",')
        print(f'reference="{entry["reference"]}",')
        print(f'retrieved_contexts={entry["retrieved_contexts"]},')
        print("-" * 40)

# ============================================================================
# EVALUATION UTILITIES
# ============================================================================

def print_results_table(result_df, questions_subset):
    """Print evaluation results in a tabular format."""
    try:
        import pandas as pd
        
        # Add question column
        result_df['Question'] = [
            questions_subset[i] if i < len(questions_subset) else f"Question {i+1}"
            for i in range(len(result_df))
        ]
        
        # Keep only Question and score columns
        score_columns = [col for col in result_df.columns 
                        if col not in ['Question', 'Unnamed: 0', 'user_input', 'retrieved_contexts', 'response', 'reference']]
        display_df = result_df[['Question'] + score_columns].round(3)
        
        print("\n" + "="*120)
        print("📊 EVALUATION RESULTS - QUESTIONS AND SCORES")
        print("="*120)
        
        # Configure pandas display
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 80)
        print(display_df.to_string(index=False))
            
    except Exception as e:
        print(f"❌ Error creating results table: {str(e)}")
        print("📄 Raw Results:")
        print(result_df)

def save_results_to_csv(result_df, questions_subset, filename="geography_evaluation_results.csv"):
    """Save evaluation results to CSV with question information."""
    try:
        # Add question column
        result_df['Question'] = [
            questions_subset[i] if i < len(questions_subset) else f"Question {i+1}" 
            for i in range(len(result_df))
        ]
        
        # Reorder columns
        cols = ['Question'] + [col for col in result_df.columns if col != 'Question']
        result_df = result_df[cols]
        
        # Save to CSV
        result_df.to_csv(filename, index=False)
        print(f"💾 Results exported to {filename}")
        
        # Show preview
        print(f"\n📋 CSV Preview:")
        print(result_df.head(3).to_string(index=False))
        
        return result_df
        
    except Exception as e:
        print(f"❌ Error saving to CSV: {str(e)}")
        return result_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("🚀 Starting RAGAS Geography Evaluation")
    print("=" * 50)
    
    # Initialize clients
    print("🔧 Initializing RAGAS clients...")
    try:
        evaluator_llm = initialize_llm()
        evaluator_embeddings = initialize_embeddings()
        print("✅ Successfully initialized evaluator clients")
    except Exception as e:
        print(f"❌ Failed to initialize clients: {str(e)}")
        return
    
    # Generate and display sample dataset
    print("\n📝 Generating Geography Dataset...")
    print_dataset_sample(EVALUATION_CONFIG["sample_display"])
    
    # Run evaluation
    num_questions = EVALUATION_CONFIG["num_questions"]
    print(f"\n" + "="*50)
    print(f"🔍 RAGAS EVALUATION ({num_questions} QUESTIONS)")
    print("="*50)
    
    try:
        # Generate dataset from configured number of questions
        questions_to_evaluate = GEOGRAPHY_QUESTIONS[:num_questions]
        dataset = generate_dataset(questions_to_evaluate)
        print(f"Generated {len(dataset)} samples")
        
        # Create evaluation dataset
        evaluation_dataset = EvaluationDataset.from_list(dataset)
        print(f"Created evaluation dataset with {len(evaluation_dataset)} samples")
        
        # Run evaluation
        print(f"⚡ Running evaluation...")
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        )
        
        print("✅ Evaluation completed!")
        
        # Process results
        result_df = result.to_pandas().fillna(0)
        
        # Display and save results
        print_results_table(result_df, questions_to_evaluate)
        save_results_to_csv(result_df, questions_to_evaluate)
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"❌ Error in RAGAS evaluation: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
