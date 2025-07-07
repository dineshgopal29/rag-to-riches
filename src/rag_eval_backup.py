"""
RAGAS Evaluation Script for Geography Q&A Dataset
=================================================

This script generates a geography-focused Q&A dataset and evaluates it using RAGAS metrics.
It supports AWS Bedrock models and provides comprehensive evaluation results with visualizations.

Author: Generated for RAG evaluation purposes
Date: 2025
"""

# Standard library imports
import os
import json
import traceback

# Third-party imports
import boto3
from dotenv import load_dotenv
from botocore.client import Config
from botocore.exceptions import ClientError

# Langchain imports
from langchain_aws import ChatBedrockConverse, BedrockEmbeddings

# RAGAS imports
from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    LLMContextRecall, 
    Faithfulness, 
    FactualCorrectness,
    NoiseSensitivity, 
    ResponseRelevancy
)

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

# AWS configuration from environment variables
AWS_CONFIG = {
    "aws_access_key_id": os.environ.get("aws_access_key_id"),
    "aws_secret_access_key": os.environ.get("aws_secret_access_key"), 
    "aws_session_token": os.environ.get("aws_session_token"),
    "region_name": os.environ.get("REGION_NAME", "us-east-1"),
}

# Model configuration - prioritize inference profile from environment
MODEL_CONFIG = {
    "llm": os.environ.get("FM_ARN", "anthropic.claude-3-5-sonnet-20240620-v1:0"),
    "embeddings": "amazon.titan-embed-text-v2:0",
    "temperature": 0.4,
}

# Fallback models if primary model fails
FALLBACK_MODELS = [
    "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "anthropic.claude-3-sonnet-20240229-v1:0",
    "anthropic.claude-3-haiku-20240307-v1:0"
]

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
    "Which country has Amsterdam as its capital and what language is spoken?",
    "What is the capital of South Korea and what language do they speak?",
    "Tell me about Argentina - its capital and main language.",
    "What language is spoken in Greece and what is its capital city?",
    "Which country has Bangkok as its capital and what language is spoken there?",
    "What is the capital of Canada and what are the official languages?"
]

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================

def initialize_evaluator_clients():
    """
    Initialize RAGAS evaluator clients with fallback model support.
    
    Returns:
        tuple: (evaluator_llm, evaluator_embeddings) or (None, None) if failed
    """
    # Try primary model first, then fallbacks
    models_to_try = [MODEL_CONFIG["llm"]] + FALLBACK_MODELS
    
    for model_id in models_to_try:
        try:
            print(f"🔄 Initializing with model: {model_id}")
            
            # Initialize LLM
            llm = LangchainLLMWrapper(ChatBedrockConverse(
                **AWS_CONFIG,
                base_url=f"https://bedrock-runtime.{AWS_CONFIG['region_name']}.amazonaws.com",
                model=model_id,
                temperature=MODEL_CONFIG["temperature"],
            ))
            
            # Initialize embeddings
            embeddings = LangchainEmbeddingsWrapper(BedrockEmbeddings(
                **AWS_CONFIG,
                model_id=MODEL_CONFIG["embeddings"],
            ))
            
            print(f"✅ Successfully initialized with model: {model_id}")
            return llm, embeddings
            
        except Exception as e:
            print(f"❌ Failed to initialize with model {model_id}: {str(e)}")
            continue
    
    print("❌ Failed to initialize with any available model")
    return None, None

# Initialize evaluator clients
print("🚀 Initializing RAGAS evaluator clients...")
evaluator_llm, evaluator_embeddings = initialize_evaluator_clients()

# ============================================================================
# BEDROCK CLIENT CONFIGURATION
# ============================================================================

def create_bedrock_client():
    """Create and configure AWS Bedrock client."""
    config = Config(
        connect_timeout=120,
        read_timeout=120,
        retries={"max_attempts": 0},
        region_name=AWS_CONFIG["region_name"],
    )
    
    return boto3.client(
        "bedrock-agent-runtime",
        config=config,
        aws_access_key_id=AWS_CONFIG["aws_access_key_id"],
        aws_secret_access_key=AWS_CONFIG["aws_secret_access_key"],
        aws_session_token=AWS_CONFIG["aws_session_token"],
    )

# Create Bedrock client (optional - for knowledge base operations)
bedrock_client = create_bedrock_client()

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
        "Where is the Eiffel Tower located?": {
            "user_input": "Where is the Eiffel Tower located?",
            "response": "The Eiffel Tower is located in Paris, France.",
            "reference": "The Eiffel Tower is located in Paris, the capital city of France.",
            "retrieved_contexts": ["Paris is the capital of France. The Eiffel Tower is one of the most famous landmarks in Paris."]
        },
        # Additional entries...
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
        },
        "Which country has Amsterdam as its capital and what language is spoken?": {
            "user_input": "Which country has Amsterdam as its capital and what language is spoken?",
            "response": "The Netherlands has Amsterdam as its capital city, and Dutch is the official language spoken throughout the country.",
            "reference": "The Netherlands is a Western European country with Amsterdam as its capital, where Dutch is the official language, and Amsterdam is famous for its canals and cultural heritage.",
            "retrieved_contexts": ["The Netherlands is a country in Western Europe with Amsterdam as its capital city. The official language of the Netherlands is Dutch, which is closely related to German and English."]
        },
        "What is the capital of South Korea and what language do they speak?": {
            "user_input": "What is the capital of South Korea and what language do they speak?",
            "response": "South Korea's capital is Seoul, and Korean is the official language spoken throughout the country.",
            "reference": "South Korea is an East Asian country with Seoul as its capital, where Korean is the official language, and Seoul is a major global technology hub.",
            "retrieved_contexts": ["South Korea is a country in East Asia with Seoul as its capital city. The official language of South Korea is Korean, which uses the Hangul writing system."]
        },
        "Tell me about Argentina - its capital and main language.": {
            "user_input": "Tell me about Argentina - its capital and main language.",
            "response": "Argentina's capital is Buenos Aires, and Spanish is the official language spoken throughout the country.",
            "reference": "Argentina is a South American country with Buenos Aires as its capital, where Spanish is the official language, and Buenos Aires is known for its European architecture and tango culture.",
            "retrieved_contexts": ["Argentina is a large country in South America with Buenos Aires as its capital city. The official language of Argentina is Spanish, and Buenos Aires is famous for its cultural attractions."]
        },
        "What language is spoken in Greece and what is its capital city?": {
            "user_input": "What language is spoken in Greece and what is its capital city?",
            "response": "Greece's capital is Athens, and Greek is the official language spoken throughout the country.",
            "reference": "Greece is a European country with Athens as its capital, where Greek is the official language, and Athens is known for its ancient history and archaeological sites.",
            "retrieved_contexts": ["Greece is a country in southeastern Europe with Athens as its capital city. The official language of Greece is Greek, which has a long historical tradition dating back thousands of years."]
        },
        "Which country has Bangkok as its capital and what language is spoken there?": {
            "user_input": "Which country has Bangkok as its capital and what language is spoken there?",
            "response": "Thailand has Bangkok as its capital city, and Thai is the official language spoken throughout the country.",
            "reference": "Thailand is a Southeast Asian country with Bangkok as its capital, where Thai is the official language, and Bangkok is a major cultural and economic center.",
            "retrieved_contexts": ["Thailand is a country in Southeast Asia with Bangkok as its capital city. The official language of Thailand is Thai, which uses its own unique script and tonal system."]
        },
        "What is the capital of Canada and what are the official languages?": {
            "user_input": "What is the capital of Canada and what are the official languages?",
            "response": "Canada's capital is Ottawa, and the country has two official languages: English and French.",
            "reference": "Canada is a North American country with Ottawa as its capital, where both English and French are official languages, reflecting the country's bilingual heritage.",
            "retrieved_contexts": ["Canada is a country in North America with Ottawa as its capital city. Canada has two official languages: English and French, which are used in government and education across the country."]
        }
    }

# ============================================================================
# DATASET GENERATION
# ============================================================================

def generate_dataset(questions_list, for_evaluation=False):
    """
    Generate a dataset of geography knowledge in the specified format.
    
    Args:
        questions_list (list): List of geography questions
        for_evaluation (bool): If True, returns SingleTurnSample objects for RAGAS
        
    Returns:
        list: Dataset of geography knowledge samples
    """
    geography_data = get_geography_knowledge_base()
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
        "Where is the Eiffel Tower located?": {
            "user_input": "Where is the Eiffel Tower located?",
            "response": "The Eiffel Tower is located in Paris, France.",
            "reference": "The Eiffel Tower is located in Paris, the capital city of France.",
            "retrieved_contexts": ["Paris is the capital of France. The Eiffel Tower is one of the most famous landmarks in Paris."]
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
        },
        "Which country has Amsterdam as its capital and what language is spoken?": {
            "user_input": "Which country has Amsterdam as its capital and what language is spoken?",
            "response": "The Netherlands has Amsterdam as its capital city, and Dutch is the official language spoken throughout the country.",
            "reference": "The Netherlands is a Western European country with Amsterdam as its capital, where Dutch is the official language, and Amsterdam is famous for its canals and cultural heritage.",
            "retrieved_contexts": ["The Netherlands is a country in Western Europe with Amsterdam as its capital city. The official language of the Netherlands is Dutch, which is closely related to German and English."]
        },
        "What is the capital of South Korea and what language do they speak?": {
            "user_input": "What is the capital of South Korea and what language do they speak?",
            "response": "South Korea's capital is Seoul, and Korean is the official language spoken throughout the country.",
            "reference": "South Korea is an East Asian country with Seoul as its capital, where Korean is the official language, and Seoul is a major global technology hub.",
            "retrieved_contexts": ["South Korea is a country in East Asia with Seoul as its capital city. The official language of South Korea is Korean, which uses the Hangul writing system."]
        },
        "Tell me about Argentina - its capital and main language.": {
            "user_input": "Tell me about Argentina - its capital and main language.",
            "response": "Argentina's capital is Buenos Aires, and Spanish is the official language spoken throughout the country.",
            "reference": "Argentina is a South American country with Buenos Aires as its capital, where Spanish is the official language, and Buenos Aires is known for its European architecture and tango culture.",
            "retrieved_contexts": ["Argentina is a large country in South America with Buenos Aires as its capital city. The official language of Argentina is Spanish, and Buenos Aires is famous for its cultural attractions."]
        },
        "What language is spoken in Greece and what is its capital city?": {
            "user_input": "What language is spoken in Greece and what is its capital city?",
            "response": "Greece's capital is Athens, and Greek is the official language spoken throughout the country.",
            "reference": "Greece is a European country with Athens as its capital, where Greek is the official language, and Athens is known for its ancient history and archaeological sites.",
            "retrieved_contexts": ["Greece is a country in southeastern Europe with Athens as its capital city. The official language of Greece is Greek, which has a long historical tradition dating back thousands of years."]
        },
        "Which country has Bangkok as its capital and what language is spoken there?": {
            "user_input": "Which country has Bangkok as its capital and what language is spoken there?",
            "response": "Thailand has Bangkok as its capital city, and Thai is the official language spoken throughout the country.",
            "reference": "Thailand is a Southeast Asian country with Bangkok as its capital, where Thai is the official language, and Bangkok is a major cultural and economic center.",
            "retrieved_contexts": ["Thailand is a country in Southeast Asia with Bangkok as its capital city. The official language of Thailand is Thai, which uses its own unique script and tonal system."]
        },
        "What is the capital of Canada and what are the official languages?": {
            "user_input": "What is the capital of Canada and what are the official languages?",
            "response": "Canada's capital is Ottawa, and the country has two official languages: English and French.",
            "reference": "Canada is a North American country with Ottawa as its capital, where both English and French are official languages, reflecting the country's bilingual heritage.",
            "retrieved_contexts": ["Canada is a country in North America with Ottawa as its capital city. Canada has two official languages: English and French, which are used in government and education across the country."]
        }
    }
    
    dataset = []
    
    # Use all available questions or limit as specified
    questions_to_process = questions_list if len(questions_list) <= len(geography_data) else questions_list[:len(geography_data)]
    
    for question in questions_to_process:
        if question in geography_data:
            data = geography_data[question]
            
            if for_evaluation:
                # Create SingleTurnSample for RAGAS evaluation
                sample = SingleTurnSample(
                    user_input=data["user_input"],
                    response=data["response"],
                    reference=data["reference"],
                    retrieved_contexts=data["retrieved_contexts"]
                )
                dataset.append(sample)
            else:
                # Create dictionary format
                dataset.append({
                    "user_input": data["user_input"],
                    "response": data["response"],
                    "reference": data["reference"],
                    "retrieved_contexts": data["retrieved_contexts"]
                })
        else:
            # For questions not in our knowledge base, create generic entries
            generic_data = {
                "user_input": question,
                "response": "This question requires specific geographical knowledge about countries, capitals, and languages.",
                "reference": "This question seeks information about geographical facts including country capitals and official languages.",
                "retrieved_contexts": [f"This question asks about geographical information: '{question}'. The answer would include details about the country, its capital city, and the primary language spoken by its residents."]
            }
            
            if for_evaluation:
                sample = SingleTurnSample(**generic_data)
                dataset.append(sample)
            else:
                dataset.append(generic_data)
    
    return dataset

def print_dataset_sample():
    """Print top 3 sample dataset entries in the requested format"""
    dataset = generate_dataset(questions[:3])
    
    print("# Generated Geography Dataset Samples (Top 3):")
    print("=" * 60)
    
    for i, entry in enumerate(dataset, 1):
        print(f"\n# Sample {i}:")
        print(f'user_input="{entry["user_input"]}",')
        print(f'response="{entry["response"]}",')
        print(f'reference="{entry["reference"]}",')
        print(f'retrieved_contexts={entry["retrieved_contexts"]},')
        print("-" * 40)

def load_dataset_from_json(json_file_path):
    """Load a dataset from a JSON file
    
    Args:
        json_file_path: Path to the JSON file containing the dataset
        
    Returns:
        list: Dataset loaded from the JSON file
    """
    try:
        with open(json_file_path, "r") as f:
            dataset = json.load(f)
        print(f"Loaded {len(dataset)} samples from {json_file_path}")
        return dataset
    except Exception as e:
        print(f"Error loading dataset from {json_file_path}: {str(e)}")
        return []

def plot_evaluation_results(csv_file_path):
    """Plot evaluation results from a CSV file
    
    Args:
        csv_file_path: Path to the CSV file containing evaluation results
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Load the CSV file
        df = pd.read_csv(csv_file_path)
        
        # Extract the metric columns - updated for comprehensive metrics
        metric_columns = [
            'llm_context_recall', 
            'faithfulness', 
            'factual_correctness', 
            'noise_sensitivity',
            'response_relevancy'
        ]
        
        # Filter to only include columns that exist in the dataframe
        available_metrics = [col for col in metric_columns if col in df.columns]
        
        if not available_metrics:
            # Fallback to any numeric columns if specific metric names don't match
            available_metrics = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] and col != 'Unnamed: 0']
        
        print(f"Available metrics for plotting: {available_metrics}")
        
        # Create a figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive RAGAS Evaluation Results', fontsize=16)
        
        # Plot 1: Bar chart of average scores
        avg_scores = df[available_metrics].mean()
        ax1 = axes[0, 0]
        avg_scores.plot(kind='bar', ax=ax1, color='skyblue', rot=45)
        ax1.set_title('Average Scores by Metric')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: Box plot of score distributions
        ax2 = axes[0, 1]
        df[available_metrics].boxplot(ax=ax2, rot=45)
        ax2.set_title('Score Distributions')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Heatmap of scores by question
        ax3 = axes[1, 0]
        # Create a new dataframe with question numbers and scores
        heatmap_data = df[available_metrics].copy()
        heatmap_data.index = [f"Q{i+1}" for i in range(len(df))]
        sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', ax=ax3, vmin=0, vmax=1, cbar_kws={'shrink': 0.8})
        ax3.set_title('Scores by Question')
        ax3.tick_params(axis='x', rotation=45)
        
        # Plot 4: Scatter plot of faithfulness vs factual correctness (if both exist)
        ax4 = axes[1, 1]
        if 'faithfulness' in available_metrics and 'factual_correctness' in available_metrics:
            ax4.scatter(df['faithfulness'], df['factual_correctness'])
            ax4.set_xlabel('Faithfulness')
            ax4.set_ylabel('Factual Correctness')
            ax4.set_title('Faithfulness vs Factual Correctness')
            ax4.set_xlim(0, 1)
            ax4.set_ylim(0, 1)
            ax4.grid(True)
            
            # Add question labels to scatter plot points
            for i, txt in enumerate(df.index):
                ax4.annotate(f"Q{i+1}", (df['faithfulness'].iloc[i], df['factual_correctness'].iloc[i]))
        else:
            # Alternative plot if faithfulness and factual_correctness are not available
            if len(available_metrics) >= 2:
                ax4.scatter(df[available_metrics[0]], df[available_metrics[1]])
                ax4.set_xlabel(available_metrics[0])
                ax4.set_ylabel(available_metrics[1])
                ax4.set_title(f'{available_metrics[0]} vs {available_metrics[1]}')
                ax4.grid(True)
            else:
                ax4.text(0.5, 0.5, 'Insufficient metrics for scatter plot', ha='center', va='center', transform=ax4.transAxes)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        # Save the figure
        plt.savefig('evaluation_results_plots.png')
        print("Plots saved to evaluation_results_plots.png")
        
        # Show the figure
        plt.show()
        
    except Exception as e:
        print(f"Error plotting evaluation results: {str(e)}")
        import traceback
        traceback.print_exc()

def print_results_table(result_df, questions_subset):
    """Print evaluation results in a nice tabular format"""
    try:
        import pandas as pd
        
        # Create a copy of the dataframe for display
        display_df = result_df.copy()
        
        # Add question column
        display_df['Question'] = [questions_subset[i] if i < len(questions_subset) else f"Question {i+1}" for i in range(len(display_df))]
        
        # Reorder columns to put Question first
        cols = ['Question'] + [col for col in display_df.columns if col != 'Question' and col != 'Unnamed: 0']
        display_df = display_df[cols]
        
        # Round numeric columns to 3 decimal places
        numeric_columns = display_df.select_dtypes(include=[float, int]).columns
        display_df[numeric_columns] = display_df[numeric_columns].round(3)
        
        print("\n" + "="*120)
        print("EVALUATION RESULTS TABLE")
        print("="*120)
        
        # Print the table
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        print(display_df.to_string(index=False))
        
        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        
        summary_stats = display_df[numeric_columns].describe()
        print(summary_stats.round(3))
        
        # Print average scores
        print("\n" + "="*40)
        print("AVERAGE SCORES BY METRIC")
        print("="*40)
        
        avg_scores = display_df[numeric_columns].mean()
        for metric, score in avg_scores.items():
            print(f"{metric:30s}: {score:.3f}")
            
    except Exception as e:
        print(f"Error creating results table: {str(e)}")
        # Fallback to simple print
        print("Results DataFrame:")
        print(result_df)

# ...existing code...
if __name__ == "__main__":
    # Print sample dataset in the requested format
    print("Generating Geography Dataset...")
    print_dataset_sample()
    
    # Generate full dataset
    print(f"\n\nGenerating complete dataset with {len(questions)} questions...")
    full_dataset = generate_dataset(questions)
    print(f"Generated {len(full_dataset)} samples")
    
    # Save dataset to file
    with open("geography_dataset.json", "w") as f:
        json.dump(full_dataset, f, indent=2)
    print("Dataset saved to geography_dataset.json")
    
    # Run RAGAS evaluation (simplified to avoid async issues)
    print("\n" + "="*50)
    print("RAGAS EVALUATION:")
    print("="*50)
    try:
        # Check if evaluator clients are initialized
        if evaluator_llm is None or evaluator_embeddings is None:
            print("Error: Evaluator clients not properly initialized. Skipping evaluation.")
            exit(1)
            
        # Generate dataset for RAGAS evaluation - using only 5 samples for testing
        evaluation_samples = generate_dataset(questions[:5], for_evaluation=True)
        evaluation_dataset = EvaluationDataset(samples=evaluation_samples)
        
        # Use comprehensive metrics for thorough evaluation
        print(f"Running evaluation with comprehensive metrics on {len(evaluation_samples)} samples...")
        
        # Start with basic metrics that are more likely to work
        basic_metrics = [
            Faithfulness(llm=evaluator_llm),
            FactualCorrectness(llm=evaluator_llm),
        ]
        
        # Add context-based metrics
        context_metrics = [
            LLMContextRecall(llm=evaluator_llm),
        ]
        
        # Add advanced metrics
        advanced_metrics = [
            NoiseSensitivity(llm=evaluator_llm),
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
        ]
        
        # Try with all metrics first, fall back to basic metrics if needed
        try:
            all_metrics = basic_metrics + context_metrics + advanced_metrics
            result = evaluate(
                dataset=evaluation_dataset, 
                metrics=all_metrics,
                llm=evaluator_llm
            )
        except Exception as e:
            print(f"Error with comprehensive metrics: {str(e)}")
            print("Falling back to basic metrics...")
            result = evaluate(
                dataset=evaluation_dataset, 
                metrics=basic_metrics,
                llm=evaluator_llm
            )
        
        print(f'Metric Results:\n{result.to_pandas()}')
        
        # Print results in tabular format
        print_results_table(result.to_pandas(), questions[:5])
        
        # Export results to CSV
        csv_filename = "geography_evaluation_results.csv"
        result_df = result.to_pandas()
        
        # Add question column to CSV
        result_df['Question'] = [questions[i] if i < len(questions) else f"Question {i+1}" for i in range(len(result_df))]
        
        # Reorder columns for CSV
        cols = ['Question'] + [col for col in result_df.columns if col != 'Question']
        result_df = result_df[cols]
        
        result_df.to_csv(csv_filename, index=False)
        print(f"Evaluation results exported to {csv_filename}")
        
        # Show CSV preview
        print(f"\nCSV Preview (first 3 rows):")
        print(result_df.head(3).to_string(index=False))
        
        # Print results in tabular format
        print("\n" + "="*50)
        print("TABULAR EVALUATION RESULTS:")
        print("="*50)
        print_results_table(result.to_pandas(), questions[:5])
        
        # Plot the evaluation results (optional - requires matplotlib)
        try:
            plot_evaluation_results(csv_filename)
        except ImportError:
            print("Matplotlib/seaborn not installed. Skipping plots.")
        except Exception as e:
            print(f"Error plotting results: {str(e)}")
            
    except Exception as e:
        print(f"Error in RAGAS evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
