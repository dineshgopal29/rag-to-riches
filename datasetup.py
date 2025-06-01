"""
PDF Processing and RAG System with AWS Bedrock

This module provides functionality to:
1. Process PDF files and create FAISS vector databases
2. Load existing vector databases
3. Answer questions using RAG with AWS Bedrock Claude models

Required dependencies:
pip install langchain pypdf faiss-cpu boto3 langchain-community python-dotenv
"""

# Standard library imports
import os
import json
import logging
from typing import List, Optional, Union
from pathlib import Path

# Third-party imports
import boto3
import dotenv
from botocore.exceptions import ClientError, BotoCoreError
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
dotenv.load_dotenv()


class PDFProcessor:
    """Main class for PDF processing and RAG operations"""
    
    def __init__(self, 
                 region: str = None,
                 embedding_model_id: str = "amazon.titan-embed-text-v2:0",
                 claude_model_id: str = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"):
        """
        Initialize PDF Processor
        
        Args:
            region: AWS region (defaults to environment variable or us-east-1)
            embedding_model_id: Bedrock embedding model ID
            claude_model_id: Bedrock Claude model ID
        """
        self.region = region or os.getenv("AWS_REGION", "us-east-1")
        self.embedding_model_id = embedding_model_id
        self.claude_model_id = claude_model_id
        
        # Initialize AWS clients
        self._init_aws_clients()
        
        # Initialize embeddings
        self.embeddings = self._get_bedrock_embeddings()

    def _init_aws_clients(self):
        """Initialize AWS Bedrock clients with error handling"""
        try:
            # Get AWS credentials from environment
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            
            client_kwargs = {
                "service_name": "bedrock-runtime",
                "region_name": self.region
            }
            
            # Add credentials if available
            if aws_access_key_id and aws_secret_access_key:
                client_kwargs.update({
                    "aws_access_key_id": aws_access_key_id,
                    "aws_secret_access_key": aws_secret_access_key
                })
            
            self.bedrock_runtime = boto3.client(**client_kwargs)
            logger.info(f"Initialized Bedrock client for region: {self.region}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
            raise

    def _get_bedrock_embeddings(self) -> Optional[BedrockEmbeddings]:
        """Initialize and return Bedrock embeddings with error handling"""
        try:
            # Get AWS credentials
            aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            
            embeddings_kwargs = {
                "model_id": self.embedding_model_id,
                "region_name": self.region
            }
            
            # Add credentials if available
            if aws_access_key_id and aws_secret_access_key:
                embeddings_kwargs.update({
                    "credentials_profile_name": None  # Use explicit credentials
                })
            
            embeddings = BedrockEmbeddings(**embeddings_kwargs)
            logger.info(f"Initialized Bedrock embeddings with model: {self.embedding_model_id}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock embeddings: {e}")
            return None

    def validate_pdf_path(self, pdf_path: Union[str, Path]) -> bool:
        """
        Validate PDF path exists and is accessible
        
        Args:
            pdf_path: Path to PDF file or directory
            
        Returns:
            bool: True if path is valid
        """
        try:
            path = Path(pdf_path)
            
            if not path.exists():
                logger.error(f"Path does not exist: {pdf_path}")
                return False
                
            if path.is_file() and not path.suffix.lower() == '.pdf':
                logger.error(f"File is not a PDF: {pdf_path}")
                return False
                
            if path.is_dir():
                pdf_files = list(path.glob("**/*.pdf"))
                if not pdf_files:
                    logger.error(f"No PDF files found in directory: {pdf_path}")
                    return False
                logger.info(f"Found {len(pdf_files)} PDF files in directory")
                
            return True
            
        except Exception as e:
            logger.error(f"Error validating PDF path {pdf_path}: {e}")
            return False

    def load_documents(self, pdf_path: Union[str, Path]) -> List:
        """
        Load PDF documents with comprehensive error handling
        
        Args:
            pdf_path: Path to PDF file or directory
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        try:
            if not self.validate_pdf_path(pdf_path):
                return documents
                
            path = Path(pdf_path)
            
            if path.is_dir():
                # Load all PDFs from directory
                loader = DirectoryLoader(
                    str(path), 
                    glob="**/*.pdf", 
                    loader_cls=PyPDFLoader,
                    show_progress=True
                )
                documents = loader.load()
                logger.info(f"Loaded {len(documents)} documents from directory {pdf_path}")
                
            elif path.is_file():
                # Load single PDF file
                loader = PyPDFLoader(str(path))
                documents = loader.load()
                logger.info(f"Loaded document: {pdf_path}")
                
            if not documents:
                logger.warning("No documents were loaded")
                
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents from {pdf_path}: {e}")
            return []

    def create_chunks(self, documents: List, 
                     chunk_size: int = 1000, 
                     chunk_overlap: int = 200) -> List:
        """
        Split documents into chunks with error handling
        
        Args:
            documents: List of documents to chunk
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of document chunks
        """
        try:
            if not documents:
                logger.warning("No documents provided for chunking")
                return []
                
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating chunks: {e}")
            return []

    def process_pdf_to_vector_db(self, 
                                pdf_path: Union[str, Path], 
                                chunk_size: int = 1000, 
                                chunk_overlap: int = 200, 
                                vector_db_path: str = "./faiss_index") -> Optional[FAISS]:
        """
        Process PDF files and create FAISS vector database with comprehensive error handling
        
        Args:
            pdf_path: Path to PDF file or directory
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            vector_db_path: Path to save vector database
            
        Returns:
            FAISS vector store or None if failed
        """
        try:
            # Check if vector database already exists
            if os.path.exists(vector_db_path) and os.path.isdir(vector_db_path):
                logger.info(f"Vector database already exists at {vector_db_path}. Loading existing database...")
                return self.load_vector_db(vector_db_path)
            
            # Validate embeddings
            if self.embeddings is None:
                logger.error("Bedrock embeddings not initialized")
                return None
            
            # Load documents
            documents = self.load_documents(pdf_path)
            if not documents:
                logger.error("Failed to load any documents")
                return None
            
            # Create chunks
            chunks = self.create_chunks(documents, chunk_size, chunk_overlap)
            if not chunks:
                logger.error("Failed to create document chunks")
                return None
            
            # Create vector store
            logger.info("Creating FAISS vector store...")
            vector_store = FAISS.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
            
            # Save vector store
            os.makedirs(os.path.dirname(vector_db_path) if os.path.dirname(vector_db_path) else '.', exist_ok=True)
            vector_store.save_local(vector_db_path)
            logger.info(f"FAISS vector database created and saved to {vector_db_path}")
            
            return vector_store
            
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            logger.error("Please install: pip install langchain pypdf faiss-cpu boto3 langchain-community")
            return None
        except Exception as e:
            logger.error(f"Error processing PDF to vector DB: {e}")
            return None

    def load_vector_db(self, vector_db_path: str = "./faiss_index") -> Optional[FAISS]:
        """
        Load existing FAISS vector database with error handling
        
        Args:
            vector_db_path: Path to vector database
            
        Returns:
            FAISS vector store or None if failed
        """
        try:
            if not os.path.exists(vector_db_path):
                logger.error(f"Vector database not found at {vector_db_path}")
                return None
            
            if self.embeddings is None:
                logger.error("Bedrock embeddings not initialized")
                return None
            
            vector_store = FAISS.load_local(vector_db_path, self.embeddings, allow_dangerous_deserialization=True)
            logger.info(f"Loaded FAISS vector database from {vector_db_path}")
            return vector_store
            
        except Exception as e:
            logger.error(f"Error loading vector database: {e}")
            return None

    def call_bedrock_claude(self, prompt: str, max_tokens: int = 1000) -> str:
        """
        Call Amazon Bedrock Claude model with retry logic and error handling
        
        Args:
            prompt: Input prompt for Claude
            max_tokens: Maximum tokens in response
            
        Returns:
            Claude's response or error message
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                request_body = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": max_tokens,
                    "temperature": 0.7,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                }
                
                response = self.bedrock_runtime.invoke_model(
                    modelId=self.claude_model_id,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response.get("body").read())
                return response_body["content"][0]["text"]
                
            except ClientError as e:
                error_code = e.response['Error']['Code']
                if error_code == 'ThrottlingException' and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Throttled, retrying in {wait_time} seconds...")
                    import time
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"AWS ClientError: {e}")
                    return f"AWS Error: {error_code}"
                    
            except BotoCoreError as e:
                logger.error(f"AWS BotoCoreError: {e}")
                return f"AWS Connection Error: {str(e)}"
                
            except Exception as e:
                logger.error(f"Error calling Bedrock Claude: {e}")
                return f"Error: {str(e)}"
        
        return "Failed to get response after retries"

    def answer_question(self, 
                       question: str, 
                       vector_db: Optional[FAISS] = None, 
                       vector_db_path: str = "./faiss_index",
                       k: int = 4) -> str:
        """
        Answer question using RAG with comprehensive error handling
        
        Args:
            question: Question to answer
            vector_db: Existing vector database (optional)
            vector_db_path: Path to vector database
            k: Number of similar documents to retrieve
            
        Returns:
            Answer to the question
        """
        try:
            if not question or not question.strip():
                return "Please provide a valid question."
            
            # Load vector database if not provided
            if vector_db is None:
                vector_db = self.load_vector_db(vector_db_path)
                if vector_db is None:
                    return "Failed to load vector database. Please ensure the database exists and is accessible."
            
            # Search for relevant documents
            logger.info(f"Searching for relevant documents for question: {question[:100]}...")
            docs = vector_db.similarity_search(question, k=k)
            
            if not docs:
                return "No relevant documents found for your question."
            
            # Prepare context
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt
            prompt = f"""You are a helpful assistant that answers questions based on the provided context from PDF documents.

Context:
{context}

Question: {question}

Instructions:
- Answer the question based only on the provided context
- If the context doesn't contain enough information, say "I don't have enough information to answer this question based on the provided documents"
- Be concise but comprehensive in your answer
- Cite relevant parts of the context when possible

Answer:"""
            
            # Get answer from Claude
            logger.info("Getting answer from Bedrock Claude...")
            answer = self.call_bedrock_claude(prompt)
            
            return answer
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error processing your question: {str(e)}"


def main():
    """Example usage with error handling"""
    try:
        # Initialize processor
        processor = PDFProcessor()
        
        # Configuration
        pdf_path = input("Enter path to PDF file or directory: ").strip()
        if not pdf_path:
            pdf_path = "data/Goolge-whitepaper_Prompt Engineering.pdf"  # Default for testing
            
        vector_db_path = "./faiss_index"
        
        # Process PDF and create vector database
        logger.info("Processing PDF and creating/loading vector database...")
        vector_db = processor.process_pdf_to_vector_db(pdf_path, vector_db_path=vector_db_path)
        
        if vector_db:
            logger.info("Vector database ready!")
            
            # Interactive Q&A loop
            while True:
                question = input("\nEnter your question (or 'quit' to exit): ").strip()
                
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not question:
                    continue
                    
                print("\nAnswering...")
                answer = processor.answer_question(question, vector_db=vector_db)
                print(f"\nQ: {question}")
                print(f"A: {answer}")
        else:
            logger.error("Failed to create or load vector database")
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")


if __name__ == "__main__":
    main()

