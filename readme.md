# RAG-to-Riches: PDF Processing and LLM Integration

This repository provides a comprehensive solution for PDF processing, vector database creation, and question-answering using Retrieval-Augmented Generation (RAG) with AWS Bedrock Claude models and OpenAI. It includes two main modules for different use cases.

## 🚀 Features

### LLM Access (`access_llm.py`)
- Call Claude Sonnet 3.5, 3.7, and 4 via AWS Bedrock using `boto3`
- Call Claude Sonnet 3.5 and 3.7 via AWS Bedrock using LangChain
- Call OpenAI GPT models (default: gpt-3.5-turbo)
- Automatic retry on throttling with exponential backoff
- Class-based architecture for easy integration

### PDF Processing (`datasetup.py`)
- Process PDF files and create FAISS vector databases
- Load existing vector databases
- Answer questions using RAG with AWS Bedrock Claude models
- Support for single files or entire directories
- Comprehensive error handling and logging
- Interactive Q&A interface

## 📋 Requirements

- Python 3.8+
- AWS credentials with Bedrock access
- OpenAI API key (for OpenAI calls)

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd rag-to-riches
```

2. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install boto3 langchain_aws langchain-core openai python-dotenv
pip install langchain pypdf faiss-cpu langchain-community
```

## ⚙️ Environment Variables

Create a `.env` file in the project root:

```env
# AWS Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=anthropic.claude-3-sonnet-20240229-v1:0

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-3.5-turbo
```

## 🎯 Usage

### LLM Access Module

**Direct execution:**
```bash
python access_llm.py
```

**Import and use in your code:**
```python
from access_llm import LLMClient

client = LLMClient()
response = client.call_claude_sonnet_35("Your prompt here")
print(response)
```

**Available methods:**
- `call_claude_sonnet_35(prompt)`
- `call_claude_sonnet_35_langchain(prompt)`
- `call_claude_sonnet_37(prompt)`
- `call_claude_sonnet_37_langchain(prompt)`
- `call_claude_sonnet_4(prompt)`
- `call_openai_llm(prompt, model=None, api_key=None)`

### PDF Processing Module

**Direct execution:**
```bash
python datasetup.py
```

**Import and use in your code:**
```python
from datasetup import PDFProcessor

# Initialize processor
processor = PDFProcessor()

# Process PDF and create vector database
vector_db = processor.process_pdf_to_vector_db("path/to/pdf")

# Ask questions
answer = processor.answer_question("What is this document about?", vector_db)
print(answer)
```

**Key methods:**
- `process_pdf_to_vector_db(pdf_path, chunk_size=1000, chunk_overlap=200)`
- `load_vector_db(vector_db_path="./faiss_index")`
- `answer_question(question, vector_db=None, k=4)`
- `load_documents(pdf_path)`
- `create_chunks(documents, chunk_size=1000, chunk_overlap=200)`

## 📁 Project Structure

```
rag-to-riches/
├── access_llm.py          # LLM client for AWS Bedrock and OpenAI
├── datasetup.py           # PDF processing and RAG system
├── call_llm.py            # Legacy LLM functions (standalone)
├── .env                   # Environment variables
├── .gitignore            # Git ignore file
├── readme.md             # This file
└── faiss_index/          # Vector database (created automatically)
```

## 🔧 Configuration Options

### Chunk Size Recommendations
- **General documents**: `chunk_size=1000, chunk_overlap=200`
- **Technical documents**: `chunk_size=1500, chunk_overlap=300`
- **Simple text**: `chunk_size=800, chunk_overlap=160`

### Model IDs
- **Claude 3.5 Sonnet**: `us.anthropic.claude-3-5-sonnet-20241022-v2:0`
- **Claude 3.7 Sonnet**: `us.anthropic.claude-3-7-sonnet-20250219-v1:0`
- **Claude Sonnet 4**: `us.anthropic.claude-sonnet-4-20250514-v1:0`
- **Embedding Model**: `amazon.titan-embed-text-v2`

## 🚨 Error Handling

Both modules include comprehensive error handling:
- **AWS throttling**: Automatic retry with exponential backoff
- **Missing files**: Validation and clear error messages
- **Network issues**: Graceful degradation and logging
- **Invalid credentials**: Clear error reporting
- **Model access**: Handles missing model permissions

## 📝 Examples

### Example 1: Simple PDF Q&A
```python
from datasetup import PDFProcessor

processor = PDFProcessor()
vector_db = processor.process_pdf_to_vector_db("document.pdf")
answer = processor.answer_question("Summarize the main points", vector_db)
print(answer)
```

### Example 2: Directory Processing
```python
from datasetup import PDFProcessor

processor = PDFProcessor()
# Process all PDFs in a directory
vector_db = processor.process_pdf_to_vector_db("./documents/")
answer = processor.answer_question("What are the key findings?", vector_db)
print(answer)
```

### Example 3: Multiple LLM Calls
```python
from access_llm import LLMClient

client = LLMClient()
prompt = "Explain quantum computing"

# Try different models
claude_35 = client.call_claude_sonnet_35(prompt)
claude_37 = client.call_claude_sonnet_37(prompt)
openai_gpt = client.call_openai_llm(prompt)
```

## 🔍 Troubleshooting

### Common Issues

1. **AWS Access Denied**: Ensure your AWS account has Bedrock model access
2. **Throttling Errors**: The system automatically retries, but you may need to reduce request frequency
3. **Missing Dependencies**: Run `pip install -r requirements.txt` if available
4. **Vector Database Not Found**: Ensure the path exists or let the system create it automatically

### Getting Model Access

1. Go to AWS Bedrock Console
2. Navigate to "Model access"
3. Request access to Anthropic Claude models
4. Wait for approval (usually instant)

## 📞 Support

For issues related to:
- **AWS Bedrock**: Check AWS documentation or contact AWS Support
- **OpenAI API**: Verify your API key and quota
- **Code issues**: Check logs for detailed error messages

## 📄 License

This project is provided as-is for educational and development purposes.

---

**Note**: Make sure your AWS account has access to Bedrock and the Anthropic Claude models. For OpenAI, ensure your API key is valid and has sufficient quota.
