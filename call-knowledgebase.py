import streamlit as st
import boto3
import os
import time
import json
from datetime import datetime
from botocore.client import Config
from botocore.exceptions import ClientError, BotoCoreError
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Social Journal Q&A Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class KnowledgeBaseClient:
    """Client for AWS Bedrock Knowledge Base operations with comprehensive error handling"""
    
    def __init__(self):
        """Initialize the Bedrock client with error handling"""
        try:
            self.bedrock_config = Config(
                connect_timeout=120,
                read_timeout=120,
                retries={"max_attempts": 3},
                region_name=os.getenv("REGION_NAME", "us-east-1"),
            )
            
            self.bedrock_agent_runtime_client = boto3.client(
                "bedrock-agent-runtime",
                config=self.bedrock_config,
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
            )
            
            # Validate required environment variables
            self._validate_env_vars()
            logger.info("Successfully initialized Bedrock client")
            
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            st.error(f"Failed to initialize AWS Bedrock client: {e}")
            st.stop()
    
    def _validate_env_vars(self):
        """Validate required environment variables"""
        required_vars = ["KB_ID", "FM_ARN", "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def get_response_from_knowledgebase(self, query: str, max_retries: int = 3) -> str:
        """
        Method to invoke AWS Bedrock Knowledge Base and get a response with retry logic
        
        Args:
            query (str): User question/query
            max_retries (int): Maximum number of retry attempts
            
        Returns:
            str: Response from the Knowledge Base or error message
        """
        if not query or not query.strip():
            return "Please provide a valid question."
        
        # Default knowledge base prompt
        default_prompt = """
        Act as a question-answering agent for the AI Social Journal Q&A Bot to help users with their questions. 
        Your role is to:
        - Provide accurate information based on the knowledge base
        - Be helpful, friendly, and professional
        - If information is not available, suggest alternative resources or clarify the question
        
        Guidelines:
        1. Answering Questions:
        - Answer the user's question strictly based on search results
        - Correct any grammatical or typographical errors in the user's question
        - If search results don't contain relevant information, state: "I could not find an exact answer to your question. Could you please provide more information or rephrase your question?"
        
        2. Validating Information:
        - Double-check search results to validate any assertions
        
        3. Clarification:
        - Ask follow-up questions to clarify if the user doesn't provide enough information
        
        Here are the search results in numbered order:
        $search_results$
        
        $output_format_instructions$
        """
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Querying knowledge base (attempt {attempt + 1}): {query[:100]}...")
                
                response = self.bedrock_agent_runtime_client.retrieve_and_generate(
                    input={"text": query},
                    retrieveAndGenerateConfiguration={
                        "type": "KNOWLEDGE_BASE",
                        "knowledgeBaseConfiguration": {
                            "knowledgeBaseId": os.getenv("KB_ID"),
                            "modelArn": os.getenv("FM_ARN"),
                            "retrievalConfiguration": {
                                "vectorSearchConfiguration": {}
                            },
                            "generationConfiguration": {
                                "promptTemplate": {"textPromptTemplate": default_prompt},
                                "inferenceConfig": {
                                    "textInferenceConfig": {
                                        "maxTokens": 2000,
                                        "temperature": 0.7,
                                        "topP": 0.9,
                                    }
                                },
                            },
                        },
                    },
                )
                
                if response and "output" in response and "text" in response["output"]:
                    logger.info("Successfully received response from knowledge base")
                    return response["output"]["text"]
                else:
                    logger.warning("Received empty or invalid response structure")
                    return "I received an incomplete response. Please try asking your question again."
                    
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', 'Unknown')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                
                if error_code == 'ThrottlingException' and attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Throttled, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"AWS ClientError: {error_code} - {error_message}")
                    return f"Sorry, I encountered an AWS error: {error_code}. Please try again later."
                    
            except BotoCoreError as e:
                logger.error(f"AWS BotoCoreError: {e}")
                return "I'm having trouble connecting to AWS services. Please check your internet connection and try again."
                
            except Exception as e:
                logger.error(f"Unexpected error querying knowledge base: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    return f"I encountered an unexpected error: {str(e)}. Please try again."
        
        return "I'm sorry, I couldn't process your request after multiple attempts. Please try again later."


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "kb_client" not in st.session_state:
        st.session_state.kb_client = KnowledgeBaseClient()

def display_chat_message(role: str, content: str):
    """Display a chat message with appropriate styling"""
    with st.chat_message(role):
        st.markdown(content)

def stream_response(response_text: str, placeholder):
    """Create a streaming effect for the response"""
    streamed_text = ""
    for char in response_text:
        streamed_text += char
        placeholder.markdown(streamed_text + "▌")
        time.sleep(0.01)  # Adjust speed as needed
    placeholder.markdown(streamed_text)
    return streamed_text

def save_conversation():
    """Save conversation to a JSON file"""
    try:
        if st.session_state.messages:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"
            
            conversation_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages
            }
            
            with open(filename, 'w') as f:
                json.dump(conversation_data, f, indent=2)
            
            st.success(f"Conversation saved as {filename}")
        else:
            st.warning("No conversation to save")
    except Exception as e:
        st.error(f"Failed to save conversation: {e}")

def clear_conversation():
    """Clear the current conversation"""
    st.session_state.messages = []
    st.rerun()

def main():
    """Main Streamlit application"""
    # Initialize session state
    initialize_session_state()
    
    # Header
    st.title("🤖 AI Social Journal Q&A Bot")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📖 About")
        st.markdown("""
        **AI Social Journal Q&A Bot** is an intelligent assistant powered by AWS Bedrock Knowledge Base.
        
        **Features:**
        - 💬 Interactive chat interface
        - 🔍 Knowledge base-powered responses
        - 💾 Conversation history
        - 📝 Save conversations
        - 🔄 Streaming responses
        
        **How to use:**
        1. Type your question in the chat input
        2. Press Enter or click Send
        3. Wait for the AI's response
        4. Continue the conversation naturally
        """)
        
        st.markdown("---")
        
        # Conversation controls
        st.header("🛠️ Controls")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Chat", use_container_width=True):
                save_conversation()
        
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                clear_conversation()
        
        # Statistics
        st.markdown("---")
        st.header("📊 Stats")
        st.metric("Messages", len(st.session_state.messages))
        st.metric("User Messages", len([m for m in st.session_state.messages if m["role"] == "user"]))
        st.metric("Bot Responses", len([m for m in st.session_state.messages if m["role"] == "assistant"]))
    
    # Main chat interface
    st.header("💬 Chat")
    
    # Display existing messages
    for message in st.session_state.messages:
        display_chat_message(message["role"], message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about the knowledge base..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        display_chat_message("user", prompt)
        
        # Generate bot response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.kb_client.get_response_from_knowledgebase(prompt)
                    
                    # Create placeholder for streaming effect
                    response_placeholder = st.empty()
                    
                    # Stream the response
                    final_response = stream_response(response, response_placeholder)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    
                except Exception as e:
                    error_message = f"I apologize, but I encountered an error: {str(e)}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Powered by AWS Bedrock Knowledge Base | Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()