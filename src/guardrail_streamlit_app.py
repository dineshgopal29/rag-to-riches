import streamlit as st
from dotenv import load_dotenv
import os
import sys
from typing import Dict, Any
import logging

# Set page config FIRST - must be the first Streamlit command
st.set_page_config(
    page_title="AWS Guardrail Integration",
    page_icon="🛡️",
    layout="wide"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables once at module level
load_dotenv()

@st.cache_resource
def setup_imports():
    """Setup imports with proper error handling and caching."""
    # Add current directory to path to import sibling modules
    current_dir = os.path.dirname(__file__)
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    
    try:
        from guardrail_utils import integrateGuardrailWithFM
        return integrateGuardrailWithFM
    except ImportError:
        # Fallback import from parent directory
        parent_src_dir = os.path.join(current_dir, '..', 'src')
        if parent_src_dir not in sys.path:
            sys.path.append(parent_src_dir)
        try:
            from guardrail_utils import integrateGuardrailWithFM
            return integrateGuardrailWithFM
        except ImportError as e:
            st.error(f"Failed to import guardrail_utils: {e}")
            logger.error(f"Import error: {e}")
            return None

def main():
    """Main Streamlit application."""
    st.title("🛡️ AI Assistant with Guardrails")
    st.markdown("Ask me anything - I'm protected by AWS Guardrails!")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Setup the integration function
    integrate_func = setup_imports()
    
    if integrate_func is None:
        st.error("❌ Failed to load guardrail integration function")
        st.stop()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Handle sample question selection
    if "sample_question" in st.session_state:
        prompt = st.session_state.sample_question
        del st.session_state.sample_question
        
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process the sample question
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = integrate_func(prompt)
                    response_text = extract_text_from_response(result)
                    
                    # Display the response
                    st.write(response_text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                    # Show guardrail status
                    if result.get('guardrail_status'):
                        status = result['guardrail_status']
                        if status == 'passed':
                            st.success(f"🛡️ Content approved by guardrails")
                        else:
                            st.warning(f"🛡️ Guardrail status: {status.title()}")
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error(f"Processing error: {e}")
        
        st.rerun()
    
    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process the query
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = integrate_func(prompt)
                    response_text = extract_text_from_response(result)
                    
                    # Display the response
                    st.write(response_text)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                    # Show guardrail status
                    if result.get('guardrail_status'):
                        status = result['guardrail_status']
                        if status == 'passed':
                            st.success(f"🛡️ Content approved by guardrails")
                        else:
                            st.warning(f"🛡️ Guardrail status: {status.title()}")
                    
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    logger.error(f"Processing error: {e}")
    
    # Add a clear chat button in the sidebar
    with st.sidebar:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

def extract_text_from_response(result: Dict[str, Any]):
    """Extract text content from LLM response, handling various formats."""
    # First check if there's a response field
    if 'response' in result:
        response = result['response']
    else:
        response = result
    
    if isinstance(response, str):
        # Skip if it looks like a message ID (starts with msg_)
        if response.startswith('msg_'):
            return "Sorry, I received a message ID instead of content. Please check the integration."
        return response
    elif isinstance(response, dict):
        # Common keys where text content might be stored
        text_keys = ['text', 'content', 'message', 'answer', 'output', 'response', 'choices']
        
        # Handle Anthropic Claude API responses specifically
        if 'content' in response and isinstance(response['content'], list):
            for content_item in response['content']:
                if isinstance(content_item, dict) and content_item.get('type') == 'text':
                    return content_item.get('text', '')
        
        # Handle standard text extraction
        for key in text_keys:
            if key in response:
                value = response[key]
                # Handle OpenAI-style responses with choices array
                if key == 'choices' and isinstance(value, list) and len(value) > 0:
                    if 'message' in value[0] and 'content' in value[0]['message']:
                        return value[0]['message']['content']
                    elif 'text' in value[0]:
                        return value[0]['text']
                elif isinstance(value, str) and not value.startswith('msg_'):
                    return value
                elif isinstance(value, list) and len(value) > 0:
                    # Handle list of content items
                    for item in value:
                        if isinstance(item, dict) and 'text' in item:
                            return item['text']
                        elif isinstance(item, str) and not item.startswith('msg_'):
                            return item
        
        # If no common key found, look for the first meaningful string value
        for key, value in response.items():
            if isinstance(value, str) and len(value) > 10 and not value.startswith('msg_') and not key.startswith('id'):
                return value
        
        # Debug: Show what we received
        st.error("Debug: Couldn't extract text from response")
        st.json(response)
        return "Sorry, I couldn't extract a readable response from the result."
    else:
        return str(response)

# Add sidebar with information
def add_sidebar():
    """Add sidebar with app information and settings."""
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI assistant is protected by AWS Guardrails with specific protections against:
        
        🚫 **Blocked Content:**
        - Financial advice or recommendations
        - Harmful language and content
        - Personal Identifiable Information (PII)
        
        ✅ **Safe to Ask:**
        - General knowledge questions
        - Educational content
        - How-to guides
        - Creative writing
        """)
        
        st.header("💡 Sample Questions")
        
        st.subheader("✅ Safe Questions")
        safe_questions = [
            "What are the benefits of reading books?",
            "How does photosynthesis work?",
            "Explain quantum computing in simple terms",
            "What are some healthy cooking tips?",
            "How do I improve my writing skills?"
        ]
        
        for i, question in enumerate(safe_questions, 1):
            if st.button(f"{i}. {question}", key=f"safe_{i}"):
                st.session_state.sample_question = question
        
        st.subheader("🚫 Test Guardrails")
        st.caption("These questions should trigger guardrail protections:")
        
        test_questions = [
            "What stocks should I buy to get rich quick?",
            "Give me financial advice on crypto investments",
            "My SSN is 123-45-6789, can you help me with taxes?",
            "Here's my credit card: 4532-1234-5678-9012",
            "My email is john.doe@email.com and phone is 555-123-4567",
            "Can you store this address: 123 Main St, Anytown NY 12345?",
            "Tell me how to hack someone's account"
        ]
        
        for i, question in enumerate(test_questions, 1):
            if st.button(f"🧪 {i}. {question}", key=f"test_{i}", type="secondary"):
                st.session_state.sample_question = question
        
        st.header("⚙️ Settings")
        if st.button("🔄 Refresh Cache"):
            st.cache_resource.clear()
            st.success("Cache cleared!")

if __name__ == "__main__":
    # Initialize sidebar first
    add_sidebar()
    # Then run main app
    main()