import streamlit as st
import os
import shutil
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

from rag import ingest_file
from agent import create_agent

# Load environment variables
load_dotenv()

# Page Config
st.set_page_config(page_title="Deep Research Agent", page_icon="🕵️‍♂️", layout="wide")

# Title
st.title("🕵️‍♂️ Deep Research Agent")

# Sidebar for Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    model_provider = st.selectbox(
        "Select Model Provider",
        ["Gemini", "OpenRouter"]
    )
    
    if model_provider == "Gemini":
        model_name = st.text_input("Model Name", value="gemini-2.5-flash-lite")
    else:
        model_name = st.text_input("Model Name", value="google/gemini-2.0-flash-exp:free")
        st.caption("Ensure OPENROUTER_API_KEY is set in .env for OpenRouter.")

    st.divider()
    
    st.header("📚 Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        # Save file temporarily
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            try:
                num_chunks = ingest_file(file_path)
                st.success(f"Ingested {num_chunks} chunks from {uploaded_file.name}!")
            except Exception as e:
                st.error(f"Error ingesting file: {e}")
            finally:
                # Cleanup
                if os.path.exists(file_path):
                    os.remove(file_path)

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = create_agent()
                
                # Construct LangChain messages from history (optional, for now just sending last message)
                # To enable full conversation history, we'd map st.session_state.messages to LangChain messages
                
                inputs = {
                    "messages": [HumanMessage(content=prompt)],
                    "context": "",
                    "sources": [],
                    "research_plan": "",
                    "sub_queries": []
                }
                
                # Pass configuration to the agent
                config = {"configurable": {"model_provider": model_provider, "model_name": model_name}}
                
                result = agent.invoke(inputs, config=config)
                response_content = result["messages"][-1].content
                
                st.markdown(response_content)
                st.session_state.messages.append({"role": "assistant", "content": response_content})
            except Exception as e:
                st.error(f"An error occurred: {e}")
