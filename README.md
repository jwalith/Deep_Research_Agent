# Deep Research Agent

A comprehensive research assistant that combines RAG (Retrieval-Augmented Generation) with web search to provide well-sourced, in-depth research reports.

![Deep Research Agent Interface](screenshot.png)

## Features

### 🔍 Deep Research Capabilities
- **Research Planning**: Automatically breaks down complex questions into focused sub-questions
- **Multi-Query Search**: Generates multiple search query variations for comprehensive coverage
- **Structured Reports**: Produces well-organized research reports with:
  - Executive Summary
  - Key Findings
  - Detailed Analysis
  - Conclusion

### 📚 RAG (Retrieval-Augmented Generation)
- Upload PDF or text files to build a knowledge base
- Semantic search across uploaded documents
- Automatic document chunking and vector storage

### 🔗 Citations & Sources
- **Inline Citations**: Every claim is backed by source citations `[1]`, `[2]`, or `[Web Search]`
- **Source Tracking**: Only shows sources that were actually cited in the response
- **Transparent Attribution**: Clear distinction between uploaded documents and web search results

### 🌐 Web Search Integration
- Automatic web search when uploaded documents don't contain answers
- Multi-query strategy for diverse information gathering
- Seamless integration with RAG results

## Technologies Used

- **LangChain**: Agent framework and RAG implementation
- **LangGraph**: Multi-step research workflow
- **ChromaDB**: Vector store for document embeddings
- **Streamlit**: Web interface
- **OpenRouter**: LLM API access (supports multiple models)
- **DuckDuckGo**: Web search capabilities

## Setup

1. Install dependencies:
```bash
cd backend
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
OPENROUTER_API_KEY=your_api_key_here
```

3. Run the application:
```bash
streamlit run backend/app.py
```

## Usage

1. **Upload Documents** (optional): Add PDF or text files to build your knowledge base
2. **Select Model**: Choose your preferred model provider (Gemini or OpenRouter)
3. **Ask Questions**: The agent will:
   - Plan the research approach
   - Search your documents and the web
   - Generate a comprehensive research report with citations

## Example Output

The agent generates structured research reports with proper citations:

```
# Research Report

## Executive Summary
[Overview with citations]

## Key Findings
[Main points with inline citations [1], [2], [Web Search]]

## Detailed Analysis
[In-depth exploration with sources]

## Conclusion
[Summary]

## Sources
[1] File: document.pdf (page 1)
[Web Search] Web Search
```

## How It Works

1. **Planning**: Breaks down complex questions into sub-questions
2. **Query Generation**: Creates multiple search query variations
3. **Retrieval**: Searches uploaded documents (RAG) and the web
4. **Synthesis**: Combines information into a structured report
5. **Citation**: Tracks and displays only sources actually used

---

Built with ❤️ for comprehensive research and information synthesis.

