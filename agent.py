import os
import re
from typing import Annotated, List, TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from rag import get_retriever
from search import web_search

class AgentState(TypedDict):
    messages: List[BaseMessage]
    context: str
    sources: List[str]
    research_plan: str
    sub_queries: List[str]

def get_llm(model_provider="Gemini", model_name="gemini-2.5-flash-lite"):
    if model_provider == "Gemini":
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found. Please set it in .env")
        return ChatGoogleGenerativeAI(model=model_name, temperature=0)
    elif model_provider == "OpenRouter":
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("OPENROUTER_API_KEY not found. Please set it in .env")
        # OpenRouter uses OpenAI client but with a different base URL
        return ChatOpenAI(
            model=model_name, 
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
    else:
        # Fallback or standard OpenAI
        return ChatOpenAI(model=model_name, temperature=0)

def _format_source_label(doc):
    source = doc.metadata.get("source", "uploaded file")
    page = doc.metadata.get("page")
    if page is not None:
        try:
            page_num = int(page) + 1
            return f"File: {source} (page {page_num})"
        except (TypeError, ValueError):
            pass
    return f"File: {source}"

def plan_research(state: AgentState, config: RunnableConfig):
    """Break down complex questions into sub-questions for deeper research."""
    model_provider = config.get("configurable", {}).get("model_provider", "Gemini")
    model_name = config.get("configurable", {}).get("model_name", "gemini-2.0-flash-exp")
    
    llm = get_llm(model_provider, model_name)
    query = state["messages"][-1].content
    
    planning_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a research planning assistant. Break down complex questions into 3-5 focused sub-questions that will help provide a comprehensive answer.

For simple questions that can be answered directly, return the original question unchanged.
For complex questions, generate 3-5 sub-questions that explore different aspects.

Format your response as:
PLAN: [brief research strategy]
SUB_QUESTIONS:
1. [first sub-question]
2. [second sub-question]
...

If the question is simple, respond with:
PLAN: Direct answer
SUB_QUESTIONS:
1. [original question]
"""),
        ("human", "{question}"),
    ])
    
    chain = planning_prompt | llm
    plan_response = chain.invoke({"question": query})
    plan_text = plan_response.content
    
    # Parse sub-questions
    sub_queries = []
    lines = plan_text.split('\n')
    in_sub_questions = False
    
    for line in lines:
        if 'SUB_QUESTIONS:' in line.upper():
            in_sub_questions = True
            continue
        if in_sub_questions and line.strip():
            # Extract question from numbered list
            match = re.match(r'^\d+\.\s*(.+)', line.strip())
            if match:
                sub_queries.append(match.group(1).strip())
            elif line.strip() and not line.strip().startswith('PLAN:'):
                sub_queries.append(line.strip())
    
    # If no sub-questions found, use original query
    if not sub_queries:
        sub_queries = [query]
    
    return {
        "research_plan": plan_text,
        "sub_queries": sub_queries
    }


def generate_search_queries(state: AgentState, config: RunnableConfig):
    """Generate multiple query variations for comprehensive search."""
    model_provider = config.get("configurable", {}).get("model_provider", "Gemini")
    model_name = config.get("configurable", {}).get("model_name", "gemini-2.0-flash-exp")
    
    llm = get_llm(model_provider, model_name)
    original_query = state["messages"][-1].content
    sub_queries = state.get("sub_queries", [original_query])
    
    # Generate 3-5 query variations for each sub-question
    all_queries = set([original_query])  # Always include original
    
    query_gen_prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate 2-3 alternative search query variations for the given question. 
These should be different phrasings that might find different information.
Return only the queries, one per line, without numbering or bullets."""),
        ("human", "{question}"),
    ])
    
    chain = query_gen_prompt | llm
    
    for sub_q in sub_queries[:3]:  # Limit to first 3 sub-questions to avoid too many queries
        try:
            variations = chain.invoke({"question": sub_q})
            for var in variations.content.strip().split('\n'):
                var = var.strip()
                if var and len(var) > 10:  # Filter out very short or empty queries
                    all_queries.add(var)
        except Exception:
            # If query generation fails, just use the sub-question
            all_queries.add(sub_q)
    
    return {"sub_queries": list(all_queries)[:5]}  # Limit to 5 total queries

def retrieve_or_search(state: AgentState):
    """Retrieve from RAG and perform multi-query web search."""
    original_query = state["messages"][-1].content
    search_queries = state.get("sub_queries", [original_query])
    
    retriever = get_retriever()
    context_lines = []
    sources = []
    
    # RAG retrieval on original query
    try:
        docs = retriever.invoke(original_query)
        if docs:
            context_lines.append("From Uploaded Files:")
            for idx, doc in enumerate(docs, start=1):
                context_lines.append(f"Source {idx}: {doc.page_content.strip()}")
                sources.append(_format_source_label(doc))
    except Exception:
        docs = []

    # Multi-query web search
    all_search_results = []
    for query in search_queries[:5]:  # Limit to 5 queries
        try:
            search_res = web_search(query)
            if search_res and search_res.strip():
                all_search_results.append(f"Query: {query}\n{search_res.strip()}")
        except Exception:
            continue
    
    if all_search_results:
        context_lines.append("\nWeb Search Results:")
        context_lines.append("\n\n---\n\n".join(all_search_results))
        sources.append("Web Search")

    context = "\n".join(context_lines).strip()

    return {"context": context, "sources": sources}

def generate(state: AgentState, config: RunnableConfig):
    # Extract model config from configurable args
    model_provider = config.get("configurable", {}).get("model_provider", "Gemini")
    model_name = config.get("configurable", {}).get("model_name", "gemini-2.0-flash-exp")
    
    llm = get_llm(model_provider, model_name)
    context = state["context"]
    sources = state.get("sources", [])
    messages = state["messages"]

    # Separate document sources from web search
    doc_sources = [s for s in sources if s != "Web Search"]
    has_web_search = "Web Search" in sources
    
    sources_for_prompt = []
    if doc_sources:
        sources_for_prompt.append("\n".join(f"[{i+1}] {label}" for i, label in enumerate(doc_sources)))
    if has_web_search:
        sources_for_prompt.append("[Web Search] Web Search")
    
    sources_list_text = "\n".join(sources_for_prompt) if sources_for_prompt else "No sources available."
    
    research_plan = state.get("research_plan", "")
    plan_section = f"\n\nResearch Plan:\n{research_plan}" if research_plan else ""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a Deep Research Agent. Synthesize a comprehensive research report based on the provided context.

        Format your response as a structured research report:

        # Research Report

        ## Executive Summary
        [2-3 sentence overview of key findings]

        ## Key Findings
        [Main points and insights, organized by topic. Use inline citations [1], [2], or [Web Search] for each claim]

        ## Detailed Analysis
        [Deeper exploration of findings with supporting evidence and citations]

        ## Conclusion
        [Summary and implications]

        ## Sources
        [List all sources used]

        Instructions:
        1. If information is available in the uploaded files, cite it using [1], [2], etc.
        2. If information is only available from Web Search, cite it as [Web Search].
        3. Synthesize information from multiple sources when available.
        4. If no information is available, state "I don't know" in the Executive Summary.
        5. Always prioritize uploaded file sources when available, but use web search when uploaded files don't contain the answer.

        Available Sources:
        {sources_list}
        {plan_section}

        Context:
        {context}
        """),
        ("human", "{question}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke(
        {
            "context": context,
            "sources_list": sources_list_text,
            "plan_section": plan_section,
            "question": messages[-1].content,
        }
    )

    # Parse response to find which citations were actually used
    response_text = response.content
    
    # Find all numbered citations [1], [2], [1, 2, 3], etc. in the entire report
    used_doc_indices = set()
    # Match both individual citations [1] and comma-separated lists [1, 2, 3]
    # This pattern matches any sequence of digits, commas, and spaces inside square brackets
    for match in re.finditer(r'\[([\d\s,]+)\]', response_text):
        # Extract all numbers from the match (handles both [1] and [1, 2, 3])
        numbers = re.findall(r'\d+', match.group(1))
        for num_str in numbers:
            idx = int(num_str)
            # Only add if the index is valid and we have that many doc sources
            if 1 <= idx <= len(doc_sources) and len(doc_sources) > 0:
                used_doc_indices.add(idx - 1)  # Convert to 0-based index
    
    # Check if [Web Search] was used
    used_web_search = bool(re.search(r'\[Web Search\]', response_text, re.IGNORECASE))
    
    # Format sources for display - only include actually cited sources
    # CRITICAL: Only show sources that were actually cited in the response
    sources_display = []
    
    # Only add document sources if they were actually cited
    if used_doc_indices:
        for idx in sorted(used_doc_indices):
            if idx < len(doc_sources):  # Safety check
                sources_display.append(f"[{idx + 1}] {doc_sources[idx]}")
    
    # Only add web search if it was actually cited
    if used_web_search:
        sources_display.append("[Web Search] Web Search")
    
    # Remove the model's Sources section and replace with our formatted one
    # Be very aggressive - remove any "Sources" section the model might have generated
    # Handle multiple variations: "## Sources", "Sources:", "Sources" (standalone), etc.
    
    # First, try to find and remove the Sources section (everything from "## Sources" or "Sources:" to end)
    # This regex matches from "## Sources" or "Sources:" (case insensitive) to the end of the string
    answer_text = re.sub(
        r'(\n|^)##?\s*Sources?.*$',
        '',
        response_text,
        flags=re.IGNORECASE | re.DOTALL | re.MULTILINE
    )
    
    # Also remove any "Sources:" followed by content
    answer_text = re.sub(
        r'\n\s*Sources?:?\s*\n.*$',
        '',
        answer_text,
        flags=re.IGNORECASE | re.DOTALL
    )
    
    # Remove any standalone "Sources" at the end of lines
    answer_text = re.sub(
        r'\n\s*Sources?\s*$',
        '',
        answer_text,
        flags=re.IGNORECASE | re.MULTILINE
    )
    
    # Clean up any double newlines that might result
    answer_text = re.sub(r'\n{3,}', '\n\n', answer_text)
    answer_text = answer_text.strip()
    
    if sources_display:
        sources_text = "\n\n## Sources\n" + "\n".join(sources_display)
    else:
        sources_text = "\n\n## Sources\nNo sources were cited."

    # Use the cleaned answer text (without model's Sources section) + our formatted sources
    final_content = answer_text + sources_text
    final_message = AIMessage(content=final_content)
    
    return {"messages": [final_message]}

def create_agent():
    workflow = StateGraph(AgentState)
    
    # Add nodes: plan -> generate queries -> retrieve/search -> generate report
    workflow.add_node("plan_research", plan_research)
    workflow.add_node("generate_queries", generate_search_queries)
    workflow.add_node("retrieve_search", retrieve_or_search)
    workflow.add_node("generate", generate)
    
    # Set entry point
    workflow.set_entry_point("plan_research")
    
    # Define flow: plan -> generate queries -> retrieve/search -> generate -> end
    workflow.add_edge("plan_research", "generate_queries")
    workflow.add_edge("generate_queries", "retrieve_search")
    workflow.add_edge("retrieve_search", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
