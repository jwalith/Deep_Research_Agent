from langchain_community.tools import DuckDuckGoSearchRun

def web_search(query: str):
    search = DuckDuckGoSearchRun()
    return search.invoke(query)
