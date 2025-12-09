from typing import List
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from src.shared.context import Context
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_core.vectorstores import InMemoryVectorStore
from src.shared.shared_store import store
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import EMBEDDING_MODEL


# Initialize vector store (singleton pattern)
_embeddings = None
_vector_store = None


def get_vector_store() -> InMemoryVectorStore:
    """Get or create the vector store singleton."""
    global _embeddings, _vector_store
    
    if _vector_store is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _vector_store = InMemoryVectorStore(embedding=_embeddings)
        
    return _vector_store


@tool("search_tool", 
      description="Search user query for relevant content and returns a list with relevant web material",
      parse_docstring=False)
def search_tool(query: str, runtime: ToolRuntime[Context]) -> str:
    """
    Searches the internet for content relevant to the user query.
    The results of the search are embedded and stored in a short term memory vector database.
    A similarity search is executed and resulting list is returned.
    
    Args:
        query: Search query string
        
    Returns:
        List of relevant documents
    """

    search = DuckDuckGoSearchRun()
    results = DuckDuckGoSearchResults(output_format="list")
    
    # Execute search
    content = search.invoke(query)
    top_results = results.run(query)
    
    # Store in vector database
    vector_store = get_vector_store()
    
    for result in top_results:
        vector_store.add_documents([
            Document(
                page_content=result["snippet"],
                metadata={
                    "title": result["title"],
                    "link": result["link"]
                }
            )
        ])
    
    # Retrieve similar documents
    top_results = vector_store.similarity_search(query)

    ctx = runtime.context

    session_id = ctx.session_id

    workflow_data = {
        "query": query,
        "search_results": top_results,
        "next_input": "search_results"
    }

    runtime.store.put(("session_id", session_id), "workflow_data", workflow_data)
    
    return "Cooking up something to smell..."