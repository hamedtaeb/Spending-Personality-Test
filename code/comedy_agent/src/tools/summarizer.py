from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from src.shared.context import Context
from src.shared.shared_store import store
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import GEMINI_MODEL


# Initialize summarizer LLM
_summarizer_llm = None


def get_summarizer_llm() -> ChatGoogleGenerativeAI:
    """Get or create the summarizer LLM singleton."""
    global _summarizer_llm
    
    if _summarizer_llm is None:
        _summarizer_llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL)
        
    return _summarizer_llm


@tool("summarizer", 
      description="Summarizes output from the content_extraction tool and returns a summary",
      parse_docstring=False)
def summarizer(full_context: str, runtime: ToolRuntime[Context]) -> str:
    """
    Summarizes the output from the content_extraction tool.
    Returns a string with a 4-5 sentence summary of the input.
    
    Args:
        full_content: The full text content to summarize
        
    Returns:
        A 4-5 sentence summary highlighting joke material
    """
    ctx = runtime.context
    session_id = ctx.session_id
    prev = runtime.store.get(("session_id", session_id), "workflow_data")

    llm = get_summarizer_llm()
    
    summarizer_prompt = (
        "Summarize the input in 4-5 sentences. "
        "Be sure to highlight material that would make for great joke material."
    )
    workflow_data = prev.value.copy()
    messages = [
        ("system", summarizer_prompt),
        ("human", workflow_data["web_context"])
    ]
    
    response = llm.invoke(messages)
    
    workflow_data = prev.value.copy()
    workflow_data.update({
        "context_summary": response
    })

    runtime.store.put(("session_id", session_id), "workflow_data", workflow_data)

    return "Hold on I think I've got something"