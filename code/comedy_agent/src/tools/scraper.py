import json
import nest_asyncio
from typing import List, Dict, Optional
from src.shared.shared_store import store
from langchain_core.tools import tool
from langchain.tools import ToolRuntime
from src.shared.context import Context
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_async_playwright_browser, create_sync_playwright_browser
from langchain_core.documents import Document


@tool("search_content", 
      description="Scrapes web content from search_tool results",
      parse_docstring=False)
def content_extraction(search_results: str, runtime: ToolRuntime[Context]) -> str:
    """
    Scrapes the web content of the input list and returns a string.
    
    Args:
        search_results: List of Document objects with metadata containing links
        
    Returns:
        Extracted text content from the first successfully scraped page
    """
    nest_asyncio.apply()

    ctx = runtime.context
    session_id = ctx.session_id
    
    prev = runtime.store.get(("session_id", session_id), "workflow_data")
    if not prev:
        return "No workflow found"

    search_results = prev.value["search_results"]
    
    browser = create_sync_playwright_browser()

    toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=browser)
    
    tools = toolkit.get_tools()
    tools_by_name = {tool.name: tool for tool in tools}
    
    navigate_tool = tools_by_name["navigate_browser"]
    get_elements_tool = tools_by_name["get_elements"]
    
    full_context = ""
    
    for result in search_results:
        try:
            # Navigate to the URL
            navigate_tool.run({"url": str(result.metadata["link"])})
            
            # Extract paragraph elements
            result_json = get_elements_tool.run({"selector": "p"})
            json_text = json.loads(result_json)
            
            # Extract inner text from all paragraphs
            text_chunks = [item["innerText"] for item in json_text]
            full_context = "\n".join(text_chunks)

            workflow_data = prev.value.copy()

            workflow_data.update({
                "step": 2,
                "web_context": str(full_context),
                "next_input": "web_context ready"
            })

            runtime.store.put(("session_id", session_id), "workflow_data", workflow_data)

            # Successfully extracted content
            break
            
        except Exception as e:
            print(f"Failed to fetch content from {result.metadata.get('link', 'unknown')}: {e}")
            continue

    browser.close()

    return "I can't smell a thing..."
