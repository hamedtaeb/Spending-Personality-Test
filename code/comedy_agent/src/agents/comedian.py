from langchain_anthropic import ChatAnthropic
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import MemorySaver
from src.shared.shared_store import store
from src.shared.context import Context
from config.settings import (
    ANTHROPIC_MODEL,
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_TOP_P,
    ANTHROPIC_THINKING_BUDGET
)
from src.tools.search import search_tool
from src.tools.scraper import content_extraction
from src.tools.summarizer import summarizer


def create_comedian_agent(theory_text: str, checkpointer=None):
    """
    Create a comedian agent with the specified theory and tools.
    
    Args:
        theory_text: The full theory text to use in the system prompt
        checkpointer: Optional checkpointer for conversation memory
        
    Returns:
        Configured agent executor
    """
    system_prompt = f"""You are an edgy Chinese comedian from China who recently immigrated to America with a style of comedy based on the incongruent theory from Arthur Schopenhauer.

    # DETAILED THEORY CONTENT

    {theory_text}

    ---

    # YOUR INSTRUCTIONS

    Use the content from the summarizer tool output as material to create the joke.

    When formulating the joke:
    1. Think of a scenario that is both relatable and polarizing in society while being nuanced and breezy about it.
    2. Apply the theoretical foundations of what makes good humor from the incongruent theory incorporated within your system prompt.
    3. Make sure the joke has the structure outlined under JOKE FORMAT below this list of instructions.
    4. If the joke does not follow the specified structure, try again until the joke has the intended structure.
    5. Do not reveal your source of material as it takes away from the buildup and do not include the output format markers as it looks tacky, taking away from the content.
    6. Juxtapose different ideas and use contradictory narratives.
    7. Include shock/awe factors in each joke.
    8. Never say the same joke more than once and the produce human-like output.

    # JOKE FORMAT
    Joke must follow the structure below:
    The setup: This introduces the characters, setting, and situation, leading the audience to a specific, predictable assumption.
    The punchline: This delivers a surprising twist that shatters the audience's initial assumption, revealing a new, equally applicable perspective.
    Relatability and truth: Start with a factual, truthful, or relatable piece of information. Ground the premise in harsh reality, to pull the user in, making the surprise more effective.
    Use the "Rule of Three" when the user submits multiple messages. Rule of three definition: This common comedy device establishes a pattern with two similar items before surprising the audience with a third, different one during a user.
    """

    # Initialize the model
    model = ChatAnthropic(
        model=ANTHROPIC_MODEL,
        max_tokens=ANTHROPIC_MAX_TOKENS,
        temperature=0.69
        # thinking={"type": "enabled", "budget_tokens": ANTHROPIC_THINKING_BUDGET},
        # top_p=ANTHROPIC_TOP_P
    )
    
    # Define tools
    tools = [search_tool, content_extraction, summarizer]
    
    # Create checkpointer if not provided
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    # Create and return agent
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=system_prompt,
        # middleware=[
        #     HumanInTheLoopMiddleware({
        #         "__root__": {
        #             "allowed_decisions" : ["Funny", "Not Funny"]
        #         }
        #     })
        # ],
        store=store,
        context_schema=Context,
        checkpointer=checkpointer
    )
    
    return agent