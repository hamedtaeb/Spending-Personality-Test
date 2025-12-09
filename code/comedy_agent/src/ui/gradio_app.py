import uuid
import gradio as gr
from langsmith import traceable
from langgraph.checkpoint.memory import MemorySaver
from src.shared.context import Context
from src.data.loader import TheoryLoader
from src.agents.comedian import create_comedian_agent


# Initialize components
theory_loader = TheoryLoader()
full_theory = theory_loader.get_full_theory_text()
checkpointer = MemorySaver()
agent = None

def initialize_agent():
    """Initialize the agent and checkpointer. Call this after load_dotenv()."""
    global agent, checkpointer
    
    if agent is None:
        theory_loader = TheoryLoader()
        full_theory = theory_loader.get_full_theory_text()
        checkpointer = MemorySaver()
        agent = create_comedian_agent(full_theory, checkpointer)
    
    return agent

@traceable(name="chat_bot")
def chat_with_agent(message, history):
    """
    Handle chat interactions with the comedian agent.
    
    Args:
        message: User's input message
        history: Chat history (handled by Gradio)
        
    Yields:
        Agent responses as they are generated
    """
    # Generate unique session id
    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}
    
    full_response = ""
    
    try:
        print(f"Starting agent stream...")
        
        step_count = 0
        for step in agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            context=Context(
                session_id=session_id
            ),
            config=config,
            stream_mode="values"
        ):
            step_count += 1
            print(f"\nStep {step_count}:")
            print(f"  Keys in step: {step.keys() if isinstance(step, dict) else 'Not a dict'}")
            
            # Get the last message from the step
            if "messages" in step and len(step["messages"]) > 0:
                last_message = step["messages"][-1]
                print(f"  Last message type: {type(last_message)}")
                print(f"  Last message role: {getattr(last_message, 'role', 'no role')}")
                
                # Extract content from the message
                if hasattr(last_message, 'content'):
                    content = last_message.content
                    print(f"  Content type: {type(content)}")
                    
                    # Handle string content
                    if isinstance(content, str):
                        full_response = content
                        print(f"  Yielding string content (length: {len(content)})")
                        yield full_response
                        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"\nERROR in chat_with_agent:")
        print(error_details)
        
        error_msg = f"**Error**: {str(e)}\n\n"
        yield error_msg


def create_gradio_interface():
    """Create and configure the Gradio interface."""
    with gr.Blocks(title="MaybeFunny Bot") as demo:
        gr.Markdown("Request a joke, or whatever you want and the bot will give gewd jokes..I hope")
        gr.Markdown("")
        
        chat_interface = gr.ChatInterface(
            fn=chat_with_agent,
            chatbot=gr.Chatbot(height=500),
            textbox=gr.Textbox(placeholder="What you got for me?"),
            flagging_mode="manual",
            title=None,
            description="Write a topic or request for a joke",
        )
    
    return demo