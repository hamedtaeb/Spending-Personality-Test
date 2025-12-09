import os
from dotenv import load_dotenv
from src.ui.gradio_app import create_gradio_interface, initialize_agent
import gradio as gr
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Load env vars immediately
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

app = FastAPI(title="zhé Wrok", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://huggingface.co"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"]
)

@app.on_event("startup")
async def startup_event():
    initialize_agent()

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Mount Gradio at root
demo = create_gradio_interface()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)