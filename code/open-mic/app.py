import requests
import gradio as gr
from dotenv import load_dotenv
from gradio_client import Client

url = "https://huggingface.co/spaces/PlebbyMcPlebFace/zhe-wrok"

headers = {
    "Authorization": "HF_BEARER_TOKEN",
    "Content-Type": "application/json"
}

client = Client("https://plebbymcplebface-zhe-wrok.hf.space/")

try:
    result = client.predict(
	messages=[],
	api_name="/flag"
    )
    yield result

except Exception as e:
    yield f"Error: {e}"