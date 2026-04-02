# Import the necessary packages
import os

import httpx
from langchain_openai import ChatOpenAI
import gradio as gr

from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY: str | None = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL: str = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_ID: str = "nvidia/nemotron-3-super-120b-a12b:free"
MODEL_TEMPERATURE: float = 0.85


llm_model = ChatOpenAI(
    model=MODEL_ID,
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=OPENROUTER_BASE_URL,
    temperature=MODEL_TEMPERATURE,
    http_client=httpx.Client(verify=False),
)

# Function to generate a response from the model
def generate_response(prompt_txt: str) -> str:
    """Generate a response from the model"""
    generated_response = llm_model.invoke(prompt_txt)
    return generated_response.content


# Gradio interface
# Create Gradio interface
chat_application = gr.Interface(
    fn=generate_response,
	allow_flagging="never",
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Custom LLM Chatbot",
    description="Ask any question and the chatbot will try to answer."
)

# Launch the application
chat_application.launch(server_name="0.0.0.0", server_port=7860)