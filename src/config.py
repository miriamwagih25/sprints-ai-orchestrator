import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

MODEL_NAME = "gemini-3.5-flash-lite"

def get_llm(temperature: float = 0):
    """Return the shared LLM client used by every agent."""
    return ChatOpenAI(
        model=MODEL_NAME,
        api_key=os.getenv("GOOGLE_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        temperature=temperature,
        max_retries=5,
    )