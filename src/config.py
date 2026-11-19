import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()  # makes GOOGLE_API_KEY from .env available to the client

MODEL_NAME = "gemini-3.5-flash-lite"


def get_llm(temperature: float | None = None) -> ChatGoogleGenerativeAI:
    """Return the shared LLM client used by every agent.
    Temperature is left at Google's default unless set explicitly."""
    kwargs = {"model": MODEL_NAME, "max_retries": 5}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatGoogleGenerativeAI(**kwargs)