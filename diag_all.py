"""Run each LLM-dependent piece alone and show the real error if it fails."""
import traceback

from src.agents.advisor_agent import Reflection
from src.agents.orchestrator import Route
from src.config import get_llm
from src.tools.rag_retriever import retrieve


def attempt(name, fn):
    try:
        print(f"[OK]   {name}: {fn()}")
    except Exception:
        print(f"[FAIL] {name}")
        traceback.print_exc()


ROUTE_Q = "Classify this request: how many learners are there in total"
REFLECT_Q = "Is 'Learners must attend at least 80 percent of sessions' enough to answer: what is the attendance policy?"

attempt("plain invoke .text", lambda: get_llm().invoke("Say hi in 3 words").text)
attempt("retrieve", lambda: retrieve("attendance policy")[0][0]["source"])
attempt(
    "Route, function_calling",
    lambda: get_llm().with_structured_output(Route, method="function_calling").invoke(ROUTE_Q),
)
attempt("Route, default method", lambda: get_llm().with_structured_output(Route).invoke(ROUTE_Q))
attempt(
    "Reflection, function_calling",
    lambda: get_llm().with_structured_output(Reflection, method="function_calling").invoke(REFLECT_Q),
)
attempt("Reflection, default method", lambda: get_llm().with_structured_output(Reflection).invoke(REFLECT_Q))