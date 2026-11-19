from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.config import get_llm
from src.state import AgentState
from src.tools.rag_retriever import retrieve

MAX_ATTEMPTS = 3
FALLBACK = "Unable to verify: I could not find this information in the curriculum documents."


class Reflection(BaseModel):
    """The model's judgement of the retrieved context."""

    sufficient: bool = Field(
        description="True if the context contains the information needed to answer the question."
    )
    better_query: str = Field(
        description="If not sufficient, a rewritten search query likely to find the missing "
        "information. Empty string if sufficient."
    )


def retrieve_node(state: AgentState) -> dict:
    """Search the curriculum and add the new chunks to the context."""
    query = state.get("search_query") or state["question"]
    attempts = state.get("attempts", 0) + 1
    try:
        hits, _ = retrieve(query)
    except Exception:
        # Retrieval is broken: stop retrying and fall back.
        return {"attempts": MAX_ATTEMPTS, "sufficient": False}
    context = list(state.get("context", []))
    for h in hits:
        chunk = f"[{h['source']}] {h['text']}"
        if chunk not in context:
            context.append(chunk)
    return {"context": context, "attempts": attempts}


def reflect_node(state: AgentState) -> dict:
    """Ask the LLM whether the context is enough to answer the question."""
    if state.get("attempts", 0) >= MAX_ATTEMPTS and not state.get("context"):
        return {"sufficient": False}
    prompt = (
        "You check whether retrieved text can answer a learner's question.\n"
        f"Question: {state['question']}\n\nRetrieved context:\n"
        + "\n---\n".join(state.get("context", []))
    )
    try:
        llm = get_llm().with_structured_output(Reflection, method="function_calling")
        result = llm.invoke(prompt)
    except Exception:
        # If reflection fails, answer with what we have rather than crash.
        return {"sufficient": bool(state.get("context"))}
    return {
        "sufficient": result.sufficient,
        "search_query": result.better_query or state["question"],
    }


def route_after_reflect(state: AgentState) -> str:
    """Re-query if the context is weak and attempts remain, otherwise answer."""
    if state.get("sufficient") or state.get("attempts", 0) >= MAX_ATTEMPTS:
        return "answer"
    return "retrieve"


def answer_node(state: AgentState) -> dict:
    """Write the final answer from the context, or return the fallback."""
    if not state.get("sufficient"):
        return {"answer": FALLBACK, "messages": [AIMessage(FALLBACK)]}
    prompt = (
        "You are the Sprints learner advisor. Answer the question using ONLY the "
        "context below. Be concise and mention the source file name(s). "
        "If the context does not contain the answer, reply exactly: "
        f"{FALLBACK}\n\nContext:\n" + "\n---\n".join(state["context"])
        + f"\n\nQuestion: {state['question']}"
    )
    try:
        text = get_llm().invoke(prompt).text
    except Exception:
        text = FALLBACK
    return {"answer": text, "messages": [AIMessage(text)]}


def build_advisor_graph():
    """Assemble and compile the Advisor's LangGraph workflow."""
    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reflect", reflect_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "reflect")
    graph.add_conditional_edges(
        "reflect", route_after_reflect, {"retrieve": "retrieve", "answer": "answer"}
    )
    graph.add_edge("answer", END)
    return graph.compile()


if __name__ == "__main__":
    advisor = build_advisor_graph()
    for q in ["How many sessions can I miss?", "What is the price of the bootcamp?"]:
        out = advisor.invoke({"question": q})
        print(f"\nQ: {q}\nAttempts: {out['attempts']}\nA: {out['answer']}")