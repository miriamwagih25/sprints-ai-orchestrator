"""Orchestrator: routes each request to the right worker, keeps per-student
memory with MemorySaver, and escalates unapproved grades to a human.

Routing:
  - a submission with code        : Assessor (rule-based, no LLM call needed)
  - a question about learner data : SQL ReAct agent
  - any other question            :Advisor (agentic RAG)
"""
from functools import lru_cache
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import interrupt
from pydantic import BaseModel, Field


from src.agents.advisor_agent import build_advisor_graph
from src.agents.assessor_agent import build_assessor_graph
from src.config import get_llm
from src.state import AgentState
from src.tools.db_queries import describe_learner_db, query_learner_db

DATA_FALLBACK = "Unable to verify: I could not retrieve this from the learner database."

DATA_PROMPT = (
    "You are a data assistant for Sprints, an EdTech company. Answer questions "
    "about learners using the database tools. Always call describe_learner_db "
    "first to see the real column names, then write one SELECT query with "
    "query_learner_db. Base your answer only on the query results. If a tool "
    "returns 'Unable to verify', say so instead of guessing."
)


class Route(BaseModel):
    """The router's decision."""

    route: Literal["advisor", "learner_data", "chat"] = Field(
        description="'learner_data' for questions about learners' records, progress, "
        "tracks or goals stored in the database; 'advisor' for questions about the "
        "bootcamp itself (program, modules, grading, attendance, careers, FAQ); "
        "'chat' for follow-ups about the conversation itself, such as 'repeat my "
        "previous question' or 'what did I just ask', and for greetings or thanks."
    )


@lru_cache(maxsize=1)
def _advisor():
    return build_advisor_graph()


@lru_cache(maxsize=1)
def _assessor():
    return build_assessor_graph()


@lru_cache(maxsize=1)
def _data_agent():
    return create_react_agent(
        get_llm(), tools=[describe_learner_db, query_learner_db], prompt=DATA_PROMPT
    )


def router_node(state: AgentState) -> dict:
    """Decide which worker handles the request."""
    if state.get("student_code"):
        return {"route": "assessor"}
    question = state["messages"][-1].content
    prompt = f"Classify this request from a learner or staff member:\n{question}"
    try:
        decision = get_llm().with_structured_output(Route, method="function_calling").invoke(prompt)
        return {"route": decision.route}
    except Exception:
        return {"route": "advisor"}  # safe default


def advisor_node(state: AgentState) -> dict:
    """Run the Advisor graph on the latest question."""
    question = state["messages"][-1].content
    out = _advisor().invoke({"question": question})
    return {"answer": out["answer"], "messages": [AIMessage(out["answer"])]}


def learner_data_node(state: AgentState) -> dict:
    """Run the SQL ReAct agent with the whole conversation, so follow-ups work."""
    try:
        out = _data_agent().invoke({"messages": state["messages"]})
        text = out["messages"][-1].text
    except Exception as e:
        text = f"{DATA_FALLBACK} (debug: {e})"
    return {"messages": [AIMessage(text)]}


def assessor_node(state: AgentState) -> dict:
    """Run the Assessor graph. Approved grades are returned directly;
    unapproved ones go on to human review."""
    out = _assessor().invoke(
        {"assignment": state["assignment"], "student_code": state["student_code"]}
    )
    approved = out["approved"] and not state.get("force_review")
    update = {
        "score": out["score"],
        "feedback": out["feedback"],
        "approved": approved,
        "iterations": out["iterations"],
        "history": out["history"],
        "student_code": "",  # clear so the next message is routed normally
        "assignment": "",
        "force_review": False,
    }
    if approved:
        update["messages"] = [out["messages"][-1]]
    return update


def human_review_node(state: AgentState) -> dict:
    """Human-in-the-loop: pause the graph and wait for an instructor's decision.
    Resume with {"score": int, "feedback": str} to override, or anything else
    to accept the automatic grade."""
    decision = interrupt(
        {
            "message": "The automatic grade was not approved. Please review it.",
            "score": state["score"],
            "feedback": state["feedback"],
        }
    )
    score, feedback = state["score"], state["feedback"]
    if isinstance(decision, dict):
        score = int(decision.get("score", score))
        feedback = decision.get("feedback", feedback)
    text = f"Final score (human-reviewed): {score}/100\n\n{feedback}"
    return {
        "score": score,
        "feedback": feedback,
        "approved": True,
        "messages": [AIMessage(text)],
    }

def chat_node(state: AgentState) -> dict:
    """Answer conversational follow-ups from the stored conversation history."""
    system = SystemMessage(
        "You are the Sprints assistant. Use the conversation history to answer "
        "follow-up questions about what was said earlier. Be brief."
    )
    try:
        text = get_llm().invoke([system] + state["messages"]).text
    except Exception:
        text = "Unable to verify: I could not read the conversation history."
    return {"messages": [AIMessage(text)]}


def route_from_router(state: AgentState) -> str:
    """Pick the next node from the router's decision."""
    return state["route"]


def route_after_assessor(state: AgentState) -> str:
    """Approved grades finish; the rest escalate to a human."""
    return END if state.get("approved") else "human_review"


def build_orchestrator():
    """Assemble the orchestrator graph with per-thread memory."""
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("advisor", advisor_node)
    graph.add_node("learner_data", learner_data_node)
    graph.add_node("assessor", assessor_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("chat", chat_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_from_router,
        {"advisor": "advisor", 
         "learner_data": "learner_data",
         "chat": "chat", 
         "assessor": "assessor"},
    )
    graph.add_edge("advisor", END)
    graph.add_edge("learner_data", END)
    graph.add_conditional_edges(
        "assessor", route_after_assessor, {END: END, "human_review": "human_review"}
    )
    graph.add_edge("human_review", END)
    graph.add_edge("chat", END)
    return graph.compile(checkpointer=MemorySaver())