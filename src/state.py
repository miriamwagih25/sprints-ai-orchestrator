"""Shared state passed between all agents in the LangGraph workflow."""
from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """Global state of the orchestrator graph. Each agent reads and writes
    only the fields it needs."""

    # Conversation
    messages: Annotated[list, add_messages]
    route: str  # which worker the orchestrator picked

    # Advisor agent (agentic RAG)
    question: str
    search_query: str
    context: list[str]
    attempts: int
    sufficient: bool
    answer: str

    # Assessor agent 
    assignment: str
    student_code: str
    execution: str
    score: int
    feedback: str
    critique: str
    approved: bool
    iterations: int
    history: list[dict]
    force_review: bool