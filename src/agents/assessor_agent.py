from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from src.config import get_llm
from src.state import AgentState
from src.tools.code_repl import run_student_code

MAX_ITERATIONS = 1
FALLBACK = "Unable to verify: the code could not be executed, so it cannot be graded reliably."
CRITIC_UNAVAILABLE = "Critic unavailable: this grade was not independently reviewed."
RUBRIC = (
    "Correctness 50 points (meets the assignment and runs without errors), "
    "code quality 30 points (clarity, naming, structure), "
    "documentation 20 points (docstrings and comments). Total 100."
)


class Grade(BaseModel):
    """The grader's structured output."""

    score: int = Field(ge=0, le=100, description="Total score out of 100 using the rubric.")
    feedback: str = Field(
        description="Specific, actionable feedback for the learner, referring to the "
        "actual output or error message."
    )


class Critique(BaseModel):
    """The critic's structured review of the grader's work."""

    approved: bool = Field(description="True only if the grade and feedback pass every check.")
    critique: str = Field(
        description="If not approved, what exactly must be fixed. Empty string if approved."
    )


def run_code_node(state: AgentState) -> dict:
    """Execute the learner's code with the REPL tool."""
    return {
        "execution": run_student_code.invoke({"code": state["student_code"]}),
        "iterations": 0,
        "history": [],
    }


def route_after_run(state: AgentState) -> str:
    """If the runner itself broke, skip grading and return the fallback."""
    return "fallback" if state["execution"].startswith("Unable to verify") else "grade"


def grade_node(state: AgentState) -> dict:
    """Grader LLM: score the submission. On later rounds it must address the critique."""
    iteration = state.get("iterations", 0) + 1
    prompt = (
        "You are a fair Python instructor grading a learner submission.\n"
        f"Rubric: {RUBRIC}\n\nAssignment:\n{state['assignment']}\n\n"
        f"Learner code:\n{state['student_code']}\n\n"
        f"Execution result:\n{state['execution']}\n"
    )
    if state.get("critique"):
        prompt += (
            "\nA reviewer rejected your previous grading.\n"
            f"Previous score: {state.get('score')}\nPrevious feedback: {state.get('feedback')}\n"
            f"Reviewer critique: {state['critique']}\nFix these problems in your new grade.\n"
        )
    try:
        grade = get_llm().with_structured_output(Grade, method="function_calling").invoke(prompt)
    except Exception:
        return {"iterations": MAX_ITERATIONS, "score": 0, "feedback": FALLBACK, "approved": False}
    return {"iterations": iteration, "score": grade.score, "feedback": grade.feedback}


def critique_node(state: AgentState) -> dict:
    """Critic LLM: check that the grade is consistent, specific and actionable."""
    if state.get("feedback") == FALLBACK:
        return {}
    prompt = (
        "You are a strict reviewer of an automated grader. Check ALL of these:\n"
        "1. The score is consistent with the execution result (code that crashed or "
        "printed wrong output must lose correctness points).\n"
        "2. The feedback names specific problems and refers to the real output or error.\n"
        "3. The feedback is actionable for the learner.\n"
        "4. The score is within the rubric.\n"
        f"Rubric: {RUBRIC}\n\nAssignment:\n{state['assignment']}\n\n"
        f"Learner code:\n{state['student_code']}\n\nExecution result:\n{state['execution']}\n\n"
        f"Grader score: {state['score']}\nGrader feedback: {state['feedback']}"
    )
    try:
        review = get_llm().with_structured_output(Critique, method="function_calling").invoke(prompt)
    except Exception:
        return {"approved": False, "critique": CRITIC_UNAVAILABLE}
    entry = {
        "iteration": state["iterations"],
        "score": state["score"],
        "approved": review.approved,
        "critique": review.critique,
    }
    return {
        "approved": review.approved,
        "critique": review.critique,
        "history": state.get("history", []) + [entry],
    }


def route_after_critique(state: AgentState) -> str:
    """Finish if approved, out of iterations, or the critic is down; else re-grade."""
    if (
        state.get("approved")
        or state.get("iterations", 0) >= MAX_ITERATIONS
        or state.get("critique") == CRITIC_UNAVAILABLE
    ):
        return "finalize"
    return "grade"


def finalize_node(state: AgentState) -> dict:
    """Write the final result; flag the grade for a human if it was never approved."""
    text = (
        f"Score: {state['score']}/100 "
        f"(iterations: {state['iterations']}, reviewer approved: {'yes' if state.get('approved') else 'no'})\n\n"
        f"{state['feedback']}"
    )
    if not state.get("approved"):
        text += "\n\nFlagged for human review."
    return {"messages": [AIMessage(text)]}


def fallback_node(state: AgentState) -> dict:
    """Return the structured fallback when the code runner failed."""
    return {
        "score": 0,
        "feedback": FALLBACK,
        "approved": False,
        "iterations": 0,
        "messages": [AIMessage(FALLBACK)],
    }


def build_assessor_graph():
    """Assemble and compile the Assessor's LangGraph workflow."""
    graph = StateGraph(AgentState)
    graph.add_node("run_code", run_code_node)
    graph.add_node("grade", grade_node)
    graph.add_node("critique", critique_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("fallback", fallback_node)
    graph.add_edge(START, "run_code")
    graph.add_conditional_edges(
        "run_code", route_after_run, {"grade": "grade", "fallback": "fallback"}
    )
    graph.add_edge("grade", "critique")
    graph.add_conditional_edges(
        "critique", route_after_critique, {"grade": "grade", "finalize": "finalize"}
    )
    graph.add_edge("finalize", END)
    graph.add_edge("fallback", END)
    return graph.compile()


if __name__ == "__main__":
    assessor = build_assessor_graph()
    out = assessor.invoke(
        {
            "assignment": (
                "Write a function average(numbers) that returns the mean of a list, "
                "then print average([3, 7, 10, 15]). Expected output: 8.75"
            ),
            "student_code": (
                "def average(numbers):\n"
                "    return sum(numbers) / len(numbers) + 1\n"
                "print(average([3, 7, 10, 15]))"
            ),
        }
    )
    print(out["messages"][-1].content)
    print("\nLoop history:")
    for step in out["history"]:
        print(step)