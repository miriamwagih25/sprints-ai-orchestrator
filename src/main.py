"""Interactive entry point for the Sprints AI Orchestrator.

Commands:
  <any text>   ask a question (routed to the Advisor or the learner database)
  /grade       grade a built-in sample submission (may trigger human review)
  /escalate    grade the sample and force human review (for demonstration)
  /student ID  switch to another student's session (separate memory)
  /quit        exit
"""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from src.agents.orchestrator import build_orchestrator

load_dotenv()

SAMPLE_ASSIGNMENT = (
    "Write a function average(numbers) that returns the mean of a list, "
    "then print average([3, 7, 10, 15]). Expected output: 8.75"
)
SAMPLE_CODE = (
    "def average(numbers):\n"
    "    return sum(numbers) / len(numbers) + 1\n"
    "print(average([3, 7, 10, 15]))"
)


def run(app, payload, config) -> str:
    """Invoke the graph; if it pauses for human review, ask the instructor."""
    result = app.invoke(payload, config)
    snapshot = app.get_state(config)
    while snapshot.next:  # the graph is paused at an interrupt
        info = snapshot.tasks[0].interrupts[0].value
        print(f"\n[HUMAN REVIEW] {info['message']}")
        print(f"Automatic score: {info['score']}\nFeedback: {info['feedback']}")
        raw = input("Enter a new score, or press Enter to accept: ").strip()
        decision = {"score": int(raw)} if raw.isdigit() else "accept"
        result = app.invoke(Command(resume=decision), config)
        snapshot = app.get_state(config)
    return result["messages"][-1].content


def main() -> None:
    app = build_orchestrator()
    student = "student-1"
    print(__doc__)
    while True:
        text = input(f"\n[{student}] > ").strip()
        if not text:
            continue
        if text == "/quit":
            break
        if text.startswith("/student"):
            student = text.split(maxsplit=1)[1] if " " in text else student
            print(f"Switched to {student}")
            continue
        config = {"configurable": {"thread_id": student}}
        if text in ("/grade", "/escalate"):
            payload = {
                "messages": [HumanMessage("Please grade my submission.")],
                "assignment": SAMPLE_ASSIGNMENT,
                "student_code": SAMPLE_CODE,
                "force_review": text == "/escalate",
            }
        else:
            payload = {"messages": [HumanMessage(text)]}
        try:
            print("\n" + run(app, payload, config))
        except Exception as e:
            print(f"Unable to verify: something went wrong ({e}).")


if __name__ == "__main__":
    main()