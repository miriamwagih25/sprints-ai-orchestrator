"""Generate mock Sprints curriculum documents for the RAG pipeline."""
from pathlib import Path

OUT = Path("data/curriculum_docs")
OUT.mkdir(parents=True, exist_ok=True)

DOCS = {
"01_program_overview.md": """# Advanced Agentic AI Bootcamp: Program Overview

The Advanced Agentic AI Bootcamp is a 12-week, cohort-based program run by Sprints.
Learners attend three live sessions per week and complete one hands-on project each week.

## Who it is for
Learners should know basic Python and have completed an introductory machine learning course.
No GPU is required, because all models are accessed through cloud APIs.

## Learning outcomes
By the end of the program, learners can build retrieval-augmented generation (RAG) systems,
design tool-using agents, orchestrate multi-agent workflows with LangGraph, and evaluate
their systems with measurable quality criteria.
""",

"02_modules_and_schedule.md": """# Modules and Weekly Schedule

## Module 1: Python for AI and LLM Basics (Weeks 1-2)
Prompting, structured outputs with Pydantic, and calling LLM APIs.

## Module 2: RAG and Vector Databases (Weeks 3-4)
Document loading, chunking strategies, embeddings, and ChromaDB retrieval.
Learners measure retrieval quality and latency.

## Module 3: Tool Use and ReAct Agents (Weeks 5-6)
Building custom tools, including SQL tools and a Python REPL tool, and the ReAct
reasoning loop.

## Module 4: LangGraph Orchestration (Weeks 7-9)
StateGraph, shared state, conditional routing, memory with checkpointers,
and human-in-the-loop interrupts.

## Module 5: Evaluation, Deployment and Capstone (Weeks 10-12)
Evaluator-optimizer loops, testing agents, and the final capstone project.
""",

"03_assessment_and_grading.md": """# Assessment and Grading Policy

The pass mark for the bootcamp is 60 percent overall.

## Python assignments
Each coding assignment is graded on correctness (50 percent), code quality (30 percent),
and documentation (20 percent). Code must include docstrings.

## Feedback and resubmission
Learners receive written feedback on every assignment and may resubmit once
within five days to improve their score.

## Late work
Submissions up to 48 hours late lose 10 percent. Later submissions are not accepted
without an approved extension.

## Capstone project
The capstone counts for 40 percent of the final grade and is presented to a panel of
instructors and employer partners.
""",

"04_career_support.md": """# Career Support and Employer Pathways

Sprints connects learners with employer partners through placement pathways.

## What learners receive
CV and LinkedIn workshops, mock technical interviews, and portfolio reviews.

## Typical roles
AI engineer, LLM application developer, machine learning engineer, and data analyst.

## Advisor guidance
Advisors compare a learner's progress and stated career goals with the skills
required by target roles and recommend the next modules or projects to focus on.
Salary expectations are discussed during one-to-one advising sessions.
""",

"05_faq.md": """# Frequently Asked Questions

## What is the attendance policy?
Learners must attend at least 80 percent of live sessions. All sessions are recorded.

## Can I defer to the next cohort?
Yes. A learner may defer once, before week 4, by contacting their advisor.

## How do I receive the certificate?
Complete all assignments, score at least 60 percent overall, and present the capstone.

## Where do I get help?
Use the cohort chat for quick questions and book an advisor session for career questions.
""",
}

for name, text in DOCS.items():
    (OUT / name).write_text(text, encoding="utf-8")
    print("wrote", OUT / name)