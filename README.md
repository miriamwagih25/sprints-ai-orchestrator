# sprints-ai-orchestrator
A multi-agent system for EdTech company **Sprints**, built with **LangGraph**. An orchestrator routes each request to a specialised worker:

- **Advisor** answers questions about the bootcamp (program, modules, grading, attendance, careers) using agentic RAG over curriculum documents.
- **Learner-data agent** answers questions about learners using a read-only SQL tool over a SQLite database.
- **Assessor** grades learners' Python code by actually running it, then improves its own grade with an evaluator-optimizer loop. Grades the reviewer never approves are escalated to a human.

---

## Architecture

```
                        user message
                             |
                             v
                   +-------------------+
                   |      Router       |   (LLM with structured output;
                   +-------------------+    rule: code submission -> Assessor)
                  /        |        |      \
                 v         v        v       v
          +---------+ +---------+ +------+ +-----------------------------+
          | Advisor | | Learner | | Chat | |          Assessor           |
          | (RAG)   | |  data   | |      | | run code -> grade -> critic |
          +---------+ | (SQL)   | +------+ |   ^____ max 3 loops ____|   |
                      +---------+          +-------------+---------------+
                                                         | not approved
                                                         v
                                               +------------------+
                                               |   Human review   |
                                               |   (interrupt)    |
                                               +------------------+
```

The whole graph is compiled with a `MemorySaver` checkpointer. Each student has their own `thread_id`, so conversations are remembered per student and never mixed.

### Advisor (agentic RAG)
`retrieve -> reflect -> (re-query with a rewritten search, up to 3 attempts) -> answer`

After each retrieval, the LLM judges whether the retrieved text is enough to answer. If not, it rewrites the search query and retrieves again. If the information is still missing after 3 attempts, the Advisor returns a structured *"Unable to verify"* message instead of guessing.

### Assessor (evaluator-optimizer)
`run code -> grade -> critique -> (re-grade using the critique, up to 3 iterations) -> finalize`

One LLM call grades the submission with a rubric (correctness 50, code quality 30, documentation 20) using the real execution output. A second LLM call acts as a strict critic and checks that the score matches the output and that the feedback is specific and true. The loop stops after at most 3 iterations. If the critic never approves, the grade is flagged and the graph pauses for a human decision.

---

## How the requirements are met

| Requirement | Where |
|---|---|
| Curriculum ingestion into ChromaDB (load, chunk, embed, store) | `src/tools/rag_retriever.py` |
| Pydantic structured outputs (`Route`, `Reflection`, `Grade`, `Critique`) | `src/agents/` |
| Custom SQL tool over the learner database | `src/tools/db_queries.py` |
| Python REPL tool for running learner code | `src/tools/code_repl.py` |
| LangGraph `StateGraph` with a global `TypedDict` state | `src/state.py`, `src/agents/orchestrator.py` |
| Per-session memory (`MemorySaver`) | `src/agents/orchestrator.py` |
| Agentic RAG with self-reflection | `src/agents/advisor_agent.py` |
| Evaluator-optimizer loop, max 3 iterations | `src/agents/assessor_agent.py`, `notebooks/evaluator_optimizer_loop.ipynb` |
| Human-in-the-loop escalation (`interrupt`) | `src/agents/orchestrator.py`, `src/main.py` |
| Retrieval latency under 1000 ms | measured and logged in `retrieve()` (about 184 ms in testing) |
| Structured fallback when a tool fails | every tool returns `Unable to verify: ...` instead of crashing |
| Docstrings on all code | all modules |

---

## Setup

Requires Python 3.10 or newer.

```bash
# 1. Clone and enter the project
git clone https://github.com/miriamwagih25/sprints-ai-orchestrator.git
cd sprints-ai-orchestrator

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Mac / Linux

# 3. Install dependencies
pip install -r requirements.txt
```

### API key
Create a free Gemini API key at [aistudio.google.com](https://aistudio.google.com), then create a file named `.env` in the project root:

```
GOOGLE_API_KEY=your_key_here
```

The `.env` file is listed in `.gitignore` and is never committed. The model is set in one place, `src/config.py` (`MODEL_NAME`).

### Build the data stores (run once)

```bash
# SQLite database from the learner CSV
python -c "from src.tools.db_queries import build_database; build_database()"

# ChromaDB index from data/curriculum_docs/ (also prints a latency check)
python -m src.tools.rag_retriever
```

---

## Usage

```bash
python -m src.main
```

| Input | What happens |
|---|---|
| `What is the attendance policy?` | Routed to the Advisor (RAG), answer cites the source file |
| `How many learners are there in total?` | Routed to the learner-data agent (SQL) |
| `What was my previous question?` | Answered from the stored conversation |
| `/grade` | Grades a built-in sample submission; may pause for human review |
| `/student <id>` | Switches to another student's session (separate memory) |
| `/quit` | Exit |

The evaluator-optimizer loop is evaluated in `notebooks/evaluator_optimizer_loop.ipynb` (select the `.venv` kernel and run all cells).

---

## Project structure

```
sprints-ai-orchestrator/
├── data/
│   ├── sprints_learner_db.csv       learner records
│   └── curriculum_docs/             curriculum documents used for RAG
├── src/
│   ├── config.py                    shared LLM client (one place to change the model)
│   ├── state.py                     global LangGraph state
│   ├── agents/
│   │   ├── orchestrator.py          router, worker nodes, memory, human review
│   │   ├── advisor_agent.py         agentic RAG with self-reflection
│   │   └── assessor_agent.py        evaluator-optimizer grading loop
│   ├── tools/
│   │   ├── rag_retriever.py         ChromaDB ingestion and search
│   │   ├── code_repl.py             subprocess-based Python runner
│   │   └── db_queries.py            read-only SQL tools
│   └── main.py                      interactive command-line entry point
├── notebooks/
│   └── evaluator_optimizer_loop.ipynb
├── docs/
│   ├── technical_report.pdf
│   └── final_presentation.pdf
├── requirements.txt
└── README.md
```

---

## Design decisions

- **One LLM factory (`get_llm`).** Every agent gets its model from `src/config.py`, so changing the provider or model is a one-line edit. This was useful in practice: the project switched providers during development without touching any agent code.
- **Rule-based routing for submissions.** A message that carries code always goes to the Assessor with no LLM call. Only free-text questions are classified by the model, with the Advisor as the safe default.
- **Read-only SQL.** The database is opened in read-only mode and only a single `SELECT` statement is accepted, so a faulty model-generated query cannot modify or delete data.
- **Local embeddings.** ChromaDB's default embedding model runs on the CPU with no API key, which keeps retrieval fast and the project free to run.
- **Fail loudly, not silently.** Tools and agents return `Unable to verify: ...` rather than inventing an answer when retrieval, SQL, or code execution fails.

---

## Limitations

- **The code runner is not a security sandbox.** Learner code runs in a separate Python process with a 5-second timeout and an output limit, which protects against infinite loops and crashes, but it can still access files and the network. A production system would use Docker or a similar isolated environment.
- **The critic is also an LLM.** It can miss mistakes, for example approving feedback that praises documentation that does not exist. Deterministic checks (such as comparing the program output with the expected output) would make grading more reliable.
- **Mock curriculum.** The curriculum documents in `data/curriculum_docs/` are generated sample content, not Sprints' real curriculum.
- **Free-tier rate limits.** Heavy testing can hit the per-minute limits of the free Gemini tier. Retries are enabled, but a burst of requests may still be slow.
- **A network failure looks like missing information.** If the LLM cannot be reached, the Advisor currently falls back to the same "Unable to verify" message used when documents lack the answer.
