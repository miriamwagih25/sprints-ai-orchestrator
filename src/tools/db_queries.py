import sqlite3
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CSV_PATH = DATA_DIR / "sprints_learner_db-csv.csv"
DB_PATH = DATA_DIR / "sprints_learner.db-csv"
TABLE = "learners"
MAX_ROWS = 50


def build_database() -> None:
    """Load the learner CSV into a SQLite table. Safe to re-run."""
    df = pd.read_csv(CSV_PATH)
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(TABLE, conn, if_exists="replace", index=False)


@tool
def describe_learner_db() -> str:
    """Show the columns and 3 sample rows of the Sprints learner table.
    Call this FIRST to learn the column names before writing any SQL."""
    try:
        conn = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)
        try:
            cols = conn.execute(f"PRAGMA table_info({TABLE})").fetchall()
            sample = pd.read_sql_query(f"SELECT * FROM {TABLE} LIMIT 3", conn)
        finally:
            conn.close()
    except Exception as e:
        return f"Unable to verify: could not read the database ({e})."
    schema = ", ".join(f"{c[1]} ({c[2]})" for c in cols)
    return f"Table '{TABLE}': {schema}\n\nSample rows:\n{sample.to_string(index=False)}"


@tool
def query_learner_db(sql: str) -> str:
    """Run ONE read-only SQL SELECT query on the Sprints learner table
    (table name: learners) and return up to 50 rows. Use it to look up a
    learner's bootcamp track, progress, or career goals. Only SELECT is allowed."""
    sql_clean = sql.strip().rstrip(";")
    if not sql_clean.lower().startswith("select") or ";" in sql_clean:
        return "Unable to verify: only a single read-only SELECT query is allowed."
    try:
        conn = sqlite3.connect(f"{DB_PATH.as_uri()}?mode=ro", uri=True)
        try:
            cur = conn.execute(sql_clean)
            cols = [d[0] for d in cur.description]
            rows = cur.fetchmany(MAX_ROWS)
        finally:
            conn.close()
    except Exception as e:
        return f"Unable to verify: the query failed ({e})."
    if not rows:
        return "Unable to verify: the query returned no rows."
    return pd.DataFrame(rows, columns=cols).to_string(index=False)