"""Python execution tool for the Assessor agent.

Runs student code in a separate Python process with a timeout and an output
limit, so an infinite loop or a crash in the student's code cannot freeze or
break the main program.
"""
import subprocess
import sys
import tempfile
from pathlib import Path
import re

from langchain_core.tools import tool

TIMEOUT_SECONDS = 5
MAX_OUTPUT_CHARS = 2000


def run_python(code: str, timeout: int = TIMEOUT_SECONDS) -> dict:
    """Run code in a fresh Python process.
    Returns a dict with: status ('success' | 'error' | 'timeout'), stdout, stderr."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "student_code.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-X", "utf8", str(script)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                cwd=tmp,
            )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "stdout": "",
                "stderr": f"Code ran longer than {timeout} seconds and was stopped.",
            }
    return {
        "status": "success" if proc.returncode == 0 else "error",
        "stdout": proc.stdout[:MAX_OUTPUT_CHARS],
        "stderr": proc.stderr[:MAX_OUTPUT_CHARS],
    }
def strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```python ... ```) that LLMs often add."""
    text = text.strip()
    match = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    return (match.group(1) if match else text).strip()


@tool
def run_student_code(code: str) -> str:
    """Execute Python code submitted by a learner and report what happened:
    its printed output, or the error message if it crashed or ran too long.
    Use it to check whether a learner's solution actually works."""
    try:
        result = run_python(strip_code_fences(code))
    except Exception as e:
        return f"Unable to verify: the code runner failed ({e})."
    if result["status"] == "success":
        return f"Execution succeeded.\nOutput:\n{result['stdout'] or '(no output)'}"
    if result["status"] == "timeout":
        return f"Execution failed: {result['stderr']}"
    return f"Execution failed.\nError:\n{result['stderr']}"


if __name__ == "__main__":
    print(run_student_code.invoke({"code": "print(sum(range(10)))"}))
    print(run_student_code.invoke({"code": "print(1/0)"}))
    print(run_student_code.invoke({"code": "while True: pass"}))