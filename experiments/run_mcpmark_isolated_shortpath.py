#!/usr/bin/env python3
"""run_mcpmark_isolated_shortpath.py -- same per-task server-restart isolation
as run_mcpmark_isolated.py, but pointed at a copy of mcpmark relocated to a
short absolute path (/private/tmp/mm) instead of this session's deeply
nested working directory. Controls for path-length-driven context bloat:
does shortening the root path (which every tool call/output path is prefixed
with) meaningfully change token counts and outcomes for the tasks that
stalled in the with_fix2_timeout3600_isolated run?

Runs the ORIGINAL mcpmark .venv's python3 interpreter (not copied -- venvs
aren't reliably relocatable) with cwd set to the short-path copy, since
_get_project_root() in filesystem_state_manager.py resolves from
Path(__file__), i.e. from wherever the running code physically lives, not
from sys.path or cwd.

Before running this, create the short-path copy (code + fixtures only, no
.venv, no results):
    mkdir -p /private/tmp/mm
    cp -a $MCPMARK_DIR/{src,tasks,test_environments,pipeline.py,pyproject.toml,.mcp_env} /private/tmp/mm/

Supplementary: this path-length ablation is not tabulated in the paper; see
docs/index.md for the full writeup.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import requests

# MCPMark (https://github.com/eval-sys/mcpmark) is a separate, third-party
# repo -- clone it yourself and point ORIGINAL_MCPMARK_DIR at it.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS_DIR = os.environ.get("NANBEIGE_HARNESS_DIR", os.path.join(_REPO_ROOT, "harness"))
ORIGINAL_MCPMARK_DIR = os.environ.get("MCPMARK_DIR", os.path.join(os.path.dirname(_REPO_ROOT), "mcpmark"))
SHORTPATH_MCPMARK_DIR = os.environ.get("MCPMARK_SHORTPATH_DIR", "/private/tmp/mm")
HARNESS_PYTHON = os.environ.get("NANBEIGE_HARNESS_PYTHON", sys.executable)
MCPMARK_PYTHON = os.environ.get("MCPMARK_PYTHON", f"{ORIGINAL_MCPMARK_DIR}/.venv/bin/python3")
SERVER_URL = "http://127.0.0.1:8100/v1/models"
EXP_NAME = "shortpath_timeout3600"
TIMEOUT = 3600

TASKS = [
    "file_context/pattern_matching",
    "papers/papers_counting",
    "student_database/duplicate_name",
]


def start_server():
    proc = subprocess.Popen(
        [HARNESS_PYTHON, "nanbeige_harness_server.py", "--port", "8100"],
        cwd=HARNESS_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(60):
        try:
            requests.get(SERVER_URL, timeout=2)
            return proc
        except requests.exceptions.RequestException:
            time.sleep(5)
    raise RuntimeError("harness server did not come up in time")


def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)


def run_task(task):
    print(f"=== task {task} ===", flush=True)
    result = subprocess.run(
        [
            MCPMARK_PYTHON, "-m", "pipeline",
            "--mcp", "filesystem", "--task-suite", "easy",
            "--models", "openai/nanbeige42-harness",
            "--exp-name", EXP_NAME,
            "--tasks", task,
            "--timeout", str(TIMEOUT),
            "--k", "1",
        ],
        cwd=SHORTPATH_MCPMARK_DIR,
    )
    print(f"=== task {task} exited {result.returncode} ===", flush=True)


def main():
    for task in TASKS:
        proc = start_server()
        try:
            run_task(task)
        finally:
            stop_server(proc)


if __name__ == "__main__":
    main()
