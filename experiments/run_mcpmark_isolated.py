#!/usr/bin/env python3
"""run_mcpmark_isolated.py -- run the MCPMark filesystem-easy suite one task
at a time, restarting nanbeige_harness_server.py fresh before each task.

Motivation: diagnose_mps_leak.py confirmed that a single caught
`RuntimeError: MPS backend out of memory` permanently degrades the process's
usable MPS memory budget -- neither torch.mps.empty_cache() nor gc.collect()
reclaim it, only a full process restart does. In a long-lived server handling
many sequential MCPMark tasks, one task's OOM (itself expected, since
real multi-turn conversations grow past the chunked-prefill memory ceiling
established in Section 3.2/Table 2) otherwise cascades into spurious OOMs on every
later, unrelated task. This mirrors the subprocess-per-trial isolation
pattern already used for the batch-size sweep (run_batch_sweep.py), applied
here per MCPMark task instead of per memory trial.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

import requests

# MCPMark (https://github.com/eval-sys/mcpmark) is a separate, third-party
# repo -- clone it yourself and point these at it. HARNESS_DIR/MCPMARK_DIR
# can use different Python environments (this repo's own deps vs. MCPMark's),
# hence separate interpreter paths rather than a single shared venv.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HARNESS_DIR = os.environ.get("NANBEIGE_HARNESS_DIR", os.path.join(_REPO_ROOT, "harness"))
MCPMARK_DIR = os.environ.get("MCPMARK_DIR", os.path.join(os.path.dirname(_REPO_ROOT), "mcpmark"))
HARNESS_PYTHON = os.environ.get("NANBEIGE_HARNESS_PYTHON", sys.executable)
MCPMARK_PYTHON = os.environ.get("MCPMARK_PYTHON", f"{MCPMARK_DIR}/.venv/bin/python3")
SERVER_URL = "http://127.0.0.1:8100/v1/models"
EXP_NAME = "with_fix2_timeout3600_isolated"
TIMEOUT = 3600

TASKS = [
    "file_context/file_splitting",
    "file_context/pattern_matching",
    "file_context/uppercase",
    "file_property/largest_rename",
    "file_property/txt_merging",
    "folder_structure/structure_analysis",
    "legal_document/file_reorganize",
    "papers/papers_counting",
    "student_database/duplicate_name",
    "student_database/recommender_name",
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
        cwd=MCPMARK_DIR,
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
