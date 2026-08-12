#!/usr/bin/env python3
"""discover_tool_schema.py -- MCP tool discovery only, no torch import at all,
so it can't interact with the later model-loading process's MPS/Metal state.
Writes the combined 55-tool OpenAI-shape schema to JSON for
reproduce_original_oom.py to load separately.

Requires the (separate, private) ironclad-agent framework installed and
importable -- set IRONCLAD_AGENT_DIR to its repo root, or pip-install it into
this environment. Included for provenance (how results/tool_schema_55.json,
already committed in this repo, was generated), not as a turnkey standalone
script: reproducing this exact step requires ironclad-agent's own MCP server
configuration (11 connected servers) to match what produced that file.
"""
import asyncio
import copy
import json
import os
import sys

sys.path.insert(0, os.environ.get("IRONCLAD_AGENT_DIR", "../ironclad_agent"))

from ironclad_agent.config import AgentConfig, _CONFIG_DEFAULTS
from ironclad_agent.promptopt import _discover_tools_by_server
from ironclad_agent.agent import mcp_tools_to_schema


def main():
    d = copy.deepcopy(_CONFIG_DEFAULTS)
    d["model"]["vllm"]["base_url"] = "http://127.0.0.1:8100/v1"
    d["model"]["vllm"]["model"] = "nanbeige42-harness"
    d["mcp"]["active"] = []
    cfg = AgentConfig.from_config(d)

    tools_by_server = asyncio.run(_discover_tools_by_server(cfg))
    all_tools = [t for tools in tools_by_server.values() for t in tools]
    print(f"discovered {len(all_tools)} tools across {len(tools_by_server)} servers: "
          f"{list(tools_by_server.keys())}")
    schema = mcp_tools_to_schema(all_tools)

    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results/tool_schema_55.json")
    with open(out_path, "w") as f:
        json.dump(schema, f)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
