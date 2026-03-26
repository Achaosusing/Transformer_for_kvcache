#!/usr/bin/env python3
"""
Quick probe: inspect simulation JSON to understand message/tool structure.
Usage: python3 scripts/probe_sim_structure.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent
SIMS_DIR = ROOT / "tau2-bench" / "data" / "simulations"


def probe_file(path: Path, max_sims: int = 3) -> None:
    with open(path) as f:
        data = json.load(f)
    sims = data.get("simulations", [])
    print(f"\n=== {path.name}  ({len(sims)} simulations) ===")
    for sim in sims[:max_sims]:
        msgs = sim.get("messages", []) or []
        roles = [m.get("role") for m in msgs]
        reward = sim.get("reward_info", {})
        print(f"  sim_id={sim['id'][:8]}  task={sim.get('task_id')}  "
              f"msgs={len(msgs)}  reward={reward.get('reward', '?')}  "
              f"roles: {roles}")
        for i, m in enumerate(msgs):
            role = m.get("role", "?")
            tc = m.get("tool_calls") or []
            c = m.get("content")
            c_str = str(c) if c is not None else ""
            print(f"    [{i:02d}] {role:<12} tc={len(tc):>2}  "
                  f"content={len(c_str):>5}ch  preview={c_str[:70].replace(chr(10),' ')!r}")


def main() -> None:
    files = sorted(SIMS_DIR.glob("stress_baseline_1x30*.json"))[:2]
    if not files:
        files = sorted(SIMS_DIR.glob("*.json"))[:2]

    for f in files:
        probe_file(f, max_sims=2)


if __name__ == "__main__":
    main()
