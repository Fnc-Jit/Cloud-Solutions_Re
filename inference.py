"""Mandatory baseline evaluation script for the OpenEnv Hackathon.

Runs an LLM agent against the CloudFinOps environment through /reset and /step.
Uses the `openai` SDK and environment variables:
  - OPENAI_API_KEY (or HF_TOKEN) for authentication
  - API_BASE_URL for the inference endpoint
  - MODEL_NAME for the model to evaluate
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "tgi")
API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN", "dummy")
MAX_STEPS = 10

client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
http = httpx.Client(timeout=30.0)

TASKS = ["easy", "medium", "hard"]

SYSTEM_PROMPT = """You are an expert cloud infrastructure engineer managing a fleet of servers.
Your goal is to optimize costs while preventing SLA breaches (any server hitting 100% CPU).

You will receive a JSON observation containing:
- servers: list of server objects with id, type, cpu_util, memory_util, cost_per_hour, status
- traffic_load: global traffic percentage
- spike_detected: whether a traffic spike is happening
- incidents: list of past incidents
- budget_remaining: remaining budget in USD
- inbox: messages from stakeholders

You MUST respond with ONLY a valid JSON object matching this schema:
{
  "command": "UPSCALE" | "DOWNSCALE" | "TERMINATE" | "REDISTRIBUTE_LOAD" | "IGNORE",
  "target_id": "<server_id or null>",
  "reply": "<optional text reply to inbox messages>"
}

Rules:
- TERMINATE idle (0% CPU) servers to save money.
- DOWNSCALE over-provisioned servers (very low CPU) carefully — it increases their load.
- UPSCALE servers approaching 100% CPU to prevent SLA breach (takes effect next step).
- Never let any running server reach 100% CPU.
- Respond to inbox messages concisely in the reply field.
"""


def parse_action(raw: str) -> Dict[str, Any]:
    """Extract a JSON action from LLM output, handling markdown fences."""
    text = raw.strip()
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                text = part
                break
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find first { ... } block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(text[start:end])
        return {"command": "IGNORE", "target_id": None, "reply": ""}


def run_task(task_id: str) -> float:
    """Run a single task and return the grader score."""
    print(f"\n{'='*60}")
    print(f"  Task: {task_id.upper()}")
    print(f"{'='*60}")

    # Reset environment
    resp = http.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()

    for step_num in range(1, MAX_STEPS + 1):
        obs_json = json.dumps(obs, indent=2)
        print(f"\n--- Step {step_num} ---")
        print(f"  Budget: ${obs.get('budget_remaining', '?'):.2f}  |  Traffic: {obs.get('traffic_load', '?')}%")

        # Ask LLM for action
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Current observation:\n{obs_json}\n\nChoose your action:"},
                ],
                temperature=0.1,
                max_tokens=300,
            )
            raw_reply = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [LLM Error] {exc} — sending IGNORE")
            raw_reply = '{"command": "IGNORE", "target_id": null, "reply": ""}'

        action = parse_action(raw_reply)
        print(f"  Action: {action.get('command', 'IGNORE')} -> {action.get('target_id', 'N/A')}")

        # Step environment
        resp = http.post(f"{ENV_BASE_URL}/step", json=action)
        resp.raise_for_status()
        result = resp.json()
        obs = result["observation"]
        done = result["done"]
        reward = result["reward"]

        print(f"  Reward: {reward}  |  Done: {done}")

        if done:
            score = result.get("info", {}).get("grader_score", 0.0)
            print(f"\n  ✅ FINAL SCORE: {score}")
            return score

    # If max steps reached without done, get final state
    print(f"\n  ⚠️ Max steps reached.")
    return 0.0


def main() -> None:
    print("=" * 60)
    print("  CloudFinOps-Env Baseline Evaluator")
    print("=" * 60)
    print(f"  Model: {MODEL_NAME}")
    print(f"  API:   {API_BASE_URL}")
    print(f"  Env:   {ENV_BASE_URL}")

    scores: Dict[str, float] = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as exc:
            print(f"  ❌ Task '{task_id}' failed: {exc}")
            scores[task_id] = 0.0

    print(f"\n{'='*60}")
    print("  FINAL RESULTS")
    print(f"{'='*60}")
    for tid, score in scores.items():
        print(f"  {tid:>8s}: {score:.4f}")
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':>8s}: {avg:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
