"""Mandatory baseline evaluation script for the OpenEnv Hackathon.

Runs an LLM agent against the CloudFinOps environment through /reset and /step.
Uses the `openai` SDK and the following MANDATORY environment variables:

  API_BASE_URL  — The API endpoint for the LLM (e.g. https://router.huggingface.co/v1)
  MODEL_NAME    — The model identifier to use for inference (e.g. gpt-4o)
  HF_TOKEN      — Your Hugging Face / API key

Participants must use OpenAI Client for all LLM calls using above variables.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

# Auto-load .env file if present (judges can use .env.example as a template)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv is optional; env vars can be exported in shell instead

import httpx
from openai import OpenAI

# ---------------------------------------------------------------------------
# MANDATORY Environment Variables (per hackathon rules)
# 
#   export API_BASE_URL="https://router.huggingface.co/v1"
#   export MODEL_NAME="gpt-4o"
#   export HF_TOKEN="hf_..."
# ---------------------------------------------------------------------------
API_BASE_URL: str = os.environ.get("API_BASE_URL", "")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")

# CloudFinOps environment URL (local Docker or HF Space)
ENV_BASE_URL: str = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

# Evaluation parameters
MAX_STEPS: int = 10
TASKS: List[str] = ["easy", "medium", "hard"]


def _validate_env() -> None:
    """Ensure all mandatory environment variables are set before proceeding."""
    missing: List[str] = []
    if not API_BASE_URL:
        missing.append("API_BASE_URL")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")
    if not HF_TOKEN:
        missing.append("HF_TOKEN")

    if missing:
        print("\n  ❌ ERROR: Missing mandatory environment variables:")
        for var in missing:
            print(f"     • {var}")
        print("\n  Please set them before running:")
        print('     export API_BASE_URL="https://router.huggingface.co/v1"')
        print('     export MODEL_NAME="gpt-4o"')
        print('     export HF_TOKEN="hf_your_token_here"')
        print()
        sys.exit(1)


# ---------------------------------------------------------------------------
# OpenAI Client — mandatory per hackathon rules
# All LLM calls go through this client using the three env vars above.
# ---------------------------------------------------------------------------
_validate_env()

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# HTTP client for environment REST calls
http = httpx.Client(timeout=60.0)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are an expert cloud infrastructure engineer managing a fleet of servers.
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
- Only one action per step. Choose the most impactful action.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {"command": "IGNORE", "target_id": None, "reply": ""}


def run_task(task_id: str) -> float:
    """Run a single task against the environment and return the grader score."""
    print(f"\n{'=' * 60}")
    print(f"  Task: {task_id.upper()}")
    print(f"{'=' * 60}")

    # Reset environment
    resp = http.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()

    for step_num in range(1, MAX_STEPS + 1):
        obs_json = json.dumps(obs, indent=2)
        budget = obs.get("budget_remaining", 0)
        traffic = obs.get("traffic_load", 0)
        print(f"\n--- Step {step_num} ---")
        print(f"  Budget: ${budget:.2f}  |  Traffic: {traffic}%")

        # Ask LLM for action via OpenAI Client (mandatory)
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Current observation:\n{obs_json}\n\n"
                            "Choose your next action (respond with JSON only):"
                        ),
                    },
                ],
                temperature=0.1,
                max_tokens=300,
            )
            raw_reply = completion.choices[0].message.content or ""
        except Exception as exc:
            print(f"  [LLM Error] {exc} — sending IGNORE")
            raw_reply = '{"command": "IGNORE", "target_id": null, "reply": ""}'

        action = parse_action(raw_reply)
        cmd = action.get("command", "IGNORE")
        target = action.get("target_id", "N/A")
        print(f"  Action: {cmd} -> {target}")

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

    print("\n  ⚠️  Max steps reached.")
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    start_time = time.time()

    print("=" * 60)
    print("  CloudFinOps-Env Baseline Evaluator")
    print("=" * 60)
    print(f"  Model:     {MODEL_NAME}")
    print(f"  API:       {API_BASE_URL}")
    print(f"  HF_TOKEN:  {'*' * 4}{HF_TOKEN[-4:] if len(HF_TOKEN) > 4 else '****'}")
    print(f"  Env:       {ENV_BASE_URL}")
    print(f"  Max Steps: {MAX_STEPS}")

    scores: Dict[str, float] = {}
    for task_id in TASKS:
        try:
            scores[task_id] = run_task(task_id)
        except Exception as exc:
            print(f"  ❌ Task '{task_id}' failed: {exc}")
            scores[task_id] = 0.0

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("  FINAL RESULTS")
    print(f"{'=' * 60}")
    for tid, score in scores.items():
        status = "✅" if score > 0.0 else "❌"
        print(f"  {status} {tid:>8s}: {score:.4f}")
    avg = sum(scores.values()) / len(scores) if scores else 0.0
    print(f"  {'AVERAGE':>10s}: {avg:.4f}")
    print(f"  {'TIME':>10s}: {elapsed:.1f}s")
    print(f"{'=' * 60}")

    # Validate all scores are in [0.0, 1.0]
    for tid, score in scores.items():
        assert 0.0 <= score <= 1.0, f"Score for {tid} out of range: {score}"

    print("\n  ✅ All scores within valid 0.0–1.0 range.")


if __name__ == "__main__":
    main()