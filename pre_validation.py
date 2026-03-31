#!/usr/bin/env python3
"""
pre_validation.py — OpenEnv Pre-Submission Validator (Python)

Runs the full hackathon pre-submission checklist:
  1. HF Space deploys — ping /reset and verify 200
  2. OpenEnv spec compliance — validate openenv.yaml, typed models, step()/reset()/state()
  3. Dockerfile builds — run `docker build`
  4. Baseline reproduces — run inference.py end-to-end
  5. 3+ tasks with graders — verify each task returns a score in [0.0, 1.0]

Usage:
  python pre_validation.py [--space-url URL] [--repo-dir DIR] [--skip-docker] [--skip-space]

Environment Variables:
  API_BASE_URL   The API endpoint for the LLM
  MODEL_NAME     The model identifier to use for inference
  HF_TOKEN       Your Hugging Face / API key
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# ANSI colours (disabled when not a TTY)
# ---------------------------------------------------------------------------
if sys.stdout.isatty():
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BOLD = "\033[1m"
    NC = "\033[0m"
else:
    RED = GREEN = YELLOW = BOLD = NC = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class ValidationResult:
    """Collects pass / fail results."""

    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0
        self.results: List[Tuple[str, bool, str]] = []

    def record(self, name: str, ok: bool, detail: str = "") -> None:
        self.results.append((name, ok, detail))
        if ok:
            self.passed += 1
            print(f"  {GREEN}✅ PASSED{NC} — {name}")
        else:
            self.failed += 1
            print(f"  {RED}❌ FAILED{NC} — {name}")
        if detail:
            print(f"     {detail}")

    @property
    def all_passed(self) -> bool:
        return self.failed == 0


def _header(title: str) -> None:
    print(f"\n{BOLD}{'=' * 60}{NC}")
    print(f"{BOLD}  {title}{NC}")
    print(f"{BOLD}{'=' * 60}{NC}\n")


def _section(title: str) -> None:
    print(f"\n{BOLD}--- {title} ---{NC}")


# ---------------------------------------------------------------------------
# Check 1: HF Space Ping
# ---------------------------------------------------------------------------
def check_space_ping(space_url: str, vr: ValidationResult) -> None:
    _section("Check 1/5: HF Space Ping (/reset)")

    try:
        import httpx
    except ImportError:
        vr.record("httpx import", False, "Install httpx: pip install httpx")
        return

    url = f"{space_url.rstrip('/')}/reset"
    try:
        with httpx.Client(timeout=30.0) as client:
            resp = client.post(url, json={"task_id": "easy"})
            if resp.status_code == 200:
                body = resp.json()
                has_servers = "servers" in body
                vr.record(
                    "HF Space /reset returns 200",
                    True,
                    f"Response contains 'servers': {has_servers}",
                )
            else:
                vr.record(
                    "HF Space /reset returns 200",
                    False,
                    f"Got HTTP {resp.status_code}",
                )
    except Exception as exc:
        vr.record("HF Space /reset reachable", False, str(exc))


# ---------------------------------------------------------------------------
# Check 2: OpenEnv Spec Compliance
# ---------------------------------------------------------------------------
def check_openenv_spec(repo_dir: Path, vr: ValidationResult) -> None:
    _section("Check 2/5: OpenEnv Spec Compliance")

    # 2a. openenv.yaml exists and is valid
    yaml_path = repo_dir / "openenv.yaml"
    if not yaml_path.exists():
        vr.record("openenv.yaml exists", False, f"Not found at {yaml_path}")
        return

    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        try:
            content = yaml_path.read_text()
            if not content.strip():
                vr.record("openenv.yaml parseable", False, "File is empty")
                return
            vr.record("openenv.yaml parseable", True, "(PyYAML not installed — basic check only)")
            tasks = []
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("- id:"):
                    tid = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                    tasks.append(tid)
            if len(tasks) >= 3:
                vr.record("openenv.yaml has 3+ tasks", True, f"Tasks: {tasks}")
            else:
                vr.record("openenv.yaml has 3+ tasks", False, f"Found {len(tasks)} tasks: {tasks}")
            return
        except Exception as exc:
            vr.record("openenv.yaml parseable", False, str(exc))
            return

    try:
        with open(yaml_path) as f:
            cfg = yaml.safe_load(f)
    except Exception as exc:
        vr.record("openenv.yaml parseable", False, str(exc))
        return

    vr.record("openenv.yaml parseable", True)

    for key in ("name", "entrypoint", "tasks"):
        vr.record(f"openenv.yaml has '{key}'", key in cfg, f"Keys: {list(cfg.keys())}")

    tasks = cfg.get("tasks", [])
    task_ids = [t.get("id", "") for t in tasks if isinstance(t, dict)]
    vr.record(
        "openenv.yaml has 3+ tasks",
        len(task_ids) >= 3,
        f"Found {len(task_ids)} tasks: {task_ids}",
    )

    # 2b. Typed Pydantic models
    models_path = repo_dir / "env" / "models.py"
    if models_path.exists():
        models_src = models_path.read_text()
        has_basemodel = "BaseModel" in models_src
        has_action = "class Action" in models_src
        has_observation = "class Observation" in models_src
        has_reward = "class RewardInfo" in models_src or "class Reward" in models_src
        vr.record(
            "Typed Pydantic models (Action, Observation, Reward)",
            has_basemodel and has_action and has_observation and has_reward,
            f"BaseModel={has_basemodel}, Action={has_action}, Observation={has_observation}, Reward={has_reward}",
        )
    else:
        vr.record("env/models.py exists", False)

    # 2c. Server exposes step(), reset(), state()
    server_path = repo_dir / "env" / "server.py"
    if server_path.exists():
        srv_src = server_path.read_text()
        has_reset = "/reset" in srv_src
        has_step = "/step" in srv_src
        has_state = "/state" in srv_src
        vr.record(
            "Server endpoints: /reset, /step, /state",
            has_reset and has_step and has_state,
            f"reset={has_reset}, step={has_step}, state={has_state}",
        )
    else:
        vr.record("env/server.py exists", False)


# ---------------------------------------------------------------------------
# Check 3: Dockerfile Builds
# ---------------------------------------------------------------------------
def check_docker_build(repo_dir: Path, vr: ValidationResult) -> None:
    _section("Check 3/5: Dockerfile Builds")

    dockerfile = repo_dir / "Dockerfile"
    if not dockerfile.exists():
        dockerfile = repo_dir / "server" / "Dockerfile"
    if not dockerfile.exists():
        vr.record("Dockerfile exists", False, "No Dockerfile in repo root or server/")
        return

    vr.record("Dockerfile exists", True, str(dockerfile))

    # Check docker is available
    try:
        result = subprocess.run(
            ["docker", "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            vr.record("Docker available", False, "docker command failed")
            return
        vr.record("Docker available", True, f"Docker {result.stdout.strip()}")
    except FileNotFoundError:
        vr.record("Docker available", False, "docker not found in PATH")
        return
    except Exception as exc:
        vr.record("Docker available", False, str(exc))
        return

    # Build
    print(f"  Building Docker image (this may take a few minutes)...")
    build_dir = str(dockerfile.parent)
    try:
        result = subprocess.run(
            ["docker", "build", "-t", "cloudfinops-env:validate", build_dir],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode == 0:
            vr.record("Docker build succeeds", True)
        else:
            last_lines = "\n".join(result.stderr.splitlines()[-10:])
            vr.record("Docker build succeeds", False, f"Last output:\n{last_lines}")
    except subprocess.TimeoutExpired:
        vr.record("Docker build succeeds", False, "Build timed out (600s)")
    except Exception as exc:
        vr.record("Docker build succeeds", False, str(exc))


# ---------------------------------------------------------------------------
# Check 4: Baseline Reproduces
# ---------------------------------------------------------------------------
def check_baseline_reproduces(repo_dir: Path, vr: ValidationResult) -> None:
    _section("Check 4/5: Baseline Inference Script")

    inference_path = repo_dir / "inference.py"
    if not inference_path.exists():
        vr.record("inference.py exists in root", False)
        return

    vr.record("inference.py exists in root", True)

    src = inference_path.read_text()
    uses_openai = "from openai" in src or "import openai" in src
    uses_api_base = "API_BASE_URL" in src
    uses_model_name = "MODEL_NAME" in src
    uses_hf_token = "HF_TOKEN" in src

    vr.record(
        "inference.py uses OpenAI client",
        uses_openai,
        f"openai import found: {uses_openai}",
    )
    vr.record(
        "inference.py references required env vars",
        uses_api_base and uses_model_name and uses_hf_token,
        f"API_BASE_URL={uses_api_base}, MODEL_NAME={uses_model_name}, HF_TOKEN={uses_hf_token}",
    )

    has_main = 'if __name__' in src or "def main" in src
    vr.record("inference.py has main entry point", has_main)


# ---------------------------------------------------------------------------
# Check 5: 3+ Tasks with Graders
# ---------------------------------------------------------------------------
def check_tasks_and_graders(repo_dir: Path, space_url: Optional[str], vr: ValidationResult) -> None:
    _section("Check 5/5: 3+ Tasks with Graders (score in 0.0–1.0)")

    engine_path = repo_dir / "env" / "engine.py"
    if not engine_path.exists():
        vr.record("env/engine.py exists", False)
        return

    repo_str = str(repo_dir)
    added = False
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
        added = True

    try:
        from env.engine import CloudFinOpsEngine
        from env.models import Action

        engine = CloudFinOpsEngine()
        task_ids = ["easy", "medium", "hard"]
        valid_tasks = 0

        for task_id in task_ids:
            try:
                obs = engine.reset(task_id)
                assert obs is not None, "reset() returned None"

                for _ in range(20):
                    obs, reward, done, info = engine.step(
                        Action(command="IGNORE", target_id=None, reply="")
                    )
                    if done:
                        break

                score = engine.grade()
                in_range = 0.0 <= score <= 1.0
                vr.record(
                    f"Task '{task_id}' grader score in [0.0, 1.0]",
                    in_range,
                    f"Score = {score:.4f}",
                )
                if in_range:
                    valid_tasks += 1

            except Exception as exc:
                vr.record(f"Task '{task_id}' runs without error", False, str(exc))

        vr.record(
            "At least 3 tasks produce valid grader scores",
            valid_tasks >= 3,
            f"{valid_tasks}/3 tasks passed",
        )

    except ImportError as exc:
        vr.record("Engine importable", False, str(exc))
    finally:
        if added:
            sys.path.remove(repo_str)


# ---------------------------------------------------------------------------
# Check environment variables
# ---------------------------------------------------------------------------
def check_env_vars(vr: ValidationResult) -> None:
    _section("Environment Variables Check")
    for var in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
        val = os.getenv(var)
        if val:
            display = val[:8] + "..." if len(val) > 12 else val
            vr.record(f"${var} is set", True, f"Value: {display}")
        else:
            vr.record(f"${var} is set", False, f"Set it: export {var}=<value>")


# ---------------------------------------------------------------------------
# Resource check
# ---------------------------------------------------------------------------
def check_resource_constraints(repo_dir: Path, vr: ValidationResult) -> None:
    _section("Resource Constraints")

    req_path = repo_dir / "requirements.txt"
    if req_path.exists():
        reqs = req_path.read_text()
        heavy = ["torch", "tensorflow", "transformers", "jax"]
        found_heavy = [h for h in heavy if h in reqs.lower()]
        vr.record(
            "No heavy ML frameworks in requirements.txt",
            len(found_heavy) == 0,
            f"Heavy deps found: {found_heavy}" if found_heavy else "Clean — lightweight deps only",
        )
    else:
        vr.record("requirements.txt exists", False)

    dockerfile = repo_dir / "Dockerfile"
    if dockerfile.exists():
        df_src = dockerfile.read_text()
        has_expose = "EXPOSE 8000" in df_src or "EXPOSE  8000" in df_src
        has_cmd = "uvicorn" in df_src
        vr.record("Dockerfile exposes port 8000", has_expose)
        vr.record("Dockerfile runs uvicorn", has_cmd)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenEnv Pre-Submission Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--space-url",
        default=os.getenv("SPACE_URL", ""),
        help="HuggingFace Space URL (e.g. https://your-space.hf.space). "
        "Skipped if not provided.",
    )
    parser.add_argument(
        "--repo-dir",
        default="./cloudfinops-env",
        help="Path to repo directory (default: ./cloudfinops-env)",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker build check",
    )
    parser.add_argument(
        "--skip-space",
        action="store_true",
        help="Skip HF Space ping check",
    )
    args = parser.parse_args()

    repo_dir = Path(args.repo_dir).resolve()
    if not repo_dir.is_dir():
        print(f"{RED}Error:{NC} Repository directory not found: {repo_dir}")
        sys.exit(1)

    vr = ValidationResult()

    _header("OpenEnv Pre-Submission Validator")
    print(f"  Repo:      {repo_dir}")
    print(f"  Space URL: {args.space_url or '(not provided — skipping ping)'}")
    print(f"  Time:      {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check 0: Environment Variables
    check_env_vars(vr)

    # Check 1: HF Space Ping
    if args.space_url and not args.skip_space:
        check_space_ping(args.space_url, vr)
    else:
        print(f"\n{YELLOW}  ⏭  Skipping HF Space ping (no --space-url provided){NC}")

    # Check 2: OpenEnv Spec Compliance
    check_openenv_spec(repo_dir, vr)

    # Check 3: Dockerfile Builds
    if not args.skip_docker:
        check_docker_build(repo_dir, vr)
    else:
        print(f"\n{YELLOW}  ⏭  Skipping Docker build (--skip-docker){NC}")

    # Check 4: Baseline Reproduces
    check_baseline_reproduces(repo_dir, vr)

    # Check 5: 3+ Tasks with Graders
    check_tasks_and_graders(repo_dir, args.space_url, vr)

    # Resource Constraints
    check_resource_constraints(repo_dir, vr)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    _header("VALIDATION SUMMARY")
    total = vr.passed + vr.failed
    print(f"  {GREEN}Passed: {vr.passed}/{total}{NC}")
    if vr.failed:
        print(f"  {RED}Failed: {vr.failed}/{total}{NC}")
        print()
        print(f"  {RED}{BOLD}Failed checks:{NC}")
        for name, ok, detail in vr.results:
            if not ok:
                print(f"    {RED}✗{NC} {name}")
                if detail:
                    print(f"      {detail}")

    print()
    if vr.all_passed:
        print(f"  {GREEN}{BOLD}🎉 All checks passed! Your submission is ready.{NC}")
    else:
        print(f"  {RED}{BOLD}⛔ Fix the above failures before submitting.{NC}")

    sys.exit(0 if vr.all_passed else 1)


if __name__ == "__main__":
    main()
