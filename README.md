# ☁️ CloudFinOps-Env

> **Meta AI & HuggingFace OpenEnv Hackathon Submission**

A Reinforcement Learning environment that simulates real-world **cloud cost optimization** combined with **SLA incident management** — complete with human-in-the-loop text-chat interactions.

---

## 🎯 What Makes This Different?

Most RL environments tackle *either* cost optimization *or* incident response. **CloudFinOps-Env** forces the agent to handle **both simultaneously** with conflicting objectives:

| Objective | Tension |
|-----------|---------|
| 💰 Cut costs | ↔ Risk overloading remaining servers |
| 🛡️ Prevent SLA breaches | ↔ Requires expensive upscaling |
| ⏱️ Scale up for traffic | ↔ Scaling is **delayed by 1 step** |
| 💬 Follow human instructions | ↔ Humans may give risky directives |

## 🏗️ Environment Mechanics

### Physics Engine
- **Delayed Scaling**: `UPSCALE` takes effect on the *next* step — the agent must anticipate, not react.
- **Load Redistribution**: Terminating a server redistributes its load across survivors.
- **SLA Breach = Catastrophe**: Any server hitting 100% CPU triggers an instant SLA failure with heavy penalty.
- **Budget Drain**: Each running server charges per-step costs. Overspending ends the episode.

### Action Space
| Command | Effect |
|---------|--------|
| `UPSCALE` | Doubles capacity (delayed 1 step), doubles cost |
| `DOWNSCALE` | Halves cost immediately, increases CPU load by 1.8× |
| `TERMINATE` | Removes server, redistributes load |
| `REDISTRIBUTE_LOAD` | Balances CPU evenly across all running servers |
| `IGNORE` | No-op |

Plus a **`reply`** field for text communication with stakeholders.

### Observation Space
- List of `ServerState` (id, type, cpu_util, memory_util, cost_per_hour, status)
- `traffic_load`, `spike_detected`, `incidents`, `budget_remaining`
- **`inbox`**: Text messages from simulated humans (CTO, Marketing, Ops)

## 🎮 The 3 Tasks

### 🟢 Easy — *Zombie Cleanup*
10 servers, 3 are completely idle. Terminate the zombies without touching active ones.

### 🟡 Medium — *The CTO Budget Squeeze*
12 over-provisioned servers at ~5% CPU. The CTO demands 50% cost reduction. Downscale smartly without triggering overload.

### 🔴 Hard — *Black Friday Chaos*
Traffic spike incoming! DBs at 85% CPU and climbing. Upscale critical databases, shut down batch jobs, manage a shrinking budget — all while preventing SLA failures.

**Grading**: `score = (uptime × 0.6) + (cost_efficiency × 0.4)`

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the environment server
uvicorn env.server:app --host 0.0.0.0 --port 8000

# In another terminal, run the baseline evaluator
python inference.py
```

### Docker
```bash
docker build -t cloudfinops-env .
docker run -p 8000:8000 cloudfinops-env
```

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | `{"task_id": "easy\|medium\|hard"}` → Initial Observation |
| `POST` | `/step` | Send Action → `{observation, reward, done, info}` |
| `GET`  | `/state` | Current Observation (no state advance) |

## 📐 Scoring

All graders output scores between **0.0** and **1.0**:

- **Easy**: 1.0 if all idle servers terminated cleanly, penalty for killing active ones
- **Medium**: Proportional to cost savings vs 50% target, penalty for SLA breach
- **Hard**: `(uptime_score × 0.6) + (cost_efficiency × 0.4)`

---

Built with ❤️ for the Meta AI & HuggingFace OpenEnv Hackathon
