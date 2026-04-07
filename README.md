---
title: Fitness Coach Agent
emoji: 🏋️‍♀️
colorFrom: blue
colorTo: green
sdk: docker
python_version: "3.10"
app_file: app.py
tags:
  - openenv
  - reinforcement-learning
  - fitness
  - health
---

# 🏋️‍♀️ Adaptive Fitness Coach Agent — OpenEnv RL Environment

A goal-conditioned, risk-aware reinforcement learning environment where an AI agent learns to plan weekly workouts by balancing energy, fatigue, injury risk, and long-term training consistency.

Built for the **Meta PyTorch × Hugging Face OpenEnv Hackathon**.

---

## 🚀 Overview

Fitness planning is inherently dynamic — energy fluctuates, fatigue accumulates, and poor decisions lead to injury or burnout. Static workout plans fail to adapt.

This environment models a real-world adaptive fitness coaching system where an agent must make daily training decisions under multiple competing constraints across a 7-day cycle.

**What makes this environment non-trivial:**
- Goal-conditioned behaviour (fat loss, muscle gain, endurance) — same action can be good or bad depending on the goal
- Injury risk accumulates from overtraining and can only be reduced by recovery
- Double recovery is penalised — the agent must balance rest and activity
- Consistency bonus after day 5 rewards varied, sustained training patterns
- Multi-objective reward: the agent cannot just optimise one metric

---

## 🧠 Environment Design

### 📥 Observation Space

```json
{
  "day": 3,
  "goal": "muscle_gain",
  "energy_level": 70,
  "injury_risk": 0.20,
  "muscle_fatigue": {
    "upper_body": 25,
    "lower_body": 40,
    "cardio": 10
  },
  "last_activity": "lower_body_strength"
}
```

| Field | Type | Range | Description |
|---|---|---|---|
| `day` | int | 0–7 | Current day in the weekly cycle |
| `goal` | str | 3 values | fat_loss / muscle_gain / endurance |
| `energy_level` | int | 0–100 | Current energy (depleted by training, restored by recovery) |
| `injury_risk` | float | 0.0–1.0 | Accumulated injury risk from overtraining |
| `muscle_fatigue` | dict | 0–100 each | Per-group fatigue: upper_body, lower_body, cardio |
| `last_activity` | str | nullable | Previous action taken |

### 🎯 Action Space

| Action | Energy Cost | Fatigue Effect | Use Case |
|---|---|---|---|
| `upper_body_strength` | −15 | upper_body +25 | Muscle gain, upper focus |
| `lower_body_strength` | −15 | lower_body +25 | Muscle gain, lower focus |
| `cardio_training` | −10 | cardio +20 | Fat loss, endurance |
| `recovery` | +20 | all groups −15 | Rest, injury prevention |

---

## 🏆 Reward Function

The reward function provides **dense signal across the full trajectory** — not just at episode end.

### Positive rewards (per step)
- `+0.5` — training action aligned with goal (e.g. cardio for fat_loss)
- `+0.3` — cardio for endurance goal
- `+0.2` — recovery for endurance goal
- `+0.3` — consistency bonus after day 5 (if last 5 actions had ≥3 unique types)

### Penalties (per step)
- `−injury_risk` — continuous penalty proportional to current risk level
- `−0.3` — energy below 30
- `−0.3` — taking recovery twice in a row (lazy agent penalty)

---

## 🧪 Tasks & Graders

Five tasks with deterministic graders, covering easy → medium → hard difficulty:

| # | Task | Difficulty | Goal Set | Grader Logic |
|---|---|---|---|---|
| 1 | `recovery_balance` | Easy | muscle_gain | Penalises any muscle fatigue > 80; rewards moderate total fatigue |
| 2 | `energy_management` | Easy | endurance | Scores based on final energy level (higher = better) |
| 3 | `training_distribution` | Medium | muscle_gain | Measures fatigue spread across muscle groups (lower spread = better) |
| 4 | `injury_management` | Medium | fat_loss | Scores based on final injury risk (lower = better) |
| 5 | `goal_alignment` | Hard | fat_loss | Checks if dominant fatigue group matches the stated goal |

All graders return scores in **[0.0, 1.0]** and are fully deterministic.

### Baseline Scores (LLM agent — Qwen/Qwen2.5-7B-Instruct)

| Task | Score | Pass (≥0.5) |
|---|---|---|
| recovery_balance | 0.60 | ✅ |
| energy_management | 1.00 | ✅ |
| training_distribution | 0.00 | ❌ |
| injury_management | 1.00 | ✅ |
| goal_alignment | 1.00 | ✅ |
| **Overall** | **0.72** | **4/5** |

---

## 🌐 API Endpoints

The HF Space exposes a REST API for programmatic evaluation:

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/reset` | POST | Reset environment, returns initial observation |
| `/step` | POST | Take action, returns observation + reward + done |
| `/state` | GET | Returns current environment state |
| `/ui` | GET | Gradio demo interface |

**Example usage:**
```bash
# Reset
curl -X POST https://prgarg-fitness-coach-agent.hf.space/reset

# Step
curl -X POST https://prgarg-fitness-coach-agent.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"activity_type": "cardio_training"}'
```

---

## 🏗️ Project Structure

```
fitness-coach-agent/
│
├── fitness_env/
│   ├── __init__.py
│   ├── fitness_coach_env.py   # Core environment
│   ├── schemas.py             # Pydantic models
│   └── evaluation.py         # Task graders
│
├── app.py                     # FastAPI + Gradio server
├── inference.py               # Baseline inference script
├── run_agent.py               # Random / Smart / LLM agents
├── openenv.yaml               # OpenEnv spec metadata
├── Dockerfile                 # Container definition
├── requirements.txt
└── README.md
```

---

## ▶️ How to Run

### Local setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run the server
```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

### Run baseline inference
```bash
export HF_TOKEN=your_token_here
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

python inference.py
```

### Run all agents
```bash
python run_agent.py
```

### Validate OpenEnv compliance
```bash
openenv validate
```

### Docker
```bash
docker build -t fitness-coach-agent .
docker run -p 7860:7860 \
  -e HF_TOKEN=your_token \
  -e MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
  fitness-coach-agent
```

---

## 🧩 OpenEnv Compliance

| Requirement | Status |
|---|---|
| Typed Pydantic models (Observation, Action) | ✅ |
| `step()` / `reset()` / `state()` implemented | ✅ |
| `openenv.yaml` with metadata | ✅ |
| Minimum 3 tasks with graders (0.0–1.0) | ✅ 5 tasks |
| Meaningful reward function (dense signal) | ✅ |
| Baseline `inference.py` using OpenAI client | ✅ |
| HF Space deployed with `openenv` tag | ✅ |
| Working Dockerfile | ✅ |
| `openenv validate` passes | ✅ |

---

## 👩‍💻 Author

**Pranshi Garg** — Built for the Meta PyTorch × Hugging Face OpenEnv Hackathon