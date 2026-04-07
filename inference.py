"""
Inference Script — Fitness Coach Agent
=======================================
MANDATORY env vars:
    HF_TOKEN       Your HuggingFace token (used as API key)
    API_BASE_URL   LLM endpoint (default: HF router)
    MODEL_NAME     Model to use (default: Qwen/Qwen2.5-72B-Instruct)

STDOUT FORMAT (strictly followed):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""

import os
import sys
from openai import OpenAI
from fitness_env.fitness_coach_env import FitnessCoachEnv
from fitness_env.schemas import FitnessAction
from fitness_env.evaluation import evaluate_task_score

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN is not set.", file=sys.stderr)
    sys.exit(1)

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

ACTIONS = [
    "upper_body_strength",
    "lower_body_strength",
    "cardio_training",
    "recovery",
]

TASKS = [
    "recovery_balance",
    "energy_management",
    "training_distribution",
    "injury_management",
    "goal_alignment",
]

BENCHMARK = "fitness_coach"


# ── Policies ──────────────────────────────────────────────────────────────────

def _fallback_policy(observation) -> str:
    """Rule-based fallback — always produces a safe, goal-aligned action."""
    energy  = observation.energy_level
    fatigue = observation.muscle_fatigue
    goal    = observation.goal
    last    = observation.last_activity

    # ── Hard safety overrides ────────────────────────────────────────────
    # Trigger recovery earlier (threshold lowered from 75 → 55)
    if energy < 30:
        return "recovery"
    if max(fatigue.values()) > 55:
        return "recovery"
    if observation.injury_risk > 0.5:
        return "recovery"

    # ── Avoid double recovery ────────────────────────────────────────────
    # If we just recovered, force a training action
    if last == "recovery":
        if goal == "fat_loss":
            return "cardio_training"
        least = min(["upper_body", "lower_body"], key=lambda k: fatigue[k])
        return f"{least}_strength"

    # ── Goal alignment ───────────────────────────────────────────────────
    if goal == "fat_loss":
        # Prefer cardio but switch if cardio fatigue is high
        if fatigue["cardio"] > 50:
            return "recovery"
        return "cardio_training"

    elif goal == "endurance":
        # Alternate cardio and recovery
        if last == "cardio_training":
            return "recovery"
        return "cardio_training"

    else:  # muscle_gain
        # Pick least fatigued strength muscle
        least = min(["upper_body", "lower_body"], key=lambda k: fatigue[k])
        action = f"{least}_strength"
        # Avoid repeating same action twice in a row
        if action == last:
            other = "lower_body" if least == "upper_body" else "upper_body"
            if fatigue[other] <= 55:
                return f"{other}_strength"
            return "recovery"
        return action


def llm_policy(observation) -> str:
    """LLM policy with hard safety pre-check before calling the model."""
    energy  = observation.energy_level
    fatigue = observation.muscle_fatigue

    # ── Hard safety override BEFORE LLM call ────────────────────────────
    # Don't waste an API call when the safe action is obvious
    if (energy < 30 or
            max(fatigue.values()) > 55 or
            observation.injury_risk > 0.5):
        return _fallback_policy(observation)

    # Also skip LLM if last action was recovery (avoid double recovery)
    if observation.last_activity == "recovery":
        return _fallback_policy(observation)

    prompt = f"""You are an AI fitness coach making daily training decisions.

Current State:
- Day: {observation.day} / 7
- Goal: {observation.goal}
- Energy Level: {energy}/100
- Injury Risk: {observation.injury_risk:.2f}  (0=safe, 1=high risk)
- Muscle Fatigue: upper_body={fatigue['upper_body']}, lower_body={fatigue['lower_body']}, cardio={fatigue['cardio']}
- Last Activity: {observation.last_activity}

Decision Rules:
1. If energy < 30 OR any fatigue > 55 OR injury_risk > 0.5 → choose recovery
2. Do NOT repeat recovery twice in a row
3. Goal alignment:
   - fat_loss    → prefer cardio_training (if cardio fatigue <= 50)
   - muscle_gain → prefer upper_body_strength or lower_body_strength (pick LOWER fatigue one)
   - endurance   → alternate cardio_training and recovery
4. Avoid repeating the same action as last_activity

Available actions (respond with EXACTLY one word, nothing else):
upper_body_strength, lower_body_strength, cardio_training, recovery"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=20,
            temperature=0,
        )
        action = response.choices[0].message.content.strip().lower()
        action = action.split()[0] if action else "recovery"
        if action not in ACTIONS:
            return _fallback_policy(observation)
        return action
    except Exception:
        return _fallback_policy(observation)


# ── Episode runner ────────────────────────────────────────────────────────────

def run_task(task_name: str) -> float:
    env = FitnessCoachEnv(task=task_name)
    obs = env.reset()

    done    = False
    step    = 0
    rewards = []

    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    while not done:
        action_str = llm_policy(obs)
        action = FitnessAction(activity_type=action_str)

        obs, reward, done, info = env.step(action)
        rewards.append(reward)

        print(
            f"[STEP] step={step} action={action_str} "
            f"reward={reward:.2f} done={str(done).lower()} error=null",
            flush=True,
        )
        step += 1

    score       = evaluate_task_score(env, task_name)
    success     = score >= 0.5
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={step} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )
    print(flush=True)

    return score


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_scores = []
    for task in TASKS:
        score = run_task(task)
        all_scores.append(score)