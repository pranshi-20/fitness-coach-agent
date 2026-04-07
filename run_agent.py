import random
import os
from openai import OpenAI

from fitness_env.fitness_coach_env import FitnessCoachEnv
from fitness_env.schemas import FitnessAction
from fitness_env.evaluation import evaluate_overall_score

# ✅ Same env config as inference.py
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

ACTIONS = [
    "upper_body_strength",
    "lower_body_strength",
    "cardio_training",
    "recovery"
]


def smart_policy(observation):
    energy = observation.energy_level
    fatigue = observation.muscle_fatigue

    if energy < 30:
        chosen_action = "recovery"
    else:
        most_fatigued = max(fatigue, key=fatigue.get)
        if fatigue[most_fatigued] > 70:
            chosen_action = "recovery"
        else:
            least_fatigued = min(fatigue, key=fatigue.get)

            if least_fatigued == "upper_body":
                chosen_action = "upper_body_strength"
            elif least_fatigued == "lower_body":
                chosen_action = "lower_body_strength"
            else:
                chosen_action = "cardio_training"

    if observation.last_activity == chosen_action:
        return "recovery"

    return chosen_action


def openai_policy(observation):
    prompt = f"""
You are an AI fitness coach making decisions.

Goal: {observation.goal}
Energy: {observation.energy_level}
Fatigue: {observation.muscle_fatigue}
Injury Risk: {observation.injury_risk}
Last activity: {observation.last_activity}

Rules:
- Avoid overusing recovery
- Prefer training when energy is sufficient
- Follow goal alignment
- Avoid repeating same action

Return ONLY one:
upper_body_strength, lower_body_strength, cardio_training, recovery
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        action = response.choices[0].message.content.strip().lower()

        if action not in ACTIONS:
            return "recovery"

        return action

    except Exception:
        # ✅ fallback to smart policy (robust)
        return smart_policy(observation)


def run_episode(agent_type="smart"):
    env = FitnessCoachEnv()
    obs = env.reset()

    done = False

    print(f"\n--- Running {agent_type.upper()} AGENT ---\n")

    while not done:
        if agent_type == "random":
            action_type = random.choice(ACTIONS)
        elif agent_type == "smart":
            action_type = smart_policy(obs)
        else:
            action_type = openai_policy(obs)

        print(f"Day {obs.day} | Goal: {obs.goal} | Injury: {obs.injury_risk:.2f}")
        print(f"Energy: {obs.energy_level} | Fatigue: {obs.muscle_fatigue}")
        print(f"→ Action: {action_type}\n")

        action = FitnessAction(activity_type=action_type)
        obs, reward, done, _ = env.step(action)

    return env


if __name__ == "__main__":
    random_env = run_episode("random")
    smart_env = run_episode("smart")
    openai_env = run_episode("openai")

    print("\n--- FINAL SCORES ---\n")

    print("RANDOM:", evaluate_overall_score(random_env))
    print("SMART:", evaluate_overall_score(smart_env))
    print("OPENAI:", evaluate_overall_score(openai_env))