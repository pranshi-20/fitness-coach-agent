"""
Fitness Coach Agent — HF Space App

Supports:
1. OpenEnv API endpoints (/reset, /step)
2. Gradio UI for manual interaction
"""

from fastapi import FastAPI
from pydantic import BaseModel
import gradio as gr

from fitness_env.fitness_coach_env import FitnessCoachEnv
from fitness_env.schemas import FitnessAction

# ─────────────────────────────────────────────────────────────
# FASTAPI (for OpenEnv evaluator)
# ─────────────────────────────────────────────────────────────

app = FastAPI()

env = FitnessCoachEnv()


class ActionRequest(BaseModel):
    activity_type: str


@app.get("/")
def health():
    return {"status": "Fitness Coach Agent running ✅"}


@app.post("/reset")
def reset():
    obs = env.reset()
    return obs.dict()


@app.post("/step")
def step(action: ActionRequest):
    act = FitnessAction(activity_type=action.activity_type)
    obs, reward, done, _ = env.step(act)

    return {
        "observation": obs.dict(),
        "reward": reward,
        "done": done,
    }


# ─────────────────────────────────────────────────────────────
# GRADIO UI (for demo + judges visual)
# ─────────────────────────────────────────────────────────────

def run_step(action):
    global env
    act = FitnessAction(activity_type=action)
    obs, reward, done, _ = env.step(act)

    return f"""
Day: {obs.day}
Goal: {obs.goal}
Energy: {obs.energy_level}
Fatigue: {obs.muscle_fatigue}
Injury Risk: {obs.injury_risk}

Reward: {reward}
Done: {done}
"""


def reset_env():
    global env
    obs = env.reset()
    return f"""
Environment Reset ✅

Goal: {obs.goal}
Energy: {obs.energy_level}
Fatigue: {obs.muscle_fatigue}
"""


with gr.Blocks() as demo:
    gr.Markdown("# 🏋️‍♀️ Fitness Coach Agent")

    with gr.Row():
        action_dropdown = gr.Dropdown(
            [
                "upper_body_strength",
                "lower_body_strength",
                "cardio_training",
                "recovery",
            ],
            label="Action",
        )

    output = gr.Textbox(label="Output")

    with gr.Row():
        step_btn = gr.Button("Step")
        reset_btn = gr.Button("Reset")

    step_btn.click(run_step, inputs=action_dropdown, outputs=output)
    reset_btn.click(reset_env, outputs=output)


# Mount Gradio app into FastAPI
app = gr.mount_gradio_app(app, demo, path="/ui")