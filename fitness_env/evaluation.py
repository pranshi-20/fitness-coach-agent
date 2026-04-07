"""
Evaluation functions for Fitness Coach Agent.
Each function returns a score in [0.0, 1.0].
"""


# ── Individual graders ────────────────────────────────────────────────────────

def evaluate_recovery_balance(env) -> float:
    """
    Task: recovery_balance
    Score based on whether fatigue is controlled without being lazy.
    """
    fatigue_values = list(env.muscle_fatigue.values())

    # Penalize overtraining any muscle
    if any(f > 80 for f in fatigue_values):
        return 0.0

    total_fatigue = sum(fatigue_values)

    # Penalize lazy agent (no training at all)
    if total_fatigue < 20:
        return 0.3

    # Reward balanced moderate fatigue
    if total_fatigue < 80:
        return 1.0

    return 0.6


def evaluate_energy_management(env) -> float:
    """
    Task: energy_management
    Score based on final energy level — agent should preserve energy.
    """
    energy = env.energy_level

    if energy < 20:
        return 0.0
    elif energy < 40:
        return 0.4
    elif energy < 60:
        return 0.7
    elif energy <= 100:
        return 1.0
    else:
        return 0.5  # unrealistically high (shouldn't happen after clamping)


def evaluate_training_distribution(env) -> float:
    """
    Task: training_distribution
    Score based on how evenly fatigue is spread across muscle groups.
    """
    fatigue = env.muscle_fatigue
    values = list(fatigue.values())

    max_f = max(values)
    min_f = min(values)
    spread = max_f - min_f

    if spread < 15:
        return 1.0
    elif spread < 30:
        return 0.8
    elif spread < 50:
        return 0.5
    elif spread < 70:
        return 0.2
    return 0.0


def evaluate_injury_management(env) -> float:
    """
    Task: injury_management
    Score based on final injury risk — agent should keep risk low.
    """
    risk = env.injury_risk

    if risk <= 0.1:
        return 1.0
    elif risk <= 0.3:
        return 0.8
    elif risk <= 0.5:
        return 0.5
    elif risk <= 0.7:
        return 0.2
    return 0.0


def evaluate_goal_alignment(env) -> float:
    """
    Task: goal_alignment
    Score based on whether training matches the stated goal.
    """
    fatigue = env.muscle_fatigue
    goal = env.goal

    if goal == "fat_loss":
        # Cardio should be highest fatigue
        if fatigue["cardio"] >= max(fatigue.values()):
            return 1.0
        elif fatigue["cardio"] >= 20:
            return 0.6
        return 0.2

    elif goal == "muscle_gain":
        # Strength fatigue should dominate cardio
        strength = fatigue["upper_body"] + fatigue["lower_body"]
        if strength > fatigue["cardio"] * 2:
            return 1.0
        elif strength > fatigue["cardio"]:
            return 0.7
        return 0.3

    elif goal == "endurance":
        # Cardio should be present but balanced
        if fatigue["cardio"] > 30:
            return 1.0
        elif fatigue["cardio"] > 10:
            return 0.6
        return 0.2

    return 0.5


# ── Task dispatcher ───────────────────────────────────────────────────────────

TASK_GRADERS = {
    "recovery_balance":      evaluate_recovery_balance,
    "energy_management":     evaluate_energy_management,
    "training_distribution": evaluate_training_distribution,
    "injury_management":     evaluate_injury_management,
    "goal_alignment":        evaluate_goal_alignment,
}


def evaluate_task_score(env, task_name: str) -> float:
    """
    Returns score in [0.0, 1.0] for the given task.
    Called at end of episode.
    """
    grader = TASK_GRADERS.get(task_name)
    if grader is None:
        raise ValueError(f"Unknown task '{task_name}'")
    score = grader(env)
    # Clamp just in case
    return round(max(0.0, min(1.0, score)), 2)


def evaluate_overall_score(env) -> float:
    """
    Average score across all tasks (uses current env state).
    """
    scores = [grader(env) for grader in TASK_GRADERS.values()]
    return round(sum(scores) / len(scores), 2)
