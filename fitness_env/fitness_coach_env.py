import random
from fitness_env.schemas import FitnessObservation, FitnessAction

VALID_TASKS = [
    "recovery_balance",
    "energy_management",
    "training_distribution",
    "injury_management",
    "goal_alignment",
]

# Fixed goals per task so graders are deterministic
TASK_GOALS = {
    "recovery_balance":      "muscle_gain",
    "energy_management":     "endurance",
    "training_distribution": "muscle_gain",
    "injury_management":     "fat_loss",
    "goal_alignment":        "fat_loss",
}


class FitnessCoachEnv:
    def __init__(self, task: str = "recovery_balance"):
        if task not in VALID_TASKS:
            raise ValueError(f"Unknown task '{task}'. Valid: {VALID_TASKS}")
        self.task = task
        self.reset()

    def reset(self):
        self.energy_level = 100
        self.muscle_fatigue = {
            "upper_body": 0,
            "lower_body": 0,
            "cardio":     0,
        }
        self.last_activity = None
        self.day = 0
        self.injury_risk = 0.0
        self.history = []

        # Deterministic goal per task (so graders reproduce)
        self.goal = TASK_GOALS[self.task]

        return self._get_observation()

    def _get_observation(self):
        return FitnessObservation(
            energy_level=self.energy_level,
            muscle_fatigue=dict(self.muscle_fatigue),
            last_activity=self.last_activity,
            goal=self.goal,
            injury_risk=round(self.injury_risk, 2),
            day=self.day,
        )

    def step(self, action: FitnessAction):
        reward = 0.0
        act = action.activity_type

        # ── Apply action effects ──────────────────────────────────────────
        if act == "upper_body_strength":
            self.energy_level -= 15
            self.muscle_fatigue["upper_body"] += 25

        elif act == "lower_body_strength":
            self.energy_level -= 15
            self.muscle_fatigue["lower_body"] += 25

        elif act == "cardio_training":
            self.energy_level -= 10
            self.muscle_fatigue["cardio"] += 20

        elif act == "recovery":
            self.energy_level = min(100, self.energy_level + 20)
            for k in self.muscle_fatigue:
                self.muscle_fatigue[k] = max(0, self.muscle_fatigue[k] - 15)

        # Clamp energy
        self.energy_level = max(0, min(100, self.energy_level))

        # ── Injury risk ───────────────────────────────────────────────────
        total_fatigue = sum(self.muscle_fatigue.values())
        if total_fatigue > 80:
            self.injury_risk = min(1.0, self.injury_risk + 0.2)
        elif act == "recovery":
            self.injury_risk = max(0.0, self.injury_risk - 0.1)

        # ── Goal-based rewards ────────────────────────────────────────────
        if self.goal == "fat_loss":
            if act == "cardio_training":
                reward += 0.5

        elif self.goal == "muscle_gain":
            if act in ["upper_body_strength", "lower_body_strength"]:
                reward += 0.5

        elif self.goal == "endurance":
            if act == "cardio_training":
                reward += 0.3
            if act == "recovery":
                reward += 0.2

        # ── Penalties ─────────────────────────────────────────────────────
        reward -= self.injury_risk

        if self.energy_level < 30:
            reward -= 0.3

        if act == "recovery" and self.last_activity == "recovery":
            reward -= 0.3

        # ── Consistency bonus (after day 5) ───────────────────────────────
        if self.day >= 5:
            unique_actions = set(self.history[-5:])
            if len(unique_actions) >= 3:
                reward += 0.3

        # ── State update ──────────────────────────────────────────────────
        self.history.append(act)
        self.last_activity = act
        self.day += 1

        done = self.day >= 7

        return self._get_observation(), round(reward, 2), done, {}

    def state(self):
        return self._get_observation()