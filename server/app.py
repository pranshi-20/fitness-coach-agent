from fitness_env.fitness_coach_env import FitnessCoachEnv
from fitness_env.schemas import FitnessAction

env = FitnessCoachEnv()

def reset():
    return env.reset().dict()

def step(action: dict):
    action_obj = FitnessAction(**action)
    obs, reward, done, info = env.step(action_obj)
    return obs.dict(), reward.value, done, info

def state():
    return env.state()


# ✅ REQUIRED: main entry point
def main():
    print("Fitness Coach OpenEnv server running...")
    # Minimal server loop (for validation purposes)
    obs = reset()
    print("Initial Observation:", obs)

    # simulate one step
    sample_action = {"activity_type": "recovery"}
    obs, reward, done, _ = step(sample_action)

    print("Step Output:", obs, reward, done)


# ✅ REQUIRED: callable entry
if __name__ == "__main__":
    main()
