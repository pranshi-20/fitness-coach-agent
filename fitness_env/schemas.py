from pydantic import BaseModel
from typing import Optional, Dict


class FitnessObservation(BaseModel):
    energy_level: int
    muscle_fatigue: Dict[str, int]
    last_activity: Optional[str]
    goal: str
    injury_risk: float
    day: int


class FitnessAction(BaseModel):
    activity_type: str