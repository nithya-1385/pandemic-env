# OpenEnv typed models: Observation, Action, Reward
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class StateObservation(BaseModel):
    name: str
    population: int
    infected: float=Field(ge=0.0, le=1.0, description="Fraction of population infected")
    vaccinated: float=Field(ge=0.0, le=1.0, description="Fraction vaccinated")
    recovered: float=Field(ge=0.0, le=1.0, description="Fraction recovered")
    in_lockdown: bool
    healthcare_capastate: float=Field(ge=0.0, description="Healthcare capastate multiplier")
    resources: float=Field(ge=0.0, description="Resource units allocated to state")
    deaths: int=Field(ge=0, description="Cumulative deaths")
    infection_count: int=Field(ge=0, description="Absolute number currently infected")

class PandemicObservation(BaseModel):
    day: int=Field(ge=0, description="Current simulation day")
    max_days: int=Field(description="Maximum days in episode")
    total_resources: float=Field(ge=0.0, description="Global resource pool remaining")
    vaccines_available: int=Field(ge=0, description="Vaccine doses remaining")
    states: List[StateObservation]
    done: bool=False
    action_feedback: str=""
    task_description: str=""

class PandemicAction(BaseModel):
    action_type: str=Field(description="One of: allocate_vaccines, set_lockdown, send_resources, no_op")
    state_index: Optional[int]=Field(default=None,description="0-based index of target state")
    doses: Optional[int]=Field(default=None,description="Number of vaccine doses to allocate (for allocate_vaccines)")
    enabled: Optional[bool]=Field(default=None,description="True=lockdown on, False=lockdown off (for set_lockdown)")
    amount: Optional[float]=Field(default=None,description="Resource units to send (for send_resources)")
    reasoning: Optional[str]=Field(default=None,description="Optional chain-of-thought from agent")

class PandemicReward(BaseModel):
    value: float=Field(description="Reward signal for this step")
    infection_delta: float=Field(description="Change in total infection fraction")
    deaths_today: int=Field(description="Deaths this step")
    partial_score: float=Field(ge=0.0, le=1.0, description="Partial progress score 0-1")

class EpisodeResult(BaseModel):
    task_id: str
    final_score: float=Field(ge=0.0, le=1.0)
    total_reward: float
    days_survived: int
    total_deaths: int
    avg_infected_final: float
    avg_vaccinated_final: float
    success: bool
