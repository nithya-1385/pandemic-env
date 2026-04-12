#Pandemic Response Environment — OpenEnv compliant FastAPI server
import os
import json
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from env.simulation import PandemicSimulation
from env.models import (PandemicObservation, PandemicAction, PandemicReward, StateObservation, EpisodeResult)
from tasks.graders import TASKS, GRADERS

app=FastAPI(
    title="Pandemic Response Environment",
    description="OpenEnv-compliant RL environment for pandemic response decision-making",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_sim: Optional[PandemicSimulation]=None
_current_task_id: str="task_1_easy"
_prev_snapshot: Optional[dict]=None
_total_reward: float=0.0
_step_count: int=0


def _sim_to_observation(sim: PandemicSimulation, feedback: str="", done: bool=False) -> PandemicObservation:
    snap=sim.get_state_snapshot()
    states_obs=[StateObservation(**c) for c in snap["states"]]
    task=TASKS[_current_task_id]
    return PandemicObservation(
        day=snap["day"],
        max_days=sim.max_days,
        total_resources=snap["total_resources"],
        vaccines_available=snap["vaccines_available"],
        states=states_obs,
        done=done,
        action_feedback=feedback,
        task_description=task["description"],
    )

class ResetRequest(BaseModel):
    task_id: Optional[str]="task_1_easy"
    seed: Optional[int]=42

class StepRequest(BaseModel):
    action: Dict[str, Any]

class ResetResponse(BaseModel):
    observation: PandemicObservation
    task: Dict[str, Any]

class StepResponse(BaseModel):
    observation: PandemicObservation
    reward: float
    done: bool
    info: Dict[str, Any]

class StateResponse(BaseModel):
    state: Dict[str, Any]
    task_id: str
    step_count: int
    total_reward: float

@app.get("/")
def root():
    return {
        "message": "Pandemic Response API is running",
        "endpoints": [
            "/health",
            "/tasks",
            "/reset",
            "/step",
            "/state",
            "/grade"
        ]
    }

@app.get("/health")
def health():
    return {"status": "ok", "env": "pandemic-response-v1"}

@app.get("/tasks")
def list_tasks():
    return {"tasks": list(TASKS.values())}

@app.post("/reset", response_model=ResetResponse)
def reset(req: ResetRequest=ResetRequest()):
    global _sim, _current_task_id, _prev_snapshot, _total_reward, _step_count
    task_id=req.task_id or "task_1_easy"
    if task_id not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}. Valid: {list(TASKS.keys())}")
    task=TASKS[task_id]
    difficulty=task["difficulty"]
    _current_task_id=task_id
    _sim=PandemicSimulation(seed=req.seed or 42, difficulty=difficulty)
    _prev_snapshot=_sim.get_state_snapshot()
    _total_reward=0.0
    _step_count=0
    obs=_sim_to_observation(_sim, feedback="Environment reset. Ready.", done=False)
    return ResetResponse(observation=obs, task=task)

@app.post("/step", response_model=StepResponse)
def step(req: StepRequest):
    global _sim, _prev_snapshot, _total_reward, _step_count
    if _sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    done=_sim.is_done()
    if done:
        obs=_sim_to_observation(_sim, feedback="Episode already done. Call /reset.", done=True)
        grade=GRADERS[_current_task_id](_sim)
        return StepResponse(
            observation=obs,
            reward=0.0,
            done=True,
            info={"grade": grade, "total_reward": _total_reward}
        )
    # Apply action
    action_dict=req.action
    feedback_dict=_sim.apply_action(action_dict)
    # Advance simulation by one day
    prev_snap=_sim.get_state_snapshot()
    _sim.step_simulation()
    _step_count+=1
    # Compute reward
    reward=_sim.compute_reward(prev_snap, feedback_dict)
    _total_reward+=reward
    done=_sim.is_done()
    obs=_sim_to_observation(_sim, feedback=feedback_dict.get("message", ""), done=done)
    info: Dict[str, Any]={
        "action_feedback": feedback_dict,
        "step": _step_count,
        "total_reward": round(_total_reward, 4),
    }
    if done:
        grade=GRADERS[_current_task_id](_sim)
        info["grade"]=grade
        info["episode_result"]={
            "task_id": _current_task_id,
            "final_score": grade["score"],
            "total_reward": round(_total_reward, 4),
            "days_survived": _sim.day,
            "total_deaths": sum(c.deaths for c in _sim.states),
            "avg_infected_final": round(sum(c.infected for c in _sim.states)/len(_sim.states), 4),
            "avg_vaccinated_final": round(sum(c.vaccinated for c in _sim.states)/len(_sim.states), 4),
            "success": grade["success"],
        }
    return StepResponse(observation=obs, reward=reward, done=done, info=info)

@app.get("/state", response_model=StateResponse)
def state():
    if _sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return StateResponse(
        state=_sim.get_state_snapshot(),
        task_id=_current_task_id,
        step_count=_step_count,
        total_reward=round(_total_reward, 4),
    )

@app.get("/grade")
def grade():
    if _sim is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    result=GRADERS[_current_task_id](_sim)
    return {"task_id": _current_task_id, "grade": result}

if __name__=="__main__":
    import uvicorn
    port=int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=False)