import os
import json
import time
import asyncio
import requests
from typing import List, Dict, Any, Optional
from openai import OpenAI

#Config
API_BASE_URL=os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME=os.environ.get("MODEL_NAME", "gpt-4o-mini")
API_KEY=os.environ.get("HF_TOKEN", os.environ.get("OPENAI_API_KEY", ""))
ENV_BASE_URL=os.environ.get("ENV_BASE_URL", "http://localhost:7860")

MAX_STEPS=30
MAX_TOTAL_REWARD=20.0
SUCCESS_THRESHOLD=0.5
TEMPERATURE=0.2
MAX_TOKENS=512
BENCHMARK="pandemic-response-v1"

TASKS=["task_1_easy","task_2_medium","task_3_hard"]

def log_start(task: str, env: str, model: str):
    print(json.dumps({
        "type":"START",
        "task":task,
        "env":env,
        "model":model,
    }), flush=True)

def log_step(step: int, action: Any, reward: float, done: bool, error: Optional[str]):
    print(json.dumps({
        "type":"STEP",
        "step":step,
        "action":action,
        "reward":reward,
        "done":done,
        "error":error,
    }), flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(json.dumps({
        "type":"END",
        "success":success,
        "steps":steps,
        "score":score,
        "rewards":rewards,
    }), flush=True)

#HTTP env
def env_reset(task_id: str, seed: int=42)->Dict:
    r=requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id, "seed": seed}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_step(action: Dict) -> Dict:
    r=requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30)
    r.raise_for_status()
    return r.json()

def env_state()->Dict:
    r=requests.get(f"{ENV_BASE_URL}/state", timeout=30)
    r.raise_for_status()
    return r.json()


#Agent prompt
SYSTEM_PROMPT="""You are an expert pandemic response coordinator.
Each day you must choose ONE action to contain disease spread across cities.

Available actions:
1. allocate_vaccines — send vaccine doses to a city
2. set_lockdown      — enable/disable lockdown in a city  
3. send_resources    — boost healthcare capacity in a city
4. no_op             — take no action

Respond ONLY with a valid JSON object. No explanation, no markdown, just JSON.

Examples:
{"action_type": "allocate_vaccines", "city_index": 0, "doses": 50000}
{"action_type": "set_lockdown", "city_index": 1, "enabled": true}
{"action_type": "send_resources", "city_index": 2, "amount": 50.0}
{"action_type": "no_op"}
"""

def build_prompt(obs: Dict, step: int, last_reward: float, history: List[str]) -> str:
    cities=obs.get("cities", [])
    city_lines=[]
    for i, c in enumerate(cities):
        city_lines.append(
            f"  [{i}] {c['name']}: infected={c['infected']*100:.1f}%  "
            f"vax={c['vaccinated']*100:.1f}%  deaths={c['deaths']:,}  "
            f"lockdown={'YES' if c['in_lockdown'] else 'no'}  resources={c['resources']:.0f}"
        )

    recent="\n".join(history[-5:]) if history else "None"

    return f"""Day {obs['day']}/{obs['max_days']} | Step {step}
Resources remaining: {obs['total_resources']:.0f}
Vaccines remaining:  {obs['vaccines_available']:,}
Last reward: {last_reward:+.4f}

Cities:
{chr(10).join(city_lines)}

Task: {obs.get('task_description', '')}
Last feedback: {obs.get('action_feedback', '')}

Recent actions:
{recent}

Choose your action for today:"""

def get_agent_action(client: OpenAI, obs: Dict, step: int, last_reward: float, history: List[str]) -> Dict:
    prompt=build_prompt(obs, step, last_reward, history)
    try:
        completion=client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        text=(completion.choices[0].message.content or "").strip()
        text=text.replace("```json", "").replace("```", "").strip()
        action=json.loads(text)
        return action
    except Exception as e:
        print(f"[DEBUG] Agent parse error: {e}", flush=True)
        #Fallback if model fails
        return _heuristic_action(obs)

#Simple fallback: vaccinate the most infected city.
def _heuristic_action(obs: Dict)->Dict:
    cities=obs.get("cities", [])
    if not cities:
        return {"action_type": "no_op"}
    most_infected=max(range(len(cities)), key=lambda i: cities[i]["infected"])
    vaccines=obs.get("vaccines_available", 0)
    if vaccines > 10000:
        return {"action_type": "allocate_vaccines", "city_index": most_infected, "doses": min(50000, vaccines)}
    resources=obs.get("total_resources", 0)
    if resources > 20:
        return {"action_type": "send_resources", "city_index": most_infected, "amount": min(30.0, resources)}
    if cities[most_infected]["infected"] > 0.10 and not cities[most_infected]["in_lockdown"]:
        return {"action_type": "set_lockdown", "city_index": most_infected, "enabled": True}
    return {"action_type": "no_op"}


#Main Task run
def run_task(client: OpenAI, task_id: str)->Dict:
    max_steps={"task_1_easy": 30, "task_2_medium": 40, "task_3_hard": 50}[task_id]
    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
    history:List[str]=[]
    rewards:List[float]=[]
    steps_taken=0
    score=0.0
    success=False
    try:
        reset_data=env_reset(task_id)
        obs=reset_data["observation"]
        last_reward=0.0
        done=obs.get("done", False)
        for step in range(1,max_steps+1):
            if done:
                break
            action=get_agent_action(client, obs, step, last_reward, history)
            try:
                step_data=env_step(action)
            except Exception as e:
                log_step(step=step, action=action, reward=0.0, done=False, error=str(e))
                break
            obs=step_data["observation"]
            reward=step_data.get("reward", 0.0)
            done=step_data.get("done", False)
            info=step_data.get("info", {})
            error=None
            rewards.append(reward)
            steps_taken=step
            last_reward=reward
            log_step(step=step, action=action, reward=reward, done=done, error=error)
            history.append(
                f"Step {step} | {action.get('action_type','?')} -> reward {reward:+.4f}"
            )
            if done:
                if "episode_result" in info:
                    score=info["episode_result"]["final_score"]
                    success=info["episode_result"]["success"]
                break
        if not done and rewards:
            # Grade currents state
            state_data=env_state()
            score_raw=sum(rewards)/MAX_TOTAL_REWARD
            score=round(min(max(score_raw, 0.0), 1.0), 4)
            success=score>=SUCCESS_THRESHOLD
    except Exception as e:
        print(f"[DEBUG] Task {task_id} error: {e}", flush=True)
    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
    return {"task_id": task_id, "score": score, "success": success, "steps": steps_taken}

#Wait for the environment server to be ready.
def wait_for_env(timeout: int=60):
    print(f"[DEBUG] Waiting for env at {ENV_BASE_URL}...", flush=True)
    start=time.time()
    while time.time()-start<timeout:
        try:
            r=requests.get(f"{ENV_BASE_URL}/health", timeout=5)
            if r.status_code==200:
                print("[DEBUG] Env ready.", flush=True)
                return
        except Exception:
            pass
        time.sleep(2)
    print("[DEBUG] Env assumed ready, continuing...")

def main():
    if not API_KEY:
        raise ValueError("Set HF_TOKEN or OPENAI_API_KEY environment variable.")
    client=OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    wait_for_env()
    results=[]
    for task_id in TASKS:
        print(f"\n{'='*60}", flush=True)
        print(f"[DEBUG] Running {task_id}", flush=True)
        result=run_task(client, task_id)
        results.append(result)
        time.sleep(1)
    print("\n"+"="*60, flush=True)
    print("[DEBUG] FINAL BASELINE SCORES:", flush=True)
    for r in results:
        status="PASS" if r["success"] else "FAIL"
        print(f"  {status}  {r['task_id']:25s}  score={r['score']:.4f}  steps={r['steps']}", flush=True)
    overall=sum(r["score"] for r in results)/len(results)
    print(f"\n  Overall average score: {overall:.4f}", flush=True)

if __name__=="__main__":
    main()