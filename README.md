# 🦠 Pandemic Response Environment

An **OpenEnv-compliant** reinforcement learning environment where an AI agent acts as a pandemic response coordinator — allocating vaccines, enforcing lockdowns, and routing healthcare resources across a network of cities to contain disease spread and minimize deaths.

> **Real-world motivation:** Public health agencies face exactly this resource allocation problem. This environment models the core trade-offs: lockdowns slow spread but cost resources; vaccines provide lasting immunity but are scarce; healthcare investment reduces death rates but requires time to deploy.

---

## 🗂 Project Structure

```
pandemic-env/
├── env/
│   ├── simulation.py     # SIR-based disease spread engine
│   └── models.py         # Pydantic typed models (Observation, Action, Reward)
├── tasks/
│   └── graders.py        # 3 tasks with deterministic graders
├── server.py             # FastAPI OpenEnv-compliant server
├── inference.py          # Baseline LLM agent script
├── openenv.yaml          # OpenEnv metadata
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## 🌍 Environment Description

The simulation runs a modified **SIR (Susceptible-Infected-Recovered)** model across a network of cities connected by travel routes. Each day:

1. Disease spreads within cities (R0 = 2.5, modified by lockdowns and vaccination)
2. Disease spreads between connected cities via travel
3. Deaths accumulate based on infection level vs. healthcare capacity
4. The agent takes one action (vaccine allocation, lockdown, resource dispatch, or no-op)

### Cities
| City | Population |
|------|-----------|
| Metropolis | 5,000,000 |
| Harbortown | 1,200,000 |
| Riverdale | 800,000 |
| Northport | 600,000 |
| Eastbridge | 400,000 |

---

## 👁 Observation Space

```json
{
  "day": 5,
  "max_days": 30,
  "total_resources": 420.0,
  "vaccines_available": 250000,
  "cities": [
    {
      "name": "Metropolis",
      "population": 5000000,
      "infected": 0.032,
      "vaccinated": 0.15,
      "recovered": 0.04,
      "in_lockdown": false,
      "healthcare_capacity": 1.0,
      "resources": 100.0,
      "deaths": 1200,
      "infection_count": 160000
    }
  ],
  "done": false,
  "action_feedback": "Vaccinated 50000 people in Metropolis",
  "task_description": "Contain a mild outbreak in one city..."
}
```

---

## 🎮 Action Space

| Action | Parameters | Description |
|--------|-----------|-------------|
| `allocate_vaccines` | `city_index`, `doses` | Send vaccine doses to a city |
| `set_lockdown` | `city_index`, `enabled` | Enable/disable lockdown (costs 10 resources) |
| `send_resources` | `city_index`, `amount` | Boost city's healthcare capacity |
| `no_op` | — | Take no action |

**Example action:**
```json
{"action_type": "allocate_vaccines", "city_index": 0, "doses": 50000}
```

---

## 🏆 Tasks

### Task 1 — Easy: Single City Outbreak Control
- **Cities:** 1 (Metropolis only)
- **Duration:** 30 days
- **Resources:** 500 units | **Vaccines:** 300,000 doses
- **Goal:** Keep infection below 5%, deaths below 5,000
- **Success threshold:** Score ≥ 0.60
- **Grader:** Weighted combination of infection rate, death count, vaccination coverage, and speed bonus

### Task 2 — Medium: Multi-City Epidemic Management
- **Cities:** 3 (Metropolis, Harbortown, Riverdale)
- **Duration:** 40 days
- **Resources:** 350 units | **Vaccines:** 200,000 doses
- **Goal:** Average infection < 3%, vaccinate ≥ 40% of population
- **Success threshold:** Score ≥ 0.65
- **Grader:** Infection containment, vaccination rate, per-city containment, total deaths

### Task 3 — Hard: National Pandemic Crisis
- **Cities:** All 5 (severe outbreak, 3× initial infections)
- **Duration:** 50 days
- **Resources:** 200 units (critically scarce) | **Vaccines:** 100,000 doses
- **Goal:** Prevent systemic collapse — deaths < 50,000, ≥1 city below 10% infected
- **Success threshold:** Score ≥ 0.50
- **Grader:** Death toll, cities below threshold, spread control, collapse penalty

---

## 📈 Reward Function

The reward is **dense** — provided every step, not just at episode end:

```
reward = +10 × (infection_reduction)
       + 0.5 × (avg_vaccination_rate)
       - 0.0001 × (deaths_this_step)
       - 0.1   × (resource_depletion_penalty)
       - 0.05  × (invalid_action_penalty)
```

This gives the agent a meaningful learning signal throughout the episode, rewarding containment behavior and penalizing destructive or invalid actions.

---

## 🚀 Setup & Usage

### Option A: Run locally with Python

```bash
git clone <your-repo>
cd pandemic-env
pip install -r requirements.txt

# Start environment server
python server.py

# In another terminal — run baseline inference
export HF_TOKEN=your_api_key
export MODEL_NAME=gpt-4o-mini
export API_BASE_URL=https://api.openai.com/v1
python inference.py
```

### Option B: Run with Docker

```bash
docker build -t pandemic-env .
docker run -p 7860:7860 pandemic-env

# Then run inference pointing at localhost
export ENV_BASE_URL=http://localhost:7860
python inference.py
```

### Option C: Google Colab

```python
# In a Colab cell:
!git clone <your-repo> && cd pandemic-env && pip install -r requirements.txt -q
import subprocess, time
proc = subprocess.Popen(["python", "server.py"], cwd="pandemic-env")
time.sleep(4)
# Then run inference
```

---

## 🌐 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/tasks` | GET | List all tasks |
| `/reset` | POST | Reset environment (`task_id`, `seed`) |
| `/step` | POST | Take an action (`action` dict) |
| `/state` | GET | Get current state |
| `/grade` | GET | Get current grade without ending episode |

---

## 📊 Baseline Scores

Baseline agent: `gpt-4o-mini` with heuristic fallback

| Task | Difficulty | Score | Success |
|------|-----------|-------|---------|
| task_1_easy | Easy | ~0.65 | ✓ |
| task_2_medium | Medium | ~0.55 | ~ |
| task_3_hard | Hard | ~0.40 | ✗ |

*Scores vary with model; hard task is intentionally challenging for frontier models.*

---

## 🔧 Environment Variables

| Variable | Description |
|----------|-------------|
| `API_BASE_URL` | LLM API endpoint (e.g. `https://api.openai.com/v1`) |
| `MODEL_NAME` | Model identifier (e.g. `gpt-4o-mini`) |
| `HF_TOKEN` | Your Hugging Face / API key |
| `ENV_BASE_URL` | Environment server URL (default: `http://localhost:7860`) |

---

## 🤗 Hugging Face Deployment

This environment is deployed as a Hugging Face Space tagged with `openenv`.
The Space runs the FastAPI server on port 7860 and responds to all OpenEnv endpoints.

Set the following Space secrets: `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`
