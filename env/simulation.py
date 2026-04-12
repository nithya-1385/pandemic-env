#Pandemic Response Simulation Engine
#Models disease spread across a network of states with resources.
import random
import math
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class State:
    name: str
    population: int
    infected: float=0.0
    vaccinated: float=0.0
    in_lockdown: bool=False
    healthcare_capastate: float=1.0
    resources: float=100.0
    deaths: int=0
    recovered: float=0.0
    @property
    def susceptible(self)->float:
        return max(0.0, 1.0-self.infected-self.vaccinated-self.recovered)
    @property
    def infection_count(self)->int:
        return int(self.infected*self.population)
    @property
    def death_rate(self)->float:
        # Higher death rate if infected exceeds healthcare capastate
        overload=max(0.0, self.infected-self.healthcare_capastate*0.2)
        base=0.01
        return min(base+overload*0.05, 0.08)

STATE_TEMPLATES=[
    {"name": "Karnataka",  "population": 5_000_000, "infected": 0.02, "vaccinated": 0.10},
    {"name": "Maharashtra",  "population": 1_200_000, "infected": 0.01, "vaccinated": 0.15},
    {"name": "Tamil_Nadu",   "population": 800_000,   "infected": 0.005,"vaccinated": 0.20},
    {"name": "Gujarat",   "population": 600_000,   "infected": 0.008,"vaccinated": 0.12},
    {"name": "Odisha",  "population": 400_000,   "infected": 0.003,"vaccinated": 0.18},]

# Adjacency (travel connections between states)
CONNECTIONS=[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)]

class PandemicSimulation:
    def __init__(self, seed: int=42, difficulty: str="easy"):
        self.seed=seed
        self.difficulty=difficulty
        self.rng=random.Random(seed)
        self.states: List[State]=[]
        self.day: int=0
        self.total_resources: float=0.0
        self.vaccines_available: int=0
        self.history: List[Dict]=[]
        self._setup(difficulty)
        
    def _setup(self, difficulty: str):
        self.states=[State(**t) for t in STATE_TEMPLATES]
        if difficulty=="easy":
            # One state, mild outbreak, plenty of resources
            self.states=self.states[:2]
            self.total_resources=500.0
            self.vaccines_available=300_000
            self.max_days=30
        elif difficulty=="medium":
            # Three states, moderate outbreak
            self.states=self.states[:3]
            # bump infections
            self.states[0].infected=0.05
            self.states[1].infected=0.03
            self.total_resources=350.0
            self.vaccines_available=200_000
            self.max_days=40
        else:
            # All five states, severe outbreak, scarce resources
            for c in self.states:
                c.infected*=3.0
                c.infected=min(c.infected, 0.25)
            self.total_resources=200.0
            self.vaccines_available=100_000
            self.max_days=50
        self.day=0
        self.history=[]

#SIR-like spread within each state, plus travel spread between connected states.
    def _spread(self):
        R0_base=2.5
        gamma=0.07   # recovery rate per day
        new_infections=[]
        for state in self.states:
            effective_R0=R0_base
            if state.in_lockdown:
                effective_R0*=0.4
            effective_R0*=(1.0-state.vaccinated*0.85)
            beta=effective_R0*gamma
            delta=beta*state.infected*state.susceptible
            delta+=self.rng.gauss(0, 0.001)   # small noise
            delta=max(0.0, delta)
            new_infections.append(delta)
        # Travel spread
        for (i, j) in CONNECTIONS:
            if i>=len(self.states) or j>=len(self.states):
                continue
            travel_rate=0.002
            if self.states[i].in_lockdown or self.states[j].in_lockdown:
                travel_rate*=0.1
            spread_ij=travel_rate*self.states[i].infected
            spread_ji=travel_rate*self.states[j].infected
            new_infections[j]=new_infections[j]+spread_ij*0.3
            new_infections[i]=new_infections[i]+spread_ji*0.3
        for idx, state in enumerate(self.states):
            delta=new_infections[idx]
            deaths_today=state.infected*state.death_rate
            state.deaths+=int(deaths_today*state.population)
            recovered_today=state.infected*gamma*0.9
            state.infected=min(1.0, max(0.0, state.infected+delta-deaths_today-recovered_today))
            state.recovered=min(1.0, state.recovered+recovered_today)

    def step_simulation(self):
        self._spread()
        self.day+=1
        snapshot=self.get_state_snapshot()
        self.history.append(snapshot)
        return snapshot

    def apply_action(self, action: dict)->dict:
        """
        Actions:
          allocate_vaccines: {state_index: int, doses: int}
          set_lockdown:      {state_index: int, enabled: bool}
          send_resources:    {state_index: int, amount: float}
          no_op:             {}
        """
        feedback={"valid": True, "message": ""}
        action_type=action.get("action_type", "no_op")
        if action_type=="allocate_vaccines":
            state_idx=action.get("state_index", 0)
            doses=int(action.get("doses", 0))
            if state_idx>=len(self.states):
                feedback={"valid": False, "message": "Invalid state index"}
            elif doses > self.vaccines_available:
                doses=self.vaccines_available
                feedback["message"]=f"Only {self.vaccines_available} doses available, used all."
            if feedback["valid"]:
                state=self.states[state_idx]
                vax_fraction=min(doses/state.population, state.susceptible)
                state.vaccinated=min(1.0, state.vaccinated+vax_fraction)
                self.vaccines_available-=doses
                feedback["message"]=f"Vaccinated {doses} people in {state.name}"
        elif action_type=="set_lockdown":
            state_idx=action.get("state_index", 0)
            enabled=action.get("enabled", True)
            if state_idx>=len(self.states):
                feedback={"valid": False, "message": "Invalid state index"}
            else:
                self.states[state_idx].in_lockdown=enabled
                status="ON" if enabled else "OFF"
                feedback["message"]=f"Lockdown {status} in {self.states[state_idx].name}"
                if enabled:
                    cost=10.0
                    self.total_resources=max(0, self.total_resources-cost)
        elif action_type=="send_resources":
            state_idx=action.get("state_index", 0)
            amount=float(action.get("amount", 0))
            if state_idx>=len(self.states):
                feedback={"valid": False, "message": "Invalid state index"}
            elif amount > self.total_resources:
                amount=self.total_resources
                feedback["message"]="Insufficient resources, used all remaining."
            if feedback["valid"]:
                self.states[state_idx].resources+=amount
                self.states[state_idx].healthcare_capastate=min(2.0, self.states[state_idx].healthcare_capastate+amount*0.005)
                self.total_resources-=amount
                feedback["message"]=f"Sent {amount:.0f} resources to {self.states[state_idx].name}"
        elif action_type=="no_op":
            feedback["message"]="No action taken"
        else:
            feedback={"valid": False, "message": f"Unknown action: {action_type}"}
        return feedback

    def get_state_snapshot(self)->dict:
        return {
            "day": self.day,
            "total_resources": self.total_resources,
            "vaccines_available": self.vaccines_available,
            "states": [
                {
                    "name": c.name,
                    "population": c.population,
                    "infected": round(c.infected, 4),
                    "vaccinated": round(c.vaccinated, 4),
                    "recovered": round(c.recovered, 4),
                    "in_lockdown": c.in_lockdown,
                    "healthcare_capastate": round(c.healthcare_capastate, 3),
                    "resources": round(c.resources, 1),
                    "deaths": c.deaths,
                    "infection_count": c.infection_count,
                }
                for c in self.states
            ]
        }

#Dense reward: reward infection reduction, penalize deaths and wasted resources.
    def compute_reward(self, prev_snapshot: dict, action_feedback: dict)->float:
        reward=0.0
        n=len(self.states)
        prev_infected=sum(c["infected"] for c in prev_snapshot["states"])
        curr_infected=sum(c.infected for c in self.states)
        infection_delta=prev_infected-curr_infected
        reward+=max(0.0, (0.20-curr_infected/n))*5.0
        reward+=infection_delta*3.0
        total_deaths_today=sum(c.deaths-prev_snapshot["states"][i]["deaths"]
            for i, c in enumerate(self.states)
        )
        reward-=min(total_deaths_today*0.00005, 0.3)
        # Reward vaccine coverage
        curr_vax=sum(c.vaccinated for c in self.states)/n
        reward+=curr_vax*0.3
        # Small penalty for running out of resources
        if self.total_resources <= 0:
            reward-=0.1
        # Invalid action penalty
        if not action_feedback.get("valid", True):
            reward-=0.05
        return round(reward, 4)

    def is_done(self)->bool:
        if self.day>=self.max_days:
            return True
        # Win condition: all states below 1% infected
        if all(c.infected < 0.01 for c in self.states):
            return True
        # Lose condition: any state above 80% infected
        if any(c.infected > 0.80 for c in self.states):
            return True
        return False

    def final_score(self)->float:
        total_pop=sum(c.population for c in self.states)
        total_deaths=sum(c.deaths for c in self.states)
        avg_infected=sum(c.infected for c in self.states)/len(self.states)
        avg_vaccinated=sum(c.vaccinated for c in self.states)/len(self.states)
        death_penalty=min(1.0, total_deaths/(total_pop*0.02))
        infection_score=max(0.0, 1.0-avg_infected/0.5)
        vax_score=avg_vaccinated
        score=(
            0.5*infection_score +
            0.3*(1.0-death_penalty) +
            0.2*vax_score
        )
        return round(min(max(score, 0.0), 1.0), 4)
