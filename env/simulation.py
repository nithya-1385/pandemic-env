"""
Pandemic Response Simulation Engine
Models disease spread across a network of cities with resources.
"""
import random
import math
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class City:
    name: str
    population: int
    infected: float = 0.0        # fraction 0–1
    vaccinated: float = 0.0      # fraction 0–1
    in_lockdown: bool = False
    healthcare_capacity: float = 1.0   # multiplier, 1.0 = normal
    resources: float = 100.0           # resource units available
    deaths: int = 0
    recovered: float = 0.0

    @property
    def susceptible(self) -> float:
        return max(0.0, 1.0 - self.infected - self.vaccinated - self.recovered)

    @property
    def infection_count(self) -> int:
        return int(self.infected * self.population)

    @property
    def death_rate(self) -> float:
        # Higher death rate if infected exceeds healthcare capacity
        overload = max(0.0, self.infected - self.healthcare_capacity * 0.2)
        base = 0.01
        return min(base + overload * 0.05, 0.08)


CITY_TEMPLATES = [
    {"name": "Metropolis",  "population": 5_000_000, "infected": 0.02, "vaccinated": 0.10},
    {"name": "Harbortown",  "population": 1_200_000, "infected": 0.01, "vaccinated": 0.15},
    {"name": "Riverdale",   "population": 800_000,   "infected": 0.005,"vaccinated": 0.20},
    {"name": "Northport",   "population": 600_000,   "infected": 0.008,"vaccinated": 0.12},
    {"name": "Eastbridge",  "population": 400_000,   "infected": 0.003,"vaccinated": 0.18},
]

# Adjacency (travel connections between cities)
CONNECTIONS = [
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4)
]


class PandemicSimulation:
    def __init__(self, seed: int = 42, difficulty: str = "easy"):
        self.seed = seed
        self.difficulty = difficulty
        self.rng = random.Random(seed)
        self.cities: List[City] = []
        self.day: int = 0
        self.total_resources: float = 0.0
        self.vaccines_available: int = 0
        self.history: List[Dict] = []
        self._setup(difficulty)

    def _setup(self, difficulty: str):
        self.cities = [City(**t) for t in CITY_TEMPLATES]

        if difficulty == "easy":
            # One city, mild outbreak, plenty of resources
            self.cities = self.cities[:2]
            self.total_resources = 500.0
            self.vaccines_available = 300_000
            self.max_days = 30

        elif difficulty == "medium":
            # Three cities, moderate outbreak
            self.cities = self.cities[:3]
            # bump infections
            self.cities[0].infected = 0.05
            self.cities[1].infected = 0.03
            self.total_resources = 350.0
            self.vaccines_available = 200_000
            self.max_days = 40

        else:  # hard
            # All five cities, severe outbreak, scarce resources
            for c in self.cities:
                c.infected *= 3.0
                c.infected = min(c.infected, 0.25)
            self.total_resources = 200.0
            self.vaccines_available = 100_000
            self.max_days = 50

        self.day = 0
        self.history = []

    def _spread(self):
        """SIR-like spread within each city, plus travel spread between connected cities."""
        R0_base = 2.5
        gamma = 0.07   # recovery rate per day

        new_infections = []
        for city in self.cities:
            effective_R0 = R0_base
            if city.in_lockdown:
                effective_R0 *= 0.4
            effective_R0 *= (1.0 - city.vaccinated * 0.85)

            beta = effective_R0 * gamma
            delta = beta * city.infected * city.susceptible
            delta += self.rng.gauss(0, 0.001)   # small noise
            delta = max(0.0, delta)
            new_infections.append(delta)

        # Travel spread
        for (i, j) in CONNECTIONS:
            if i >= len(self.cities) or j >= len(self.cities):
                continue
            travel_rate = 0.002
            if self.cities[i].in_lockdown or self.cities[j].in_lockdown:
                travel_rate *= 0.1
            spread_ij = travel_rate * self.cities[i].infected
            spread_ji = travel_rate * self.cities[j].infected
            new_infections[j] = new_infections[j] + spread_ij * 0.3
            new_infections[i] = new_infections[i] + spread_ji * 0.3

        for idx, city in enumerate(self.cities):
            delta = new_infections[idx]
            deaths_today = city.infected * city.death_rate
            city.deaths += int(deaths_today * city.population)
            recovered_today = city.infected * gamma * 0.9
            city.infected = min(1.0, max(0.0, city.infected + delta - deaths_today - recovered_today))
            city.recovered = min(1.0, city.recovered + recovered_today)

    def step_simulation(self):
        self._spread()
        self.day += 1
        snapshot = self.get_state_snapshot()
        self.history.append(snapshot)
        return snapshot

    def apply_action(self, action: dict) -> dict:
        """
        Actions:
          allocate_vaccines: {city_index: int, doses: int}
          set_lockdown:      {city_index: int, enabled: bool}
          send_resources:    {city_index: int, amount: float}
          no_op:             {}
        """
        feedback = {"valid": True, "message": ""}
        action_type = action.get("action_type", "no_op")

        if action_type == "allocate_vaccines":
            city_idx = action.get("city_index", 0)
            doses = int(action.get("doses", 0))
            if city_idx >= len(self.cities):
                feedback = {"valid": False, "message": "Invalid city index"}
            elif doses > self.vaccines_available:
                doses = self.vaccines_available
                feedback["message"] = f"Only {self.vaccines_available} doses available, used all."
            if feedback["valid"]:
                city = self.cities[city_idx]
                vax_fraction = min(doses / city.population, city.susceptible)
                city.vaccinated = min(1.0, city.vaccinated + vax_fraction)
                self.vaccines_available -= doses
                feedback["message"] = f"Vaccinated {doses} people in {city.name}"

        elif action_type == "set_lockdown":
            city_idx = action.get("city_index", 0)
            enabled = action.get("enabled", True)
            if city_idx >= len(self.cities):
                feedback = {"valid": False, "message": "Invalid city index"}
            else:
                self.cities[city_idx].in_lockdown = enabled
                status = "ON" if enabled else "OFF"
                feedback["message"] = f"Lockdown {status} in {self.cities[city_idx].name}"
                if enabled:
                    cost = 10.0
                    self.total_resources = max(0, self.total_resources - cost)

        elif action_type == "send_resources":
            city_idx = action.get("city_index", 0)
            amount = float(action.get("amount", 0))
            if city_idx >= len(self.cities):
                feedback = {"valid": False, "message": "Invalid city index"}
            elif amount > self.total_resources:
                amount = self.total_resources
                feedback["message"] = "Insufficient resources, used all remaining."
            if feedback["valid"]:
                self.cities[city_idx].resources += amount
                self.cities[city_idx].healthcare_capacity = min(
                    2.0, self.cities[city_idx].healthcare_capacity + amount * 0.005
                )
                self.total_resources -= amount
                feedback["message"] = f"Sent {amount:.0f} resources to {self.cities[city_idx].name}"

        elif action_type == "no_op":
            feedback["message"] = "No action taken"
        else:
            feedback = {"valid": False, "message": f"Unknown action: {action_type}"}

        return feedback

    def get_state_snapshot(self) -> dict:
        return {
            "day": self.day,
            "total_resources": self.total_resources,
            "vaccines_available": self.vaccines_available,
            "cities": [
                {
                    "name": c.name,
                    "population": c.population,
                    "infected": round(c.infected, 4),
                    "vaccinated": round(c.vaccinated, 4),
                    "recovered": round(c.recovered, 4),
                    "in_lockdown": c.in_lockdown,
                    "healthcare_capacity": round(c.healthcare_capacity, 3),
                    "resources": round(c.resources, 1),
                    "deaths": c.deaths,
                    "infection_count": c.infection_count,
                }
                for c in self.cities
            ]
        }

    def compute_reward(self, prev_snapshot: dict, action_feedback: dict) -> float:
        """Dense reward: reward infection reduction, penalize deaths and wasted resources."""
        reward = 0.0

        n = len(self.cities)
        prev_infected = sum(c["infected"] for c in prev_snapshot["cities"])
        curr_infected = sum(c.infected for c in self.cities)
        infection_delta = prev_infected - curr_infected
        reward += max(0.0, (0.20 - curr_infected / n)) * 5.0
        reward += infection_delta * 3.0

        total_deaths_today = sum(
            c.deaths - prev_snapshot["cities"][i]["deaths"]
            for i, c in enumerate(self.cities)
        )
        reward -= min(total_deaths_today * 0.00005, 0.3)

        # Reward vaccine coverage
        curr_vax = sum(c.vaccinated for c in self.cities) / n
        reward += curr_vax * 0.3

        # Small penalty for running out of resources
        if self.total_resources <= 0:
            reward -= 0.1

        # Invalid action penalty
        if not action_feedback.get("valid", True):
            reward -= 0.05

        return round(reward, 4)

    def is_done(self) -> bool:
        if self.day >= self.max_days:
            return True
        # Win condition: all cities below 1% infected
        if all(c.infected < 0.01 for c in self.cities):
            return True
        # Lose condition: any city above 80% infected
        if any(c.infected > 0.80 for c in self.cities):
            return True
        return False

    def final_score(self) -> float:
        """Score 0–1 for use by graders."""
        total_pop = sum(c.population for c in self.cities)
        total_deaths = sum(c.deaths for c in self.cities)
        avg_infected = sum(c.infected for c in self.cities) / len(self.cities)
        avg_vaccinated = sum(c.vaccinated for c in self.cities) / len(self.cities)

        death_penalty = min(1.0, total_deaths / (total_pop * 0.02))
        infection_score = max(0.0, 1.0 - avg_infected / 0.5)
        vax_score = avg_vaccinated

        score = (
            0.5 * infection_score +
            0.3 * (1.0 - death_penalty) +
            0.2 * vax_score
        )
        return round(min(max(score, 0.0), 1.0), 4)
