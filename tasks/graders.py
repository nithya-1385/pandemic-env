"""
Three tasks with deterministic graders: easy, medium, hard.
Each returns a score in [0.0, 1.0].
"""
from typing import Dict, Any
from env.simulation import PandemicSimulation


TASKS = {
    "task_1_easy": {
        "id": "task_1_easy",
        "name": "Single City Outbreak Control",
        "difficulty": "easy",
        "description": (
            "A mild outbreak is spreading in Metropolis (population 5M). "
            "You have 30 days, 500 resource units, and 300,000 vaccine doses. "
            "Goal: Keep average infection below 5% and total deaths below 5,000."
        ),
        "success_threshold": 0.6,
        "max_days": 30,
    },
    "task_2_medium": {
        "id": "task_2_medium",
        "name": "Multi-City Epidemic Management",
        "difficulty": "medium",
        "description": (
            "A growing epidemic is spreading across 3 connected cities. "
            "You have 40 days, 350 resource units, and 200,000 vaccine doses. "
            "Goal: Contain spread to under 3% average infection and vaccinate at least 40% of the population."
        ),
        "success_threshold": 0.65,
        "max_days": 40,
    },
    "task_3_hard": {
        "id": "task_3_hard",
        "name": "National Pandemic Crisis",
        "difficulty": "hard",
        "description": (
            "A severe pandemic is ravaging all 5 cities simultaneously. "
            "Resources are critically scarce. You have 50 days, only 200 resource units, "
            "and 100,000 vaccine doses. Goal: Prevent systemic collapse — keep deaths below "
            "50,000 and at least one city below 10% infection."
        ),
        "success_threshold": 0.5,
        "max_days": 50,
    },
}


def grade_task_1(sim: PandemicSimulation) -> Dict[str, Any]:
    """Easy grader: single city infection + death control."""
    city = sim.cities[0]
    avg_infected = city.infected
    total_deaths = city.deaths

    # Score components
    infection_score = max(0.0, 1.0 - avg_infected / 0.10)      # 0 if >=10% infected
    death_score = max(0.0, 1.0 - total_deaths / 10_000)         # 0 if >=10k deaths
    vax_score = min(1.0, city.vaccinated / 0.30)                 # full score at 30% vax
    speed_bonus = max(0.0, (sim.max_days - sim.day) / sim.max_days) * 0.1

    score = (
        0.45 * infection_score +
        0.35 * death_score +
        0.15 * vax_score +
        0.05 * speed_bonus
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    return {
        "score": score,
        "success": score >= TASKS["task_1_easy"]["success_threshold"],
        "breakdown": {
            "infection_score": round(infection_score, 4),
            "death_score": round(death_score, 4),
            "vax_score": round(vax_score, 4),
            "speed_bonus": round(speed_bonus, 4),
            "final_infected_pct": round(avg_infected * 100, 2),
            "total_deaths": total_deaths,
        }
    }


def grade_task_2(sim: PandemicSimulation) -> Dict[str, Any]:
    """Medium grader: multi-city spread containment + vaccination."""
    avg_infected = sum(c.infected for c in sim.cities) / len(sim.cities)
    avg_vaccinated = sum(c.vaccinated for c in sim.cities) / len(sim.cities)
    total_deaths = sum(c.deaths for c in sim.cities)
    cities_contained = sum(1 for c in sim.cities if c.infected < 0.03)

    infection_score = max(0.0, 1.0 - avg_infected / 0.15)
    vax_score = min(1.0, avg_vaccinated / 0.40)
    containment_score = cities_contained / len(sim.cities)
    death_score = max(0.0, 1.0 - total_deaths / 30_000)

    score = (
        0.35 * infection_score +
        0.25 * vax_score +
        0.25 * containment_score +
        0.15 * death_score
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    return {
        "score": score,
        "success": score >= TASKS["task_2_medium"]["success_threshold"],
        "breakdown": {
            "infection_score": round(infection_score, 4),
            "vax_score": round(vax_score, 4),
            "containment_score": round(containment_score, 4),
            "death_score": round(death_score, 4),
            "avg_infected_pct": round(avg_infected * 100, 2),
            "avg_vaccinated_pct": round(avg_vaccinated * 100, 2),
            "cities_contained": cities_contained,
            "total_deaths": total_deaths,
        }
    }


def grade_task_3(sim: PandemicSimulation) -> Dict[str, Any]:
    """Hard grader: prevent national collapse under severe resource constraints."""
    total_deaths = sum(c.deaths for c in sim.cities)
    max_infected = max(c.infected for c in sim.cities)
    min_infected = min(c.infected for c in sim.cities)
    avg_infected = sum(c.infected for c in sim.cities) / len(sim.cities)
    cities_below_10 = sum(1 for c in sim.cities if c.infected < 0.10)
    collapse = any(c.infected > 0.70 for c in sim.cities)

    death_score = max(0.0, 1.0 - total_deaths / 100_000)
    survival_score = cities_below_10 / len(sim.cities)
    collapse_penalty = 0.0 if not collapse else 0.3
    spread_score = max(0.0, 1.0 - avg_infected / 0.40)

    score = (
        0.40 * death_score +
        0.30 * survival_score +
        0.20 * spread_score -
        collapse_penalty
    )
    score = round(min(max(score, 0.0), 1.0), 4)

    return {
        "score": score,
        "success": score >= TASKS["task_3_hard"]["success_threshold"],
        "breakdown": {
            "death_score": round(death_score, 4),
            "survival_score": round(survival_score, 4),
            "spread_score": round(spread_score, 4),
            "collapse_penalty": collapse_penalty,
            "total_deaths": total_deaths,
            "cities_below_10pct": cities_below_10,
            "avg_infected_pct": round(avg_infected * 100, 2),
            "collapse_occurred": collapse,
        }
    }


GRADERS = {
    "task_1_easy": grade_task_1,
    "task_2_medium": grade_task_2,
    "task_3_hard": grade_task_3,
}
