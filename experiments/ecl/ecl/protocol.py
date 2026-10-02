"""Evaluation plans and summaries; no simulator or model dependencies."""
import math
import random
import statistics


def initial_state_plan(count, trials, seed, task_id):
    if count < 1 or trials < 1:
        raise ValueError("Initial state and trial counts must be positive")
    rng = random.Random(seed * 1000003 + task_id)
    plan = []
    while len(plan) < trials:
        cycle = list(range(count))
        rng.shuffle(cycle)
        plan.extend(cycle)
    return plan[:trials]


def aggregate(records, seeds, task_ids, trials):
    expected = {(seed, task, episode) for seed in seeds for task in task_ids for episode in range(trials)}
    actual = [(r["seed"], r["task_id"], r["episode"]) for r in records]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError("Missing, duplicate or unexpected evaluation episodes")
    tasks = {}
    for task in task_ids:
        rows = [r for r in records if r["task_id"] == task]
        per_seed = {str(s): statistics.mean(float(r["success"]) for r in rows if r["seed"] == s) for s in seeds}
        rate = statistics.mean(per_seed.values())
        tasks[str(task)] = {
            "task": rows[0]["task"], "episodes": len(rows),
            "successes": sum(bool(r["success"]) for r in rows),
            "per_seed": per_seed, "mean": rate,
            "seed_std": statistics.stdev(per_seed.values()) if len(seeds) > 1 else 0.0,
            "binomial_std": math.sqrt(rate * (1 - rate) / len(rows)),
        }
    macro = {str(s): statistics.mean(tasks[str(t)]["per_seed"][str(s)] for t in task_ids) for s in seeds}
    return {"seeds": seeds, "trials_per_task_per_seed": trials, "tasks": tasks,
            "overall": {"per_seed": macro, "mean": statistics.mean(macro.values()),
                        "seed_std": statistics.stdev(macro.values()) if len(seeds) > 1 else 0.0}}
