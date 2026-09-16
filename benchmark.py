import time
from pathlib import Path
from graphplan import Graphplan
from parser import parse_problem
from satplan import SATPlan


def benchmark_graphplan(problem_path: str | Path) -> dict:
    """
    Run Graphplan on a problem and collect basic performance metrics.
    """
    problem = parse_problem(problem_path)
    planner = Graphplan(problem)

    start = time.perf_counter()
    plan = planner.solve()
    elapsed = time.perf_counter() - start

    return {
        "planner": "Graphplan",
        "solved": plan is not None,
        "plan_length": len(plan) if plan is not None else None,
        "runtime": elapsed,
        "state_levels": len(planner.graph.state_levels),
        "action_levels": len(planner.graph.action_levels),
    }


def benchmark_satplan(
    problem_path: str | Path,
    max_horizon: int = 50,
) -> dict:
    """
    Run SATPlan on a problem and collect basic performance metrics.
    """
    problem = parse_problem(problem_path)
    planner = SATPlan(problem)

    start = time.perf_counter()
    plan = planner.solve(max_horizon=max_horizon)
    elapsed = time.perf_counter() - start

    return {
        "planner": "SATPlan",
        "solved": plan is not None,
        "plan_length": len(plan) if plan is not None else None,
        "runtime": elapsed,
        "state_levels": None,
        "action_levels": None,
    }


def benchmark_problem(
    problem_path: str | Path,
    max_horizon: int = 50,
) -> list[dict]:
    """
    Benchmark both planners on the same planning problem.
    """
    graphplan_result = benchmark_graphplan(problem_path)
    satplan_result = benchmark_satplan(
        problem_path,
        max_horizon=max_horizon,
    )

    return [
        graphplan_result,
        satplan_result,
    ]


if __name__ == "__main__":
    problem_path = Path("domains/blocks_world/problem.txt")

    results = benchmark_problem(problem_path)

    print("Benchmark results:")

    for result in results:
        print(f"\n{result['planner']}")

        for key, value in result.items():
            if key != "planner":
                print(f"  {key}: {value}")

