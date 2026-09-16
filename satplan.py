from planning import Action, PlanningProblem
from sat_encoder import SATEncoder

class SATPlan:
    """
    SATPlan planner.

    Iteratively encodes the planning problem at increasing horizons
    and uses a SAT solver to determine whether a plan of that length
    exists.

    The first satisfiable horizon gives a shortest sequential plan.
    """

    def __init__(self, problem: PlanningProblem):
        self.problem = problem

    def solve(self, max_horizon: int = 50) -> list[Action] | None:
        """
        Find a shortest plan up to the given maximum horizon.

        Args:
            max_horizon: Largest number of action steps to consider.

        Returns:
            A list of actions forming a valid plan, or None if no
            plan is found within max_horizon.
        """
        for horizon in range(max_horizon + 1):
            encoder = SATEncoder(self.problem)
            encoding = encoder.encode(horizon)

            plan = self._solve_encoding(
                encoder,
                encoding,
                horizon,
            )

            if plan is not None:
                return plan

        return None

    @staticmethod
    def _solve_encoding(
        encoder: SATEncoder,
        encoding,
        horizon: int,
    ) -> list[Action] | None:
        """
        Solve a single SAT encoding and decode the resulting plan.

        Returns:
            A decoded plan if the formula is satisfiable, otherwise None.
        """
        # Imported here so that the core planning representation and
        # encoder do not depend directly on a particular SAT solver.
        from pysat.solvers import Solver

        with Solver(bootstrap_with=encoding.cnf.clauses) as solver:
            if not solver.solve():
                return None

            model = solver.get_model()

        plan = encoder.decode_plan(model, horizon)

        if not SATPlan._validate_plan(plan, encoder.problem):
            raise RuntimeError(
                "SAT solver returned a plan that does not solve the problem."
            )

        return plan

    @staticmethod
    def _validate_plan(
        plan: list[Action],
        problem: PlanningProblem,
    ) -> bool:
        """
        Verify that a decoded plan is executable and reaches the goal.
        """
        state = problem.initial_state

        for action in plan:
            if not action.is_applicable(state):
                return False

            state = action.apply(state)

        return problem.is_goal(state)