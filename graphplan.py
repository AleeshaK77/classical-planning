from planning import Action, PlanningProblem
from planning_graph import PlanningGraph


class Graphplan:
    """
    Graphplan planner.

    Builds a planning graph and then performs backward search
    through the graph to extract a sequence of actions.
    """

    def __init__(self, problem: PlanningProblem):
        self.problem = problem
        self.graph = PlanningGraph(problem)

        # cache failed subproblems 
        self.memo: set[tuple[int, frozenset[str]]] = set()

    def solve(self, max_levels: int = 50) -> list[Action] | None:
        """
        Find a plan using Graphplan.

        Returns:
            A list of actions forming a valid plan, or None if
            no plan is found within max_levels.
        """
        self.graph.build(max_levels)

        # The graph may stop because it leveled off before the
        # goals became achievable.
        if not self.graph.goals_non_mutex():
            return None

        level = len(self.graph.state_levels) - 1

        plan = self._extract_plan(
            frozenset(self.problem.goal),
            level,
        )

        if plan is None:
            return None

        return list(plan)

    def _extract_plan(
        self,
        goals: frozenset[str],
        level: int,
    ) -> tuple[Action, ...] | None:
        """
        Recursively extract a plan for the given goals at a state level.

        The search proceeds backward through the planning graph.
        """
        if not goals:
            return ()

        # At S0, the goals must already be true.
        if level == 0:
            if goals.issubset(self.graph.state_levels[0].facts):
                return ()
            return None

        memo_key = (level, goals)

        if memo_key in self.memo:
            return None

        action_level = self.graph.action_levels[level - 1]

        # Find actions capable of achieving each goal.
        candidates = {
            goal: [
                action
                for action in action_level.actions
                if goal in action.add_effects
            ]
            for goal in goals
        }

        # If a goal has no possible producer, this branch fails.
        if any(not actions for actions in candidates.values()):
            self.memo.add(memo_key)
            return None

        goal_list = list(goals)

        result = self._select_actions(
            goal_list,
            candidates,
            0,
            set(),
            level,
        )

        if result is None:
            self.memo.add(memo_key)

        return result

    def _select_actions(
        self,
        goals: list[str],
        candidates: dict[str, list[Action]],
        index: int,
        selected: set[Action],
        level: int,
    ) -> tuple[Action, ...] | None:
        """
        Choose a set of non-mutex actions that achieves all goals.

        Once a complete action set has been selected, recursively
        solve the union of its preconditions at the previous level.
        """
        if index == len(goals):
            action_level = self.graph.action_levels[level - 1]

            # Verify that all selected actions are pairwise non-mutex.
            selected_list = list(selected)

            for i in range(len(selected_list)):
                for j in range(i + 1, len(selected_list)):
                    if action_level.is_mutex(
                        selected_list[i],
                        selected_list[j],
                    ):
                        return None

            # Collect the preconditions that must hold at the
            # previous state level.
            preconditions = frozenset(
                precondition
                for action in selected
                for precondition in action.preconditions
            )

            previous_plan = self._extract_plan(
                preconditions,
                level - 1,
            )

            if previous_plan is None:
                return None

            return previous_plan + tuple(selected)

        goal = goals[index]

        if any(goal in action.add_effects for action in selected):
            return self._select_actions(
                goals,
                candidates,
                index + 1,
                selected,
                level,
            )

        for action in candidates[goal]:
            #check mutex against actions already selected.
            action_level = self.graph.action_levels[level - 1]

            if any(
                action_level.is_mutex(action, other)
                for other in selected
            ):
                continue

            selected.add(action)

            result = self._select_actions(
                goals,
                candidates,
                index + 1,
                selected,
                level,
            )

            if result is not None:
                return result

            selected.remove(action)

        return None