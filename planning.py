from dataclasses import dataclass
from typing import FrozenSet, Iterable


@dataclass(frozen = True)
class Action:
    """
    A STRIPS-style planning action.

    An action can be applied to a state when all of its preconditions
    are satisfied. Applying the action removes its delete effects and
    adds its add effects.
    """

    name: str
    preconditions: FrozenSet[str]
    add_effects: FrozenSet[str]
    delete_effects: FrozenSet[str]

    def is_applicable(self, state: FrozenSet[str]) -> bool:
        """Return True if all preconditions are satisfied in the state."""
        return self.preconditions.issubset(state)

    def apply(self, state: FrozenSet[str]) -> FrozenSet[str]:
        """
        Apply the action to a state and return the resulting state.

        Raises:
            ValueError: If the action is not applicable.
        """
        if not self.is_applicable(state):
            raise ValueError(
                f"Action '{self.name}' is not applicable in the given state."
            )

        return (state - self.delete_effects) | self.add_effects


@dataclass(frozen=True)
class PlanningProblem:
    """
    A classical planning problem.

    Attributes:
        initial_state: Facts that are true initially.
        goal: Facts that must be true for the problem to be solved.
        actions: Actions available to the planner.
    """

    initial_state: FrozenSet[str]
    goal: FrozenSet[str]
    actions: tuple[Action, ...]

    def is_goal(self, state: FrozenSet[str]) -> bool:
        """Return True if the state satisfies the goal."""
        return self.goal.issubset(state)

    def applicable_actions(
        self, state: FrozenSet[str]
    ) -> tuple[Action, ...]:
        """Return all actions applicable in the given state."""
        return tuple(
            action for action in self.actions
            if action.is_applicable(state)
        )

    def successors(
        self, state: FrozenSet[str]
    ) -> Iterable[tuple[Action, FrozenSet[str]]]:
        """
        Generate successor states.

        Yields:
            (action, resulting_state) pairs for every applicable action.
        """
        for action in self.applicable_actions(state):
            yield action, action.apply(state)

