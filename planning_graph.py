from dataclasses import dataclass, field
from planning import Action, PlanningProblem


@dataclass
class ActionLevel:
    """
    One action level in a planning graph.

    Contains all actions that may occur between the preceding
    state level and the next state level, together with mutex
    relations between those actions.
    """

    actions: frozenset[Action]
    mutex: set[frozenset[Action]] = field(default_factory=set)

    def is_mutex(self, action1: Action, action2: Action) -> bool:
        """Return True if two actions are mutex."""
        if action1 == action2: return False

        return frozenset((action1, action2)) in self.mutex

@dataclass
class StateLevel:
    """
    One state level in a planning graph.

    Contains all propositions that could be true at this level,
    together with mutex relations between those propositions.
    """

    facts: frozenset[str]
    mutex: set[frozenset[str]] = field(default_factory=set)

    def is_mutex(self, fact1: str, fact2: str) -> bool:
        """Return True if two facts are mutex."""
        if fact1 == fact2:
            return False

        return frozenset((fact1, fact2)) in self.mutex


class PlanningGraph:
    """
    A Graphplan planning graph.

    The graph alternates between state levels and action levels:

        S0 -> A0 -> S1 -> A1 -> S2 -> ...

    State levels contain propositions that may be true.
    Action levels contain actions whose preconditions are available.

    Mutex relations are maintained at both levels.
    """

    def __init__(self, problem: PlanningProblem):
        self.problem = problem

        self.state_levels: list[StateLevel] = []
        self.action_levels: list[ActionLevel] = []

        #initial state is S0.
        self.state_levels.append(
            StateLevel(frozenset(problem.initial_state))
        )

    def expand(self) -> None:
        """
        Expand the planning graph by one action level and one state level.
        """
        current_state = self.state_levels[-1]

        actions = self._applicable_actions(current_state)
        action_level = ActionLevel(frozenset(actions))

        self._compute_action_mutex(action_level, current_state)

        next_facts = self._compute_next_facts(actions)
        next_state = StateLevel(frozenset(next_facts))

        self._compute_state_mutex(next_state, action_level)

        self.action_levels.append(action_level)
        self.state_levels.append(next_state)

    def build(self, max_levels: int = 50) -> None:
        """
        Build the planning graph until the goals are simultaneously
        achievable, the graph levels off, or max_levels is reached.
        """
        for _ in range(max_levels):
            if self.goals_non_mutex(): return

            if self.is_leveled_off(): return

            self.expand()

    def goals_reached(self) -> bool:
        """
        Return True if every goal proposition appears in the
        current state level.
        """
        current_state = self.state_levels[-1]
        return self.problem.is_goal(current_state.facts)

    def goals_non_mutex(self) -> bool:
        """
        Return True if all goal propositions are present and no
        pair of goals is mutex at the current state level.
        """
        current_state = self.state_levels[-1]

        if not self.problem.is_goal(current_state.facts): return False

        goals = list(self.problem.goal)

        for i in range(len(goals)):
            for j in range(i + 1, len(goals)):
                if current_state.is_mutex(goals[i], goals[j]): return False

        return True

    def is_leveled_off(self) -> bool:
        """
        Return True when two consecutive state levels contain the
        same facts and the same mutex relations.
        """
        if len(self.state_levels) < 2: return False

        previous = self.state_levels[-2]
        current = self.state_levels[-1]

        return (
            previous.facts == current.facts
            and previous.mutex == current.mutex
        )

    def _applicable_actions(
        self,
        state_level: StateLevel,
    ) -> set[Action]:
        """
        Return actions whose preconditions are all available
        at the current state level.

        Persistence actions are also added so that facts can
        remain true from one level to the next.
        """
        actions = {
            action
            for action in self.problem.actions
            if action.preconditions.issubset(state_level.facts)
        }

        # Add persistence actions for every currently available fact.
        for fact in state_level.facts:
            actions.add(self._make_persistence_action(fact))

        return actions

    @staticmethod
    def _make_persistence_action(fact: str) -> Action:
        """
        Create a no-op/persistence action for a fact.

        A persistence action has the fact as both its precondition
        and positive effect, allowing the proposition to persist
        between state levels.
        """
        return Action(
            name = f"Persist({fact})",
            preconditions = frozenset({fact}),
            add_effects = frozenset({fact}),
            delete_effects = frozenset(),
        )

    @staticmethod
    def _compute_next_facts(
        actions: set[Action],
    ) -> set[str]:
        """Return all facts produced by the actions."""
        facts = set()

        for action in actions:
            facts.update(action.add_effects)

        return facts

    def _compute_action_mutex(
        self,
        action_level: ActionLevel,
        state_level: StateLevel,
    ) -> None:
        """
        Compute mutex relations between actions.

        Two actions are mutex when they exhibit:

        1. Inconsistent effects
        2. Interference
        3. Competing needs
        """
        actions = list(action_level.actions)

        for i in range(len(actions)):
            for j in range(i + 1, len(actions)):
                action1 = actions[i]
                action2 = actions[j]

                if self._inconsistent_effects(action1, action2):
                    self._add_mutex(action_level.mutex, action1, action2)

                elif self._interference(action1, action2):
                    self._add_mutex(action_level.mutex, action1, action2)

                elif self._competing_needs(
                    action1,
                    action2,
                    state_level,
                ):
                    self._add_mutex(action_level.mutex, action1, action2)

    @staticmethod
    def _inconsistent_effects(
        action1: Action,
        action2: Action,
    ) -> bool:
        """
        Return True when one action deletes an effect added by
        the other action.
        """
        return bool(
            action1.add_effects & action2.delete_effects
            or action2.add_effects & action1.delete_effects
        )

    @staticmethod
    def _interference(
        action1: Action,
        action2: Action,
    ) -> bool:
        """
        Return True when one action deletes a precondition of
        the other action.
        """
        return bool(
            action1.delete_effects & action2.preconditions
            or action2.delete_effects & action1.preconditions
        )

    @staticmethod
    def _competing_needs(
        action1: Action,
        action2: Action,
        state_level: StateLevel,
    ) -> bool:
        """
        Return True when a pair of preconditions from the two
        actions are mutex at the preceding state level.
        """
        for precondition1 in action1.preconditions:
            for precondition2 in action2.preconditions:
                if state_level.is_mutex(
                    precondition1,
                    precondition2,
                ):
                    return True

        return False

    @staticmethod
    def _add_mutex(
        mutex_set: set[frozenset],
        item1,
        item2,
    ) -> None:
        """Add an unordered pair to a mutex set."""
        mutex_set.add(frozenset((item1, item2)))

    def _compute_state_mutex(
        self,
        state_level: StateLevel,
        action_level: ActionLevel,
    ) -> None:
        """
        Compute mutex relations between propositions.

        Two propositions are mutex when every pair of actions
        capable of producing them is mutex.
        """
        facts = list(state_level.facts)

        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                fact1 = facts[i]
                fact2 = facts[j]

                producers1 = self._producers(fact1, action_level)
                producers2 = self._producers(fact2, action_level)

                if not producers1 or not producers2:
                    continue

                if all(
                    action_level.is_mutex(a1, a2)
                    for a1 in producers1
                    for a2 in producers2
                ):
                    self._add_mutex(
                        state_level.mutex,
                        fact1,
                        fact2,
                    )

    @staticmethod
    def _producers(
        fact: str,
        action_level: ActionLevel,
    ) -> set[Action]:
        """Return all actions that produce a given fact."""
        return {
            action
            for action in action_level.actions
            if fact in action.add_effects
        }