from dataclasses import dataclass
from pysat.formula import CNF
from planning import Action, PlanningProblem

@dataclass
class SATEncoding:
    """
    A SAT encoding of a classical planning problem.

    Attributes:
        cnf: The resulting CNF formula.
        fact_variables: Mapping from (fact, time) to SAT variable.
        action_variables: Mapping from (action, time) to SAT variable.
    """

    cnf: CNF
    fact_variables: dict[tuple[str, int], int]
    action_variables: dict[tuple[Action, int], int]


class SATEncoder:
    """
    Encode a classical planning problem as a propositional SAT problem.

    For a horizon H:

        fact@0 -> fact@1 -> ... -> fact@H
                    ^
                    |
                 action@t

    Fact variables describe states, while action variables describe
    which actions occur between consecutive states.
    """

    def __init__(self, problem: PlanningProblem):
        self.problem = problem

        self.cnf = CNF()

        self.fact_variables: dict[tuple[str, int], int] = {}
        self.action_variables: dict[tuple[Action, int], int] = {}

        self._next_variable = 1

        self.facts = self._collect_facts()

    def encode(self, horizon: int) -> SATEncoding:
        """
        Construct the SAT encoding for the given planning horizon.

        Args:
            horizon: Number of action steps allowed.

        Returns:
            SATEncoding containing the CNF formula and variable mappings.
        """
        if horizon < 0:
            raise ValueError("Horizon must be non-negative.")

        self.cnf = CNF()
        self.fact_variables.clear()
        self.action_variables.clear()
        self._next_variable = 1

        self._create_fact_variables(horizon)
        self._create_action_variables(horizon)

        self._encode_initial_state()
        self._encode_goal(horizon)
        self._encode_action_preconditions(horizon)
        self._encode_action_effects(horizon)
        self._encode_frame_axioms(horizon)
        self._encode_sequential_actions(horizon)

        return SATEncoding(
            cnf=self.cnf,
            fact_variables=self.fact_variables,
            action_variables=self.action_variables,
        )

    def _collect_facts(self) -> frozenset[str]:
        """Collect every fact appearing anywhere in the problem."""
        facts = set(self.problem.initial_state)
        facts.update(self.problem.goal)

        for action in self.problem.actions:
            facts.update(action.preconditions)
            facts.update(action.add_effects)
            facts.update(action.delete_effects)

        return frozenset(facts)

    def _new_variable(self) -> int:
        """Return a fresh SAT variable number."""
        variable = self._next_variable
        self._next_variable += 1
        return variable

    def _create_fact_variables(self, horizon: int) -> None:
        """
        Create one Boolean variable for every fact at every state level.
        """
        for time in range(horizon + 1):
            for fact in sorted(self.facts):
                self.fact_variables[(fact, time)] = self._new_variable()

    def _create_action_variables(self, horizon: int) -> None:
        """
        Create one Boolean variable for every action at every action level.

        An action at time t transforms state t into state t+1.
        """
        for time in range(horizon):
            for action in self.problem.actions:
                self.action_variables[(action, time)] = self._new_variable()

    def _fact_var(self, fact: str, time: int) -> int:
        """Return the SAT variable representing a fact at a time."""
        return self.fact_variables[(fact, time)]

    def _action_var(self, action: Action, time: int) -> int:
        """Return the SAT variable representing an action at a time."""
        return self.action_variables[(action, time)]

    def _encode_initial_state(self) -> None:
        """
        Encode the complete initial state.

        Initial facts are true and every other known fact is false.
        """
        for fact in self.facts:
            variable = self._fact_var(fact, 0)

            if fact in self.problem.initial_state:
                self.cnf.append([variable])
            else:
                self.cnf.append([-variable])

    def _encode_goal(self, horizon: int) -> None:
        """Require every goal fact to be true at the final state."""
        for goal in self.problem.goal:
            self.cnf.append([self._fact_var(goal, horizon)])

    def _encode_action_preconditions(self, horizon: int) -> None:
        """
        Encode:

            action_t -> precondition_t

        for every action precondition.
        """
        for time in range(horizon):
            for action in self.problem.actions:
                action_variable = self._action_var(action, time)

                for precondition in action.preconditions:
                    fact_variable = self._fact_var(precondition, time)

                    self.cnf.append([
                        -action_variable,
                        fact_variable,
                    ])

    def _encode_action_effects(self, horizon: int) -> None:
        """
        Encode the direct effects of each action.

        Add effects:

            action_t -> fact_{t+1}

        Delete effects:

            action_t -> not fact_{t+1}
        """
        for time in range(horizon):
            for action in self.problem.actions:
                action_variable = self._action_var(action, time)

                for fact in action.add_effects:
                    fact_variable = self._fact_var(fact, time + 1)

                    self.cnf.append([
                        -action_variable,
                        fact_variable,
                    ])

                for fact in action.delete_effects:
                    fact_variable = self._fact_var(fact, time + 1)

                    self.cnf.append([
                        -action_variable,
                        -fact_variable,
                    ])

    def _encode_frame_axioms(self, horizon: int) -> None:
        """
        Encode frame axioms describing how facts persist.

        A fact can change only if some action explains that change.

        For a fact p:

            p_t and no deleting action
                -> p_{t+1}

            p_{t+1} and no adding action
                -> p_t

        This prevents the SAT solver from arbitrarily making facts
        appear or disappear between states.
        """
        for time in range(horizon):
            for fact in self.facts:
                adders = [
                    self._action_var(action, time)
                    for action in self.problem.actions
                    if fact in action.add_effects
                ]

                deleters = [
                    self._action_var(action, time)
                    for action in self.problem.actions
                    if fact in action.delete_effects
                ]

                current = self._fact_var(fact, time)
                next_state = self._fact_var(fact, time + 1)

                # If p is true now and no action deletes it,
                # p must remain true.
                self.cnf.append(
                    [-current, next_state] + deleters
                )

                # If p is true next and no action adds it,
                # p must have been true already.
                self.cnf.append(
                    [-next_state, current] + adders
                )

    def _encode_sequential_actions(self, horizon: int) -> None:
        """
        Require at most one ordinary action at each timestep.

        This gives the encoder sequential-plan semantics.
        """
        for time in range(horizon):
            actions = list(self.problem.actions)

            for i in range(len(actions)):
                for j in range(i + 1, len(actions)):
                    action1 = self._action_var(actions[i], time)
                    action2 = self._action_var(actions[j], time)

                    # ¬action1 ∨ ¬action2
                    self.cnf.append([
                        -action1,
                        -action2,
                    ])

    def decode_plan(
        self,
        model: list[int],
        horizon: int,
    ) -> list[Action]:
        """
        Extract a sequential plan from a satisfying SAT model.

        Args:
            model: SAT solver model represented as a list of literals.
            horizon: Horizon used to create the encoding.

        Returns:
            Actions selected by the satisfying assignment, in order.
        """
        true_variables = {
            variable
            for variable in model
            if variable > 0
        }

        plan = []

        for time in range(horizon):
            for action in self.problem.actions:
                variable = self._action_var(action, time)

                if variable in true_variables:
                    plan.append(action)

        return plan
