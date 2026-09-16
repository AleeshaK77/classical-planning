from pathlib import Path
from planning import Action, PlanningProblem

def parse_facts(lines: list[str]) -> frozenset[str]:
    """Parse a block of facts, ignoring blank lines and comments."""
    facts = []

    for line in lines:
        line = line.strip()

        if not line or line.startswith("#"): continue

        facts.append(line)

    return frozenset(facts)

def parse_action(lines: list[str]) -> Action:
    """
    Parse a single action.

    Expected format:

        ActionName
            PRE:
                Fact1
                Fact2
            ADD:
                Fact3
            DEL:
                Fact4
    """
    name = lines[0].strip()

    preconditions = []
    add_effects = []
    delete_effects = []

    current_section = None

    for line in lines[1:]:
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if line == "PRE:": current_section = preconditions
        elif line == "ADD:": current_section = add_effects
        elif line == "DEL:": current_section = delete_effects
        elif current_section is not None: current_section.append(line)
        else:
            raise ValueError(
                f"Fact encountered outside a section in action '{name}': {line}"
            )

    return Action(
        name = name,
        preconditions = frozenset(preconditions),
        add_effects = frozenset(add_effects),
        delete_effects = frozenset(delete_effects),
    )


def parse_problem(path: str | Path) -> PlanningProblem:
    """
    Parse a planning problem file.

    The file contains:

        INITIAL
            ...
        GOAL
            ...
        ACTIONS
            ...

    Actions use the format documented in parse_action().
    """

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        lines = [line.rstrip() for line in file]

    initial_state = []
    goal = []
    actions = []

    current_section = None
    current_action = []

    def finish_action():
        if current_action:
            actions.append(parse_action(current_action))
            current_action.clear()

    for raw_line in lines:
        line = raw_line.strip()

        if not line or line.startswith("#"): continue

        if line == "INITIAL":
            finish_action()
            current_section = "INITIAL"
            continue

        if line == "GOAL":
            finish_action()
            current_section = "GOAL"
            continue

        if line == "ACTIONS":
            finish_action()
            current_section = "ACTIONS"
            continue

        if current_section == "INITIAL": initial_state.append(line)

        elif current_section == "GOAL": goal.append(line)

        elif current_section == "ACTIONS":
            if not raw_line.startswith((" ", "\t")): finish_action()

            current_action.append(line)

        else: raise ValueError(f"Unexpected line: {line}")

    finish_action()

    return PlanningProblem(
        initial_state = frozenset(initial_state),
        goal = frozenset(goal),
        actions = tuple(actions),
    )