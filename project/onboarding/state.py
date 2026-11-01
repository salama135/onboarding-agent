import operator
from collections.abc import Mapping
from typing import Annotated, Any, Literal, get_args

from langchain.agents import AgentState

OnboardingTask = Literal["hr_paperwork", "laptop", "accounts", "badge", "security_training", "meet_manager"]
ONBOARDING_TASKS: tuple[str, ...] = get_args(OnboardingTask)


class OnboardingState(AgentState):
    completed_tasks: Annotated[list[str], operator.add]
    blocked_tasks: Annotated[list[dict], operator.add]


def remaining_tasks(state: Mapping[str, Any]) -> list[str]:
    done = set(state.get("completed_tasks", []))
    return [task for task in ONBOARDING_TASKS if task not in done]


def open_blockers(state: Mapping[str, Any]) -> list[dict]:
    done = set(state.get("completed_tasks", []))
    latest = {b["task"]: b for b in state.get("blocked_tasks", [])}
    return [b for task, b in latest.items() if task not in done]


def format_progress(state: Mapping[str, Any]) -> str:
    done = [t for t in ONBOARDING_TASKS if t in set(state.get("completed_tasks", []))]
    blockers = open_blockers(state)
    return (
        f"Done: {', '.join(done) or 'nothing yet'}. "
        f"Remaining: {', '.join(remaining_tasks(state)) or 'nothing - all done!'}. "
        f"Blocked: {'; '.join(f'{b['task']} ({b['reason']})' for b in blockers) or 'none'}."
    )
