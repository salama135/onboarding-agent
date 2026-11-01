from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.types import Command

from onboarding.context import OnboardingContext
from onboarding.memory import delete_fact, save_fact
from onboarding.state import OnboardingState, OnboardingTask, format_progress

Runtime = ToolRuntime[OnboardingContext, OnboardingState]


@tool
def get_progress(runtime: Runtime) -> str:
    """Show the employee's onboarding checklist: done, remaining, and blocked tasks."""
    return format_progress(runtime.state)


@tool
def complete_task(task: OnboardingTask, runtime: Runtime) -> Command:
    """Mark one onboarding checklist task as done. Call once per finished task."""
    return Command(update={
        "completed_tasks": [task],
        "messages": [ToolMessage(f"Marked '{task}' as done.", tool_call_id=runtime.tool_call_id)],
    })


@tool
def report_blocker(task: OnboardingTask, reason: str, runtime: Runtime) -> Command:
    """Record a checklist task the employee could not finish, and why."""
    return Command(update={
        "blocked_tasks": [{"task": task, "reason": reason.strip()[:200]}],
        "messages": [ToolMessage(f"Logged blocker on '{task}'.", tool_call_id=runtime.tool_call_id)],
    })


@tool
def remember(key: str, value: str, runtime: Runtime) -> str:
    """Save a lasting fact about the employee for future conversations.
    Use short keys like: language, diet, office, team, preferred_name."""
    try:
        saved_key = save_fact(runtime.store, runtime.context.user_id, key, value)
    except ValueError as err:
        return str(err)
    return f"Remembered {saved_key}."


@tool
def forget(key: str, runtime: Runtime) -> str:
    """Delete a saved fact about the employee when they ask you to forget it."""
    try:
        delete_fact(runtime.store, runtime.context.user_id, key)
    except ValueError as err:
        return str(err)
    return f"Forgot {key}."


SUPERVISOR_TOOLS = [get_progress, complete_task, report_blocker, remember, forget]
