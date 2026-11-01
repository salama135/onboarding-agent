import operator
from dataclasses import dataclass
from typing import Annotated

from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_ollama import ChatOllama
from langgraph.types import Command


class OnboardingAgentState(AgentState):
    employee_name: str
    completed_tasks: Annotated[list[str], operator.add]
    blocked_tasks: Annotated[list[str], operator.add]


@dataclass
class Context:
    company: str


@tool
def get_progress(runtime: ToolRuntime[Context, OnboardingAgentState]) -> str:
    """Show completed and blocked onboarding tasks so far."""
    done = runtime.state.get("completed_tasks", [])
    blocked = runtime.state.get("blocked_tasks", [])
    return (
        f"{runtime.state['employee_name']} completed: {', '.join(done) or 'nothing yet'}. "
        f"Blocked: {', '.join(blocked) or 'none'}."
    )

@tool
def complete_task(task: str, runtime: ToolRuntime[Context, OnboardingAgentState]) -> Command:
    """Mark one onboarding task as completed. Call once per task."""
    return Command(update={
        "completed_tasks": [task],
        "messages": [ToolMessage(
            f"Marked '{task}' as done at {runtime.context.company}.",
            tool_call_id=runtime.tool_call_id,
        )],
    })

@tool
def report_blocker(task: str, reason: str, runtime: ToolRuntime[Context, OnboardingAgentState]) -> Command:
    """Record an onboarding task the employee could not finish, and why."""
    return Command(update={
        "blocked_tasks": [f"{task}: {reason}"],
        "messages": [ToolMessage(
            f"Logged blocker '{task}' for {runtime.context.company} IT support.",
            tool_call_id=runtime.tool_call_id,
        )],
    })

agent = create_agent(
    model=ChatOllama(model="llama3.2"),
    tools=[get_progress, complete_task, report_blocker],
    system_prompt=(
        "You are an onboarding assistant. "
        "If the user finished a task, call complete_task once per task. "
        "If the user could not finish a task, call report_blocker with the task and reason. "
        "Then call get_progress and summarize in 2 sentences."
    ),
    state_schema=OnboardingAgentState,
    context_schema=Context,
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "I did my HR paperwork, but I couldn't get my VPN working."}],
            "employee_name": "Sara",
            "completed_tasks": [],
            "blocked_tasks": [],
        },
        context=Context(company="Deloitte"),
    )
    print("Completed:", result["completed_tasks"])
    print("Blocked:  ", result["blocked_tasks"])
    print("Agent says:", result["messages"][-1].content)