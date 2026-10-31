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


@dataclass
class Context:
    company: str


@tool
def get_progress(runtime: ToolRuntime[Context, OnboardingAgentState]) -> str:
    """Show which onboarding tasks the employee has completed so far."""
    done = runtime.state.get("completed_tasks", [])
    return f"{runtime.state['employee_name']} completed: {', '.join(done) or 'nothing yet'}"


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


agent = create_agent(
    model=ChatOllama(model="llama3.2"),  # llama3.2 works but is weaker at tool calling
    tools=[get_progress, complete_task],
    system_prompt=(
        "You are an onboarding assistant. When the user says they finished tasks, "
        "call complete_task once per task, then call get_progress and summarize."
    ),
    state_schema=OnboardingAgentState,
    context_schema=Context,
)


if __name__ == "__main__":
    result = agent.invoke(
        {
            "messages": [{"role": "user", "content": "I finished my HR paperwork and laptop setup."}],
            "employee_name": "Sara",
            "completed_tasks": [],
        },
        context=Context(company="Deloitte"),
    )
    print("State field completed_tasks:", result["completed_tasks"])
    print("Agent says:", result["messages"][-1].content)