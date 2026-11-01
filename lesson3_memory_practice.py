import operator
from typing import Annotated, Literal, get_args

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware, before_model, dynamic_prompt, wrap_model_call,
)
from langchain.messages import HumanMessage, RemoveMessage, ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.messages.utils import count_tokens_approximately
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


TaskName = Literal[
    "HR paperwork",
    "Laptop setup",
    "Email & Slack access",
    "VPN access",
    "Security training",
    "Meet manager",
    "Meet the team",
]
ONBOARDING_TASKS = list(get_args(TaskName))

class OnboardingAgentState(AgentState):
    role: str
    employee_name: str
    completed_tasks: Annotated[list[str], add_unique]


STRATEGY = "facts"          # try: "none", "window", "summary"
KEEP_LAST_MESSAGES = 6
MAX_STATE_MESSAGES = 10
MAX_TOKENS = 1000

model = ChatOllama(model="llama3.2", num_ctx=8192)


def add_unique(existing: list[str], new: list[str]) -> list[str]:
    return existing + [t for t in new if t not in existing]


def recent_turns(messages, keep):
    start = max(len(messages) - keep, 0)
    while start > 0 and not isinstance(messages[start], HumanMessage):  # never start mid tool-call
        start -= 1
    return messages[start:]


def recent_turns_by_tokens(messages, max_tokens):
    start, total = len(messages), 0
    while start > 0:
        cost = count_tokens_approximately([messages[start - 1]])
        if total + cost > max_tokens:
            break
        total += cost
        start -= 1
    while start < len(messages) and not isinstance(messages[start], HumanMessage):
        start += 1
    if start == len(messages):  # newest turn alone exceeds budget: keep it anyway
        start = max(i for i, m in enumerate(messages) if isinstance(m, HumanMessage))
    return messages[start:]


@before_model
def delete_old_messages(state, runtime):
    messages = state["messages"]
    if len(messages) <= MAX_STATE_MESSAGES:
        return None
    keep = recent_turns(messages, KEEP_LAST_MESSAGES)
    to_delete = messages[: len(messages) - len(keep)]
    return {"messages": [RemoveMessage(id=m.id) for m in to_delete]}


@wrap_model_call
def token_window(request, handler):
    return handler(request.override(messages=recent_turns_by_tokens(request.messages, MAX_TOKENS)))


@wrap_model_call
def sliding_window(request, handler):
    return handler(request.override(messages=recent_turns(request.messages, KEEP_LAST_MESSAGES)))


@wrap_model_call
def log_context_size(request, handler):
    tokens = count_tokens_approximately(request.messages)
    print(f"  [model sees] {len(request.messages)} messages, ~{tokens} tokens")
    return handler(request)


@dynamic_prompt
def onboarding_facts(request) -> str:
    s = request.state
    done = s.get("completed_tasks", [])
    pending = [t for t in ONBOARDING_TASKS if t not in done]
    return (
        "You are a friendly onboarding assistant. Keep answers to 1-2 sentences.\n"
        f"Employee: {s.get('employee_name', 'unknown')} ({s.get('role', 'unknown')}).\n"
        f"Completed tasks: {', '.join(done) or 'none'}.\n"
        f"Pending tasks: {', '.join(pending) or 'none'}.\n"
        "Rules:\n"
        "- Only call complete_task when the user clearly says they FINISHED a pending task.\n"
        "- Never call complete_task to answer a question. Answer from the lists above.\n"
        "- The only valid tasks are the ones listed. Never invent tasks.\n"
        "- When asked what to do next, suggest pending tasks."
    )


@tool
def complete_task(task: TaskName, runtime: ToolRuntime[None, OnboardingAgentState]) -> Command | str:
    """Mark one onboarding task as completed.
    Only call this when the user explicitly says they FINISHED the task. Never call it for questions."""
    if task in runtime.state.get("completed_tasks", []):
        return f"'{task}' is already completed. Do not mark it again."
    return Command(update={
        "completed_tasks": [task],
        "messages": [ToolMessage(f"Marked '{task}' as done.", tool_call_id=runtime.tool_call_id)],
    })


summarizer = SummarizationMiddleware(
    model=ChatOllama(model="qwen2.5:7b", num_ctx=8192),
    trigger=("tokens", 600),          # deliberately tiny so you see it fire
    keep=("messages", KEEP_LAST_MESSAGES),
)

middleware = {
    "none": [log_context_size],
    "window": [sliding_window, log_context_size],
    "summary": [summarizer, log_context_size],
    "delete": [delete_old_messages, log_context_size],
    "tokens": [token_window, log_context_size],    
    "facts": [onboarding_facts, sliding_window, log_context_size],
}[STRATEGY]


agent = create_agent(
    model=model,
    tools=[complete_task],
    middleware=middleware,
    state_schema=OnboardingAgentState,
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "sara-day-1"}}


if __name__ == "__main__":
    print(f"Strategy: {STRATEGY}. Type 'quit' to stop.")
    first_turn = True
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        payload = {"messages": [{"role": "user", "content": text}]}
        if first_turn:
            payload |= {"employee_name": "Sara", "role": "data engineer", "completed_tasks": []}
            first_turn = False
        result = agent.invoke(payload, config)
        print("Agent:", result["messages"][-1].content)
        print(f"  [state saved] {len(result['messages'])} messages | completed: {result['completed_tasks']}")