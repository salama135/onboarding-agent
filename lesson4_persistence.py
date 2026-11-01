import operator
import sqlite3
from pathlib import Path
from typing import Annotated

from langchain.agents import AgentState, create_agent
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

DB_PATH = Path.home() / "onboarding-data" / "checkpoints.db"  # Linux FS: SQLite locking is unreliable on /mnt/c


class OnboardingAgentState(AgentState):
    completed_tasks: Annotated[list[str], operator.add]


@tool
def complete_task(task: str, runtime: ToolRuntime) -> Command:
    """Mark one onboarding task as completed. Call once per task."""
    return Command(update={
        "completed_tasks": [task],
        "messages": [ToolMessage(f"Marked '{task}' as done.", tool_call_id=runtime.tool_call_id)],
    })


@tool
def get_progress(runtime: ToolRuntime) -> str:
    """List the onboarding tasks completed so far."""
    return ", ".join(runtime.state.get("completed_tasks", [])) or "nothing yet"


def build_agent(checkpointer):
    return create_agent(
        model=ChatOllama(model="qwen2.5:7b", num_ctx=8192),
        tools=[complete_task, get_progress],
        system_prompt=(
            "You are an onboarding assistant. Keep answers to 1-2 sentences. "
            "When the user finishes a task, call complete_task once per task. "
            "When asked about progress, call get_progress."
            "Split combined tasks: 'X and Y' means two separate complete_task calls."
        ),
        state_schema=OnboardingAgentState,
        checkpointer=checkpointer,
    )


def show_state(agent, config):
    snap = agent.get_state(config)
    print(f"  messages: {len(snap.values.get('messages', []))}")
    print(f"  completed_tasks: {snap.values.get('completed_tasks', [])}")
    print(f"  next: {snap.next or '(turn finished)'}")


def show_history(agent, config):
    for i, snap in enumerate(agent.get_state_history(config)):
        meta = snap.metadata or {}
        msgs = snap.values.get("messages", [])
        print(
            f"  [{i:>2}] step={meta.get('step'):>2} source={meta.get('source', ''):<6} "
            f"next={str(snap.next):<16} msgs={len(msgs):>2} "
            f"tasks={snap.values.get('completed_tasks', [])} | {describe(msgs[-1]) if msgs else ''}"
        )


def describe(msg) -> str:
    if getattr(msg, "tool_calls", None):
        return "CALL " + ", ".join(f"{c['name']}({c['args']})" for c in msg.tool_calls)
    text = str(msg.content)[:40].replace("\n", " ")
    return f"{msg.type}: {text}"


def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    agent = build_agent(SqliteSaver(conn))

    thread_id = input("Thread id (e.g. sara): ").strip() or "sara"
    base_config = {"configurable": {"thread_id": thread_id}}
    config = base_config

    existing = agent.get_state(base_config).values
    if existing:
        print(f"Resuming '{thread_id}': {len(existing['messages'])} messages, "
              f"completed: {existing.get('completed_tasks', [])}")
    else:
        print(f"New conversation '{thread_id}'")
    print("Commands: /state  /history  /rewind N  quit")

    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        if text == "/state":
            show_state(agent, base_config)
            continue
        if text == "/history":
            show_history(agent, base_config)
            continue
        if text.startswith("/rewind "):
            snaps = list(agent.get_state_history(base_config))
            config = snaps[int(text.split()[1])].config
            print("  Next message will branch from that checkpoint.")
            continue

        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
        config = base_config  # after a fork, the new branch is the thread's latest checkpoint
        print("Agent:", result["messages"][-1].content)
        print("  [completed_tasks]", result.get("completed_tasks", []))

    conn.close()


if __name__ == "__main__":
    main()