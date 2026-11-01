import sqlite3
from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

DB = sqlite3.connect(":memory:", check_same_thread=False)


class State(TypedDict):
    employee_id: str
    approved: bool


def approval_broken(state: State):
    answer = interrupt({"question": "Approve?", "db": DB})
    return {"approved": answer == "yes"}


def approval_fixed(state: State):
    answer = interrupt({"question": "Approve?", "employee_id": state["employee_id"]})
    return {"approved": answer == "yes"}


for label, node in [("BROKEN", approval_broken), ("FIXED", approval_fixed)]:
    b = StateGraph(State)
    b.add_node("approval", node)
    b.add_edge(START, "approval")
    b.add_edge("approval", END)
    graph = b.compile(checkpointer=InMemorySaver())
    print(f"\n=== {label} ===")
    try:
        result = graph.invoke({"employee_id": "E123", "approved": False}, {"configurable": {"thread_id": "t"}})
        print("  paused with payload:", result["__interrupt__"][0].value)
    except Exception as e:
        print(f"  {type(e).__name__}: {e}")