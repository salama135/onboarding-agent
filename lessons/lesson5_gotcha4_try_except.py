from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

PRICES = {"MacBook Air": 1200}


class State(TypedDict):
    model: str
    approved: bool


def approval_broken(state: State):
    try:
        price = PRICES[state["model"]]
        answer = interrupt(f"Approve {state['model']} (${price})?")
    except Exception as e:
        print(f"  caught {type(e).__name__} - treating as 'no'")
        answer = "no"
    return {"approved": answer == "yes"}


def approval_fixed(state: State):
    try:
        price = PRICES[state["model"]]
    except KeyError:
        price = "unknown"
    answer = interrupt(f"Approve {state['model']} (${price})?")
    return {"approved": answer == "yes"}


for label, node in [("BROKEN", approval_broken), ("FIXED", approval_fixed)]:
    b = StateGraph(State)
    b.add_node("approval", node)
    b.add_edge(START, "approval")
    b.add_edge("approval", END)
    graph = b.compile(checkpointer=InMemorySaver())
    print(f"\n=== {label} ===")
    result = graph.invoke({"model": "MacBook Air", "approved": False}, {"configurable": {"thread_id": "t"}})
    print("  paused?", "__interrupt__" in result, "| approved:", result["approved"])