from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt


class State(TypedDict):
    approved: bool


def approval(state: State):
    answer = interrupt("Approve the laptop?")
    return {"approved": answer == "yes"}


builder = StateGraph(State)
builder.add_node("approval", approval)
builder.add_edge(START, "approval")
builder.add_edge("approval", END)

sara = {"configurable": {"thread_id": "sara"}}
omar = {"configurable": {"thread_id": "omar"}}

print("1) No checkpointer at all:")
try:
    no_saver = builder.compile()
    no_saver.invoke({"approved": False})
    no_saver.invoke(Command(resume="yes"))
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

print("\n2) Checkpointer, but no thread_id in config:")
graph = builder.compile(checkpointer=InMemorySaver())
try:
    graph.invoke({"approved": False})
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

print("\n3) Wrong thread_id:")
graph.invoke({"approved": False}, sara)
print("  sara waiting:", bool(graph.get_state(sara).interrupts))
print("  omar waiting:", bool(graph.get_state(omar).interrupts))

print("\n4) InMemorySaver after a 'restart':")
restarted = builder.compile(checkpointer=InMemorySaver())  # new saver = new empty RAM
print("  sara waiting after restart:", bool(restarted.get_state(sara).interrupts))