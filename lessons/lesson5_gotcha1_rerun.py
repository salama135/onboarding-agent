from typing import TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

emails_sent = []


def send_email(to: str):
    emails_sent.append(to)
    print(f"  [EMAIL] approval request sent to {to}")


class State(TypedDict):
    approved: bool


def approval_broken(state: State):
    send_email("manager@corp.com")
    answer = interrupt("Approve the laptop?")
    return {"approved": answer == "yes"}


def notify_manager(state: State):
    send_email("manager@corp.com")
    return {}


def approval_fixed(state: State):
    answer = interrupt("Approve the laptop?")
    return {"approved": answer == "yes"}


def build_broken():
    b = StateGraph(State)
    b.add_node("approval", approval_broken)
    b.add_edge(START, "approval")
    b.add_edge("approval", END)
    return b.compile(checkpointer=InMemorySaver())


def build_fixed():
    b = StateGraph(State)
    b.add_node("notify_manager", notify_manager)
    b.add_node("approval", approval_fixed)
    b.add_edge(START, "notify_manager")
    b.add_edge("notify_manager", "approval")
    b.add_edge("approval", END)
    return b.compile(checkpointer=InMemorySaver())


for label, graph in [("BROKEN", build_broken()), ("FIXED", build_fixed())]:
    emails_sent.clear()
    config = {"configurable": {"thread_id": "t1"}}
    print(f"\n=== {label} ===")
    print("start:")
    graph.invoke({"approved": False}, config)
    print("resume:")
    result = graph.invoke(Command(resume="yes"), config)
    print(graph.get_graph().draw_mermaid())  # paste into https://mermaid.live
    print(f"  approved={result['approved']}, emails sent: {len(emails_sent)}")