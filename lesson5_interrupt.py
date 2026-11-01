import operator
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

DB_PATH = Path.home() / "onboarding-data" / "approvals.db"


class OnboardingState(TypedDict):
    name: str
    role: str
    laptop_model: str
    approved: bool
    checklist: Annotated[list[str], operator.add]


def hr_desk(state: OnboardingState):
    return {"checklist": [f"HR welcomed {state['name']}"]}


def manager_approval(state: OnboardingState):
    decision = interrupt({
        "question": f"Approve a {state['laptop_model']} for {state['name']} ({state['role']})?",
        "options": ["approve", "edit", "reject"],
    })
    if decision["type"] == "approve":
        return {"approved": True, "checklist": ["Manager approved laptop"]}
    if decision["type"] == "edit":
        return {
            "approved": True,
            "laptop_model": decision["laptop_model"],
            "checklist": [f"Manager changed laptop to {decision['laptop_model']}"],
        }
    return {"approved": False, "checklist": [f"Laptop rejected: {decision['reason']}"]}


def order_laptop(state: OnboardingState):
    print(f"  [MOCK PURCHASE] {state['laptop_model']} for {state['name']}")
    return {"checklist": [f"Ordered {state['laptop_model']}"]}


def meet_manager(state: OnboardingState):
    return {"checklist": [f"Manager met {state['name']}"]}


def approval_route(state: OnboardingState) -> str:
    return "order_laptop" if state["approved"] else "meet_manager"


builder = StateGraph(OnboardingState)
builder.add_node("hr_desk", hr_desk)
builder.add_node("manager_approval", manager_approval)
builder.add_node("order_laptop", order_laptop)
builder.add_node("meet_manager", meet_manager)

builder.add_edge(START, "hr_desk")
builder.add_edge("hr_desk", "manager_approval")
builder.add_conditional_edges("manager_approval", approval_route, ["order_laptop", "meet_manager"])
builder.add_edge("order_laptop", "meet_manager")
builder.add_edge("meet_manager", END)


def ask_manager(question: dict) -> dict:
    print(f"\n  {question['question']}")
    choice = input("  approve / edit / reject: ").strip().lower()
    if choice == "edit":
        return {"type": "edit", "laptop_model": input("  New model: ").strip()}
    if choice == "reject":
        return {"type": "reject", "reason": input("  Reason: ").strip()}
    return {"type": "approve"}


def print_result(result: dict):
    if "__interrupt__" in result:
        print("  PAUSED - waiting for manager. Run:  python3 lesson5_interrupt.py review", sys.argv[2])
        return
    print("  DONE. Checklist:")
    for item in result["checklist"]:
        print("   -", item)


def main():
    mode, thread_id = sys.argv[1], sys.argv[2]
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH, check_same_thread=False) as conn:
        graph = builder.compile(checkpointer=SqliteSaver(conn))
        config = {"configurable": {"thread_id": thread_id}}

        if mode == "start":
            result = graph.invoke({
                "name": thread_id.capitalize(), "role": "engineer",
                "laptop_model": "MacBook Pro 16", "approved": False, "checklist": [],
            }, config)
        elif mode == "review":
            snap = graph.get_state(config)
            if not snap.interrupts:
                print("  Nothing is waiting for approval.")
                return
            result = graph.invoke(Command(resume=ask_manager(snap.interrupts[0].value)), config)
        else:
            raise SystemExit("Usage: python3 lesson5_interrupt.py start|review <thread_id>")

        print_result(result)
    
    # print(graph.get_graph().draw_mermaid())  # paste into https://mermaid.live


if __name__ == "__main__":
    main()