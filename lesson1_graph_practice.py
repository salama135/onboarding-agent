from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

llm = ChatOllama(model="llama3.2")

MIN_WELCOME_LENGTH = 20
MAX_WELCOME_ATTEMPTS = 3


class OnboardingState(TypedDict):
    name: str
    role: str
    has_laptop: bool
    checklist: list[str]
    welcome_note: str
    welcome_attempts: int


def hr_desk(state: OnboardingState):
    return {"checklist": state["checklist"] + [f"HR welcomed {state['name']}"]}

def order_laptop(state: OnboardingState):
    return {"checklist": state["checklist"] + ["Laptop ordered"], "has_laptop": True}

def setup_accounts(state: OnboardingState):
    return {"checklist": state["checklist"] + ["Email & Slack accounts created"]}

def assign_buddy(state: OnboardingState):
    return {"checklist": state["checklist"] + [f"Engineering buddy assigned to {state['name']}"]}

def meet_manager(state: OnboardingState):
    reply = llm.invoke(f"Write a 2-sentence friendly welcome from a manager to {state['name']}.")
    return {"welcome_note": reply.content, "welcome_attempts": state["welcome_attempts"] + 1}


def laptop_check(state: OnboardingState) -> str:
    return "setup_accounts" if state["has_laptop"] else "order_laptop"

def assign_buddy_check(state: OnboardingState) -> str:
    return "assign_buddy" if state["role"] == "engineer" else "meet_manager"

def welcome_check(state: OnboardingState) -> str:
    too_short = len(state["welcome_note"]) < MIN_WELCOME_LENGTH
    if too_short and state["welcome_attempts"] < MAX_WELCOME_ATTEMPTS:
        return "meet_manager"
    return END


builder = StateGraph(OnboardingState)
builder.add_node("hr_desk", hr_desk)
builder.add_node("order_laptop", order_laptop)
builder.add_node("setup_accounts", setup_accounts)
builder.add_node("assign_buddy", assign_buddy)
builder.add_node("meet_manager", meet_manager)

builder.add_edge(START, "hr_desk")
builder.add_conditional_edges("hr_desk", laptop_check, ["order_laptop", "setup_accounts"])
builder.add_edge("order_laptop", "setup_accounts")
builder.add_conditional_edges("setup_accounts", assign_buddy_check, ["assign_buddy", "meet_manager"])
builder.add_edge("assign_buddy", "meet_manager")
builder.add_conditional_edges("meet_manager", welcome_check, ["meet_manager", END])

graph = builder.compile()


if __name__ == "__main__":
    for role, has_laptop in [("engineer", False), ("tester", True)]:
        result = graph.invoke({
            "name": "Sara", "role": role, "has_laptop": has_laptop,
            "checklist": [], "welcome_note": "", "welcome_attempts": 0,
        })
        print(f"\n=== {role} (attempts: {result['welcome_attempts']}) ===")
        for item in result["checklist"]:
            print("-", item)
        print("Welcome:", result["welcome_note"])

    # print(graph.get_graph().draw_mermaid())  # paste into https://mermaid.live