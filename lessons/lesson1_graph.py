from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

llm = ChatOllama(model="llama3.2")


# 1. The clipboard
class OnboardingState(TypedDict):
    name: str
    has_laptop: bool
    checklist: list[str]


# 2. The desks (nodes): receive the state, return only the keys they change
def hr_desk(state: OnboardingState):
    return {"checklist": state["checklist"] + [f"HR welcomed {state['name']}"]}

def order_laptop(state: OnboardingState):
    return {"checklist": state["checklist"] + ["Laptop ordered"], "has_laptop": True}

def setup_accounts(state: OnboardingState):
    return {"checklist": state["checklist"] + ["Email & Slack accounts created"]}

def meet_manager(state: OnboardingState):
    reply = llm.invoke(f"Write a 2-sentence friendly welcome from a manager to {state['name']}.")
    return {"checklist": state["checklist"] + [f"Manager: {reply.content}"]}


# 3. The decision (router): returns the name of the next node
def laptop_check(state: OnboardingState) -> str:
    return "setup_accounts" if state["has_laptop"] else "order_laptop"


# 4. Connect everything
builder = StateGraph(OnboardingState)
builder.add_node("hr_desk", hr_desk)
builder.add_node("order_laptop", order_laptop)
builder.add_node("setup_accounts", setup_accounts)
builder.add_node("meet_manager", meet_manager)

builder.add_edge(START, "hr_desk")
builder.add_conditional_edges("hr_desk", laptop_check, ["order_laptop", "setup_accounts"])
builder.add_edge("order_laptop", "setup_accounts")
builder.add_edge("setup_accounts", "meet_manager")
builder.add_edge("meet_manager", END)

graph = builder.compile()

# 5. Run and visualize
if __name__ == "__main__":
    # result = graph.invoke({"name": "Sara", "has_laptop": False, "checklist": []})
    result = graph.invoke({"name": "Sara", "has_laptop": True, "checklist": []})
    print(result["checklist"])
    for item in result["checklist"]:
        print("-", item)

    # print(graph.get_graph().draw_mermaid())  # paste into https://mermaid.live