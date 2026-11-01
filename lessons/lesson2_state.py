import operator
from dataclasses import dataclass
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime


class OnboardingState(TypedDict):
    name: str                                          # overwrite (default)
    role: str                                          # overwrite (default)
    checklist: Annotated[list[str], operator.add]      # append (reducer)
    # checklist: list[str] # will cause a InvalidUpdateError : because two nodes try to write into the state at once 


@dataclass
class OnboardingContext:
    company: str
    office: str


def hr_desk(state: OnboardingState, runtime: Runtime[OnboardingContext]):
    return {"checklist": [f"HR welcomed {state['name']} to {runtime.context.company}"]}

def it_desk(state: OnboardingState):
    return {"checklist": ["Laptop and email ready"]}

def facilities_desk(state: OnboardingState, runtime: Runtime[OnboardingContext]):
    return {"checklist": [f"Badge printed for the {runtime.context.office} office"]}

def meet_manager(state: OnboardingState):
    return {"checklist": [f"Manager met the new {state['role']}"]}


builder = StateGraph(OnboardingState, context_schema=OnboardingContext)
builder.add_node("hr_desk", hr_desk)
builder.add_node("it_desk", it_desk)
builder.add_node("facilities_desk", facilities_desk)
builder.add_node("meet_manager", meet_manager)

builder.add_edge(START, "hr_desk")
builder.add_edge("hr_desk", "it_desk")
builder.add_edge("hr_desk", "facilities_desk")
builder.add_edge(["it_desk", "facilities_desk"], "meet_manager")  # waits for both
builder.add_edge("meet_manager", END)

graph = builder.compile()


if __name__ == "__main__":
    inputs = {"name": "Sara", "role": "engineer", "checklist": []}
    ctx = OnboardingContext(company="Deloitte", office="Riyadh")

    # "updates" mode shows what each node wrote, step by step
    for step in graph.stream(inputs, context=ctx, stream_mode="updates"):
        print(step)

    final = graph.invoke(inputs, context=ctx)
    print(graph.get_graph().draw_mermaid())
    print("\nFinal checklist:")
    for item in final["checklist"]:
        print("-", item)