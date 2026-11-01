from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


@tool
def order_laptop(model: str, employee: str) -> str:
    """Order a laptop for a new employee."""
    print(f"  [MOCK PURCHASE] {model} for {employee}")
    return f"Order placed: {model} for {employee}. Delivery in 3 days."


@tool
def complete_task(task: str) -> str:
    """Mark an onboarding task as done."""
    return f"Marked '{task}' as done."


agent = create_agent(
    model=ChatOllama(model="qwen2.5:7b", num_ctx=8192),
    tools=[order_laptop, complete_task],
    system_prompt=(
        "You are an onboarding assistant. Use order_laptop for laptop requests and "
        "complete_task for finished tasks. If an action is rejected, explain why to the "
        "user and do not retry it."
    ),
    middleware=[
        HumanInTheLoopMiddleware(interrupt_on={
            "order_laptop": {"allowed_decisions": ["approve", "edit", "reject"]},
            "complete_task": False,
        }),
    ],
    checkpointer=InMemorySaver(),
)


def ask_manager(interrupt_value: dict) -> dict:
    decisions = []
    for action in interrupt_value["action_requests"]:  # one entry per paused tool call
        print(f"\n  Agent wants: {action['name']}({action['args']})")
        choice = input("  approve / edit / reject: ").strip().lower()
        if choice == "edit":
            new_args = {**action["args"], "model": input("  New model: ").strip()}
            decisions.append({"type": "edit", "edited_action": {"name": action["name"], "args": new_args}})
        elif choice == "reject":
            decisions.append({"type": "reject", "message": input("  Reason (sent to agent): ").strip()})
        else:
            decisions.append({"type": "approve"})
    return {"decisions": decisions}


if __name__ == "__main__":
    config = {"configurable": {"thread_id": "sara-hitl"}}
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
        while "__interrupt__" in result:
            result = agent.invoke(Command(resume=ask_manager(result["__interrupt__"][0].value)), config)
        print("Agent:", result["messages"][-1].content)