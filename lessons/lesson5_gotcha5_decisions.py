from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.messages import AIMessage
from langchain.tools import tool
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command


class FakeToolModel(GenericFakeChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


@tool
def order_laptop(model: str, employee: str) -> str:
    """Order a laptop for a new employee."""
    print(f"  [MOCK PURCHASE] {model} for {employee}")
    return f"Ordered {model} for {employee}"


def make_agent():
    model = FakeToolModel(messages=iter([
        AIMessage(content="", tool_calls=[
            {"name": "order_laptop", "args": {"model": "MacBook Pro", "employee": "Sara"}, "id": "call_sara"},
            {"name": "order_laptop", "args": {"model": "ThinkPad", "employee": "Omar"}, "id": "call_omar"},
        ]),
        AIMessage(content="Finished processing laptop requests."),
    ]))
    return create_agent(
        model=model,
        tools=[order_laptop],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={"order_laptop": True})],
        checkpointer=InMemorySaver(),
    )


def start(agent, config):
    result = agent.invoke({"messages": [{"role": "user", "content": "Laptops for Sara and Omar"}]}, config)
    return result["__interrupt__"][0].value["action_requests"]


policy = {"Omar": "approve", "Sara": "reject"}  # intent: buy Omar's, refuse Sara's
config = {"configurable": {"thread_id": "t"}}

print("=== BROKEN A: too few decisions ===")
agent = make_agent()
start(agent, config)
try:
    agent.invoke(Command(resume={"decisions": [{"type": "approve"}]}), config)
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

print("\n=== BROKEN B: right count, wrong order ===")
agent = make_agent()
start(agent, config)
decisions = [{"type": t, "message": "no"} if t == "reject" else {"type": t} for t in policy.values()]
agent.invoke(Command(resume={"decisions": decisions}), config)

print("\n=== FIXED: one decision per action_request, in its order ===")
agent = make_agent()
actions = start(agent, config)
decisions = []
for action in actions:
    choice = policy[action["args"]["employee"]]
    decisions.append({"type": "reject", "message": "Not in budget"} if choice == "reject" else {"type": "approve"})
agent.invoke(Command(resume={"decisions": decisions}), config)