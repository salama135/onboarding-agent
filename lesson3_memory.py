from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, wrap_model_call
from langchain.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

STRATEGY = "window"          # try: "none", "window", "summary"
KEEP_LAST_MESSAGES = 6

model = ChatOllama(model="llama3.2", num_ctx=8192)


def recent_turns(messages, keep):
    start = max(len(messages) - keep, 0)
    while start > 0 and not isinstance(messages[start], HumanMessage):  # never start mid tool-call
        start -= 1
    return messages[start:]


@wrap_model_call
def sliding_window(request, handler):
    return handler(request.override(messages=recent_turns(request.messages, KEEP_LAST_MESSAGES)))


@wrap_model_call
def log_context_size(request, handler):
    tokens = count_tokens_approximately(request.messages)
    print(f"  [model sees] {len(request.messages)} messages, ~{tokens} tokens")
    return handler(request)


summarizer = SummarizationMiddleware(
    model=ChatOllama(model="llama3.2", num_ctx=8192),
    trigger=("tokens", 600),          # deliberately tiny so you see it fire
    keep=("messages", KEEP_LAST_MESSAGES),
)

middleware = {
    "none": [log_context_size],
    "window": [sliding_window, log_context_size],
    "summary": [summarizer, log_context_size],
}[STRATEGY]

agent = create_agent(
    model=model,
    tools=[],
    system_prompt="You are a friendly onboarding assistant. Keep answers to 1-2 sentences.",
    middleware=middleware,
    checkpointer=InMemorySaver(),
)

config = {"configurable": {"thread_id": "sara-day-1"}}

if __name__ == "__main__":
    print(f"Strategy: {STRATEGY}. Type 'quit' to stop.")
    while True:
        text = input("\nYou: ").strip()
        if text.lower() in {"quit", "exit"}:
            break
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
        print("Agent:", result["messages"][-1].content)
        print(f"  [state saved] {len(result['messages'])} messages")