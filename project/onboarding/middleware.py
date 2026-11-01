import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain.agents import AgentState
from langchain.agents.middleware import before_agent, dynamic_prompt, wrap_model_call
from langchain.messages import AIMessage, AnyMessage, HumanMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.runtime import Runtime

from onboarding.context import OnboardingContext
from onboarding.memory import load_profile
from onboarding.state import format_progress

BANNED_PATTERN = re.compile(r"\b(hack|hacking|exploit|malware|bypass security)\b", re.IGNORECASE)
BLOCKED_REPLY = "I can't help with that. Please rephrase your request."


@before_agent(can_jump_to=["end"])
def content_filter(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    last = state["messages"][-1]  # newest message: with a checkpointer, [0] is the first message ever
    if last.type == "human" and BANNED_PATTERN.search(last.text):
        return {"messages": [AIMessage(BLOCKED_REPLY)], "jump_to": "end"}
    return None


def recent_turns_by_tokens(messages: Sequence[AnyMessage], max_tokens: int) -> list[AnyMessage]:
    """Newest messages that fit the budget, always starting at a human turn so tool calls stay paired."""
    human_indexes = [i for i, m in enumerate(messages) if isinstance(m, HumanMessage)]
    if not human_indexes:
        return list(messages)

    start, total = len(messages), 0
    while start > 0:
        cost = count_tokens_approximately([messages[start - 1]])
        if total + cost > max_tokens:
            break
        total += cost
        start -= 1

    later_humans = [i for i in human_indexes if i >= start]
    start = later_humans[0] if later_humans else human_indexes[-1]
    return list(messages[start:])


def token_window(max_tokens: int):
    @wrap_model_call(name="TokenWindow")
    def _token_window(request, handler):
        return handler(request.override(messages=recent_turns_by_tokens(request.messages, max_tokens)))

    return _token_window


def build_system_prompt(context: OnboardingContext, profile: Mapping[str, str], state: Mapping[str, Any]) -> str:
    facts = "\n".join(f"- {k}: {v}" for k, v in profile.items()) or "- nothing yet"
    return (
        f"You are the onboarding buddy at {context.company}, helping {context.employee_name} "
        f"({context.role}) through their first weeks. Be warm and keep answers to 1-3 sentences.\n\n"
        "Rules:\n"
        "- When the employee says they finished a checklist task, call complete_task once per task.\n"
        "- When they could not finish a task, call report_blocker.\n"
        "- For laptops, badges, or email/slack/jira/vpn accounts, call ask_it_specialist.\n"
        "- When they share a lasting personal fact or preference, call remember.\n"
        "- If a tool reports an error, explain it plainly and suggest a next step.\n\n"
        f"Checklist status: {format_progress(state)}\n\n"
        f"Known about {context.employee_name} (use it, don't recite it):\n{facts}"
    )


@dynamic_prompt
def onboarding_prompt(request) -> str:
    context: OnboardingContext = request.runtime.context
    profile = load_profile(request.runtime.store, context.user_id)
    return build_system_prompt(context, profile, request.state)
