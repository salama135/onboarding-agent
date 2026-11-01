from dataclasses import dataclass, field
from typing import Any

from langchain.messages import AIMessage, HumanMessage
from langgraph.types import Command

from onboarding.context import OnboardingContext
from onboarding.memory import forget_user, load_profile
from onboarding.state import ONBOARDING_TASKS, open_blockers


def approve() -> dict:
    return {"type": "approve"}


def reject(message: str) -> dict:
    return {"type": "reject", "message": message}


def edit(action: dict, new_args: dict) -> dict:
    return {"type": "edit", "edited_action": {"name": action["name"], "args": {**action["args"], **new_args}}}


def build_decision(action: dict, choice: str, new_args: dict | None = None, reason: str = "") -> dict:
    if choice == "approve":
        return approve()
    if choice == "edit":
        return edit(action, new_args or {})
    if choice == "reject":
        return reject(reason.strip() or "Rejected by manager")
    raise ValueError(f"Unknown decision '{choice}'")


@dataclass
class Reply:
    text: str
    pending_approvals: list[dict] = field(default_factory=list)


class OnboardingSession:
    """One employee's conversation. Shared by the CLI and the Streamlit UI."""

    def __init__(self, agent, context: OnboardingContext, thread_id: str | None = None, store=None):
        self.agent = agent
        self.context = context
        self.store = store
        self.thread_id = thread_id or f"{context.user_id}-onboarding"

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def send(self, text: str) -> Reply:
        result = self.agent.invoke({"messages": [{"role": "user", "content": text}]}, self.config, context=self.context)
        return self._to_reply(result)

    def decide(self, decisions: list[dict]) -> Reply:
        result = self.agent.invoke(Command(resume={"decisions": decisions}), self.config, context=self.context)
        return self._to_reply(result)

    def pending_approvals(self) -> list[dict]:
        interrupts = self.agent.get_state(self.config).interrupts
        return list(interrupts[0].value["action_requests"]) if interrupts else []

    def progress(self) -> dict[str, Any]:
        values = self.agent.get_state(self.config).values
        done = set(values.get("completed_tasks", []))
        return {
            "done": [t for t in ONBOARDING_TASKS if t in done],
            "remaining": [t for t in ONBOARDING_TASKS if t not in done],
            "blocked": open_blockers(values),
        }

    def history(self) -> list[tuple[str, str]]:
        messages = self.agent.get_state(self.config).values.get("messages", [])
        return [
            ("user" if isinstance(m, HumanMessage) else "assistant", m.text)
            for m in messages
            if isinstance(m, (HumanMessage, AIMessage)) and m.text.strip()
        ]

    def memories(self) -> dict[str, str]:
        return load_profile(self.store, self.context.user_id)

    def forget_me(self) -> int:
        """Erase long-term memory and this employee's conversation checkpoints."""
        removed = forget_user(self.store, self.context.user_id) if self.store is not None else 0
        self.agent.checkpointer.delete_thread(self.thread_id)
        return removed

    @staticmethod
    def _to_reply(result: dict) -> Reply:
        if "__interrupt__" in result:
            return Reply(text="", pending_approvals=list(result["__interrupt__"][0].value["action_requests"]))
        return Reply(text=result["messages"][-1].text)
