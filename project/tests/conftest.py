from collections.abc import Iterable

import pytest
from langchain.messages import AIMessage
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from pydantic import Field

from onboarding.agents import build_it_specialist, build_supervisor
from onboarding.config import Settings
from onboarding.context import OnboardingContext
from onboarding.mock_systems import MockITSystems
from onboarding.session import OnboardingSession


class ScriptedModel(GenericFakeChatModel):
    """Replays scripted AI replies and records every prompt it receives."""

    received: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.received.append(messages)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def scripted(*replies: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(replies))


def call(name: str, call_id: str = "c1", **args) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def calls(*specs: tuple[str, dict]) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": n, "args": a, "id": f"c{i}"} for i, (n, a) in enumerate(specs)])


@pytest.fixture
def settings() -> Settings:
    return Settings(retry_initial_delay=0.0, summary_trigger_tokens=100_000, context_token_budget=100_000)


@pytest.fixture
def it_systems() -> MockITSystems:
    return MockITSystems()


@pytest.fixture
def context(it_systems) -> OnboardingContext:
    return OnboardingContext(user_id="sara", employee_name="Sara", role="data engineer", it_systems=it_systems)


@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def make_session(settings, context, store):
    def _make(supervisor_replies: Iterable[AIMessage], specialist_replies: Iterable[AIMessage] = ()):
        supervisor_model = scripted(*supervisor_replies)
        specialist = build_it_specialist(scripted(*specialist_replies), settings, store)
        agent = build_supervisor(supervisor_model, specialist, settings, checkpointer=InMemorySaver(), store=store)
        return OnboardingSession(agent, context, store=store), supervisor_model

    return _make
