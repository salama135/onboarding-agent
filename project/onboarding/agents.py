from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelCallLimitMiddleware,
    ModelFallbackMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
    ToolRetryMiddleware,
)
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.store.base import BaseStore

from onboarding.config import Settings
from onboarding.context import OnboardingContext
from onboarding.middleware import content_filter, onboarding_prompt, token_window
from onboarding.state import OnboardingState
from onboarding.tools.it_tools import IT_TOOLS
from onboarding.tools.onboarding_tools import SUPERVISOR_TOOLS

IT_SPECIALIST_PROMPT = (
    "You are the IT specialist. You only handle laptops, office badges, and accounts "
    "(email, slack, jira, vpn). Use your tools, call create_account once per system, "
    "and finish with a one-sentence summary of what succeeded and what failed. "
    "If a manager rejects an action, do not retry it."
)


def make_chat_model(name: str, settings: Settings) -> ChatOllama:
    return ChatOllama(model=name, num_ctx=settings.num_ctx, temperature=0)


def build_it_specialist(model: BaseChatModel, settings: Settings, store: BaseStore | None = None):
    """No checkpointer: as a sub-agent it inherits the supervisor's, which lets approvals pause and resume."""
    return create_agent(
        model=model,
        tools=IT_TOOLS,
        system_prompt=IT_SPECIALIST_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(
                interrupt_on={"order_laptop": {"allowed_decisions": ["approve", "edit", "reject"]}},
                description_prefix="Manager approval needed",
            ),
            ToolRetryMiddleware(
                max_retries=2,
                tools=["check_badge_status", "create_account"],
                retry_on=(ConnectionError,),
                initial_delay=settings.retry_initial_delay,
            ),
            ToolCallLimitMiddleware(tool_name="order_laptop", run_limit=1),
            ModelCallLimitMiddleware(run_limit=8),
        ],
        context_schema=OnboardingContext,
        store=store,
        name="it_specialist",
    )


def make_ask_it_specialist(specialist):
    @tool
    def ask_it_specialist(request: str, runtime: ToolRuntime[OnboardingContext]) -> str:
        """Delegate IT work to the IT specialist: laptops (orders need manager approval),
        office badge status, and email/slack/jira/vpn accounts. Describe the full request."""
        result = specialist.invoke({"messages": [{"role": "user", "content": request}]}, context=runtime.context)
        tool_results = [f"- {m.name}: {m.text}" for m in result["messages"] if isinstance(m, ToolMessage)]
        summary = result["messages"][-1].text
        if not tool_results:
            return summary
        # small models paraphrase badly, so pass the raw system results as the source of truth
        return f"{summary}\n\nIT system results (trust these over the summary):\n" + "\n".join(tool_results)

    return ask_it_specialist


def build_supervisor(
    model: BaseChatModel,
    specialist,
    settings: Settings,
    *,
    checkpointer: BaseCheckpointSaver | None = None,
    store: BaseStore | None = None,
    fallback_model: BaseChatModel | None = None,
):
    middleware = [
        content_filter,
        onboarding_prompt,
        SummarizationMiddleware(
            model=model,
            trigger=("tokens", settings.summary_trigger_tokens),
            keep=("messages", settings.summary_keep_messages),
        ),
        token_window(settings.context_token_budget),
        ModelCallLimitMiddleware(run_limit=10),
    ]
    if fallback_model is not None:
        middleware.append(ModelFallbackMiddleware(fallback_model))

    return create_agent(
        model=model,
        tools=[*SUPERVISOR_TOOLS, make_ask_it_specialist(specialist)],
        middleware=middleware,
        state_schema=OnboardingState,
        context_schema=OnboardingContext,
        checkpointer=checkpointer,
        store=store,
        name="onboarding_supervisor",
    )


def build_onboarding_agent(settings: Settings, checkpointer: BaseCheckpointSaver, store: BaseStore):
    model = make_chat_model(settings.model, settings)
    fallback = make_chat_model(settings.fallback_model, settings) if settings.fallback_model else None
    specialist = build_it_specialist(model, settings, store)
    return build_supervisor(model, specialist, settings, checkpointer=checkpointer, store=store, fallback_model=fallback)
