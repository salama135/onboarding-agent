import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    model: str = "llama3.2"
    fallback_model: str | None = None
    num_ctx: int = 8192
    data_dir: Path = Path.home() / "onboarding-data"
    context_token_budget: int = 3000
    summary_trigger_tokens: int = 2500
    summary_keep_messages: int = 10
    retry_initial_delay: float = 0.5


def load_settings(env: Mapping[str, str] = os.environ) -> Settings:
    defaults = Settings()
    return Settings(
        model=env.get("ONBOARDING_MODEL", defaults.model),
        fallback_model=env.get("ONBOARDING_FALLBACK_MODEL") or None,
        num_ctx=int(env.get("ONBOARDING_NUM_CTX", defaults.num_ctx)),
        data_dir=Path(env.get("ONBOARDING_DATA_DIR", str(defaults.data_dir))).expanduser(),
        context_token_budget=int(env.get("ONBOARDING_CONTEXT_TOKENS", defaults.context_token_budget)),
        summary_trigger_tokens=int(env.get("ONBOARDING_SUMMARY_TRIGGER_TOKENS", defaults.summary_trigger_tokens)),
    )
