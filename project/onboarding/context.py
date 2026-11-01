import re
from dataclasses import dataclass, field

from onboarding.mock_systems import MockITSystems

_USER_ID_PATTERN = re.compile(r"^[a-z0-9_-]{1,40}$")


def validate_user_id(user_id: str) -> str:
    """User ids become store namespaces and thread ids, so only allow a safe character set."""
    normalized = user_id.strip().lower()
    if not _USER_ID_PATTERN.fullmatch(normalized):
        raise ValueError("user_id must be 1-40 characters of a-z, 0-9, '_' or '-'")
    return normalized


@dataclass(frozen=True)
class OnboardingContext:
    user_id: str
    employee_name: str
    role: str
    company: str = "Deloitte"
    it_systems: MockITSystems = field(default_factory=MockITSystems, compare=False)
