from pathlib import Path

import pytest

from onboarding.config import Settings, load_settings
from onboarding.context import validate_user_id
from onboarding.mock_systems import MockITSystems
from onboarding.state import format_progress, open_blockers, remaining_tasks


class TestSettings:
    def test_defaults_when_env_is_empty(self):
        assert load_settings({}) == Settings()

    def test_env_overrides(self):
        s = load_settings({
            "ONBOARDING_MODEL": "qwen2.5:7b",
            "ONBOARDING_FALLBACK_MODEL": "llama3.2",
            "ONBOARDING_NUM_CTX": "4096",
            "ONBOARDING_DATA_DIR": "/tmp/x",
            "ONBOARDING_CONTEXT_TOKENS": "1000",
            "ONBOARDING_SUMMARY_TRIGGER_TOKENS": "800",
        })
        assert (s.model, s.fallback_model, s.num_ctx) == ("qwen2.5:7b", "llama3.2", 4096)
        assert (s.data_dir, s.context_token_budget, s.summary_trigger_tokens) == (Path("/tmp/x"), 1000, 800)

    def test_blank_fallback_means_none(self):
        assert load_settings({"ONBOARDING_FALLBACK_MODEL": ""}).fallback_model is None


class TestValidateUserId:
    @pytest.mark.parametrize("raw, expected", [("Sara", "sara"), ("  omar_1 ", "omar_1"), ("a-b", "a-b")])
    def test_accepts_and_normalizes(self, raw, expected):
        assert validate_user_id(raw) == expected

    @pytest.mark.parametrize("raw", ["", "sara/../omar", "a b", "x" * 41, "s@ra"])
    def test_rejects_unsafe_ids(self, raw):
        with pytest.raises(ValueError):
            validate_user_id(raw)


class TestMockITSystems:
    def test_order_is_idempotent(self):
        it = MockITSystems()
        first = it.place_laptop_order("sara", "MacBook Air")
        second = it.place_laptop_order("sara", "MacBook Pro 16")
        assert first == second == {"order_id": "sara-laptop", "model": "MacBook Air", "price": 1200}
        assert len(it.orders) == 1

    def test_unknown_laptop_rejected(self):
        with pytest.raises(ValueError, match="Unknown laptop"):
            MockITSystems().place_laptop_order("sara", "Gaming Rig")

    def test_badge_fails_then_succeeds(self):
        it = MockITSystems(badge_failures_before_success=1)
        with pytest.raises(ConnectionError):
            it.badge_status("sara")
        assert "ready" in it.badge_status("sara")

    def test_accounts_created_and_down_system_fails(self):
        it = MockITSystems()
        assert it.create_account("sara", "email") == "email account created"
        assert it.accounts == {"sara": {"email"}}
        with pytest.raises(ConnectionError, match="vpn"):
            it.create_account("sara", "vpn")

    def test_unknown_account_system_rejected(self):
        with pytest.raises(ValueError):
            MockITSystems().create_account("sara", "fax")


class TestProgress:
    def test_remaining_excludes_done(self):
        assert "laptop" not in remaining_tasks({"completed_tasks": ["laptop"]})
        assert len(remaining_tasks({})) == 6

    def test_blockers_resolved_by_completion_and_latest_reason_wins(self):
        state = {
            "completed_tasks": ["badge"],
            "blocked_tasks": [
                {"task": "accounts", "reason": "old"},
                {"task": "accounts", "reason": "vpn down"},
                {"task": "badge", "reason": "printer"},
            ],
        }
        assert open_blockers(state) == [{"task": "accounts", "reason": "vpn down"}]

    def test_format_progress(self):
        text = format_progress({"completed_tasks": ["laptop"], "blocked_tasks": [{"task": "accounts", "reason": "vpn"}]})
        assert "Done: laptop." in text
        assert "Blocked: accounts (vpn)." in text

    def test_format_progress_empty(self):
        assert "nothing yet" in format_progress({}) and "Blocked: none." in format_progress({})
