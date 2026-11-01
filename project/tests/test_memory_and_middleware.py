import pytest
from langchain.messages import AIMessage, HumanMessage, ToolMessage

from onboarding.context import OnboardingContext
from onboarding.memory import delete_fact, forget_user, load_profile, normalize_key, save_fact
from onboarding.middleware import build_system_prompt, recent_turns_by_tokens


class TestMemory:
    def test_save_overwrites_same_key(self, store):
        save_fact(store, "sara", "Language", "Arabic")
        save_fact(store, "sara", "language ", "English")
        assert load_profile(store, "sara") == {"language": "English"}

    def test_users_are_isolated(self, store):
        save_fact(store, "sara", "diet", "vegetarian")
        assert load_profile(store, "omar") == {}

    def test_value_is_truncated(self, store):
        save_fact(store, "sara", "bio", "x" * 500)
        assert len(load_profile(store, "sara")["bio"]) == 200

    def test_delete_and_forget_user(self, store):
        save_fact(store, "sara", "diet", "vegetarian")
        save_fact(store, "sara", "office", "Riyadh")
        delete_fact(store, "sara", "diet")
        assert load_profile(store, "sara") == {"office": "Riyadh"}
        assert forget_user(store, "sara") == 1
        assert load_profile(store, "sara") == {}

    def test_normalize_key(self):
        assert normalize_key(" Preferred Name! ") == "preferred_name"
        with pytest.raises(ValueError):
            normalize_key("!!!")

    def test_load_profile_without_store(self):
        assert load_profile(None, "sara") == {}


def turn(i: int, words: int = 5):
    return [HumanMessage("q " * words, id=f"h{i}"), AIMessage("a " * words, id=f"a{i}")]


class TestTokenWindow:
    def test_everything_fits(self):
        msgs = turn(1) + turn(2)
        assert recent_turns_by_tokens(msgs, 10_000) == msgs

    def test_trims_old_turns_and_starts_at_human(self):
        msgs = turn(1, 50) + turn(2, 50) + turn(3, 5)
        window = recent_turns_by_tokens(msgs, 40)
        assert isinstance(window[0], HumanMessage)
        assert [m.id for m in window] == ["h3", "a3"]

    def test_never_splits_tool_call_from_result(self):
        msgs = [
            *turn(1, 50),
            HumanMessage("do it", id="h2"),
            AIMessage("", id="a2", tool_calls=[{"name": "t", "args": {}, "id": "x"}]),
            ToolMessage("ok", tool_call_id="x", id="t2"),
        ]
        window = recent_turns_by_tokens(msgs, 15)
        assert [m.id for m in window] == ["h2", "a2", "t2"]

    def test_oversized_latest_turn_is_kept(self):
        msgs = turn(1) + [HumanMessage("big " * 2000, id="h2")]
        assert [m.id for m in recent_turns_by_tokens(msgs, 10)] == ["h2"]

    def test_no_human_messages_returns_all(self):
        msgs = [AIMessage("hi")]
        assert recent_turns_by_tokens(msgs, 1) == msgs


def test_system_prompt_includes_profile_and_progress():
    ctx = OnboardingContext(user_id="sara", employee_name="Sara", role="data engineer")
    prompt = build_system_prompt(ctx, {"diet": "vegetarian"}, {"completed_tasks": ["laptop"]})
    assert "Sara (data engineer)" in prompt
    assert "- diet: vegetarian" in prompt
    assert "Done: laptop." in prompt
