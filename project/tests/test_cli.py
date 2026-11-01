import pytest
from langchain.messages import AIMessage

from cli import ask_manager, parse_args, run_chat
from onboarding.session import Reply, build_decision
from tests.conftest import call


def feeder(*answers):
    it = iter(answers)
    return lambda _prompt: next(it)


ACTION = {"name": "order_laptop", "args": {"model": "MacBook Pro 16"}}


class TestBuildDecision:
    def test_variants(self):
        assert build_decision(ACTION, "approve") == {"type": "approve"}
        assert build_decision(ACTION, "edit", {"model": "MacBook Air"})["edited_action"] == {
            "name": "order_laptop", "args": {"model": "MacBook Air"},
        }
        assert build_decision(ACTION, "reject", reason="  ") == {"type": "reject", "message": "Rejected by manager"}

    def test_unknown_choice(self):
        with pytest.raises(ValueError):
            build_decision(ACTION, "maybe")


class TestAskManager:
    def test_reprompts_until_valid_then_edits(self):
        out = []
        decisions = ask_manager([ACTION], feeder("yes", "edit", "MacBook Air"), out.append)
        assert decisions[0]["edited_action"]["args"] == {"model": "MacBook Air"}

    def test_edit_with_blank_keeps_original(self):
        decisions = ask_manager([ACTION], feeder("edit", ""), lambda *_: None)
        assert decisions[0]["edited_action"]["args"] == {"model": "MacBook Pro 16"}

    def test_reject_with_reason(self):
        decisions = ask_manager([ACTION], feeder("reject", "Budget"), lambda *_: None)
        assert decisions == [{"type": "reject", "message": "Budget"}]


def test_parse_args_defaults():
    args = parse_args(["--user", "sara", "--name", "Sara"])
    assert (args.user, args.role, args.thread) == ("sara", "new joiner", None)


class TestRunChat:
    def test_full_conversation_with_approval_and_commands(self, make_session, it_systems):
        session, _ = make_session(
            [
                call("ask_it_specialist", call_id="s1", request="order laptop"),
                AIMessage("Laptop on its way!"),
                call("remember", key="diet", value="vegetarian"),
                AIMessage("Noted."),
            ],
            [call("order_laptop", model="MacBook Pro 16"), AIMessage("IT: ordered.")],
        )
        out = []
        run_chat(
            session,
            feeder("", "I need a laptop", "approve", "I'm vegetarian", "/progress", "/memories", "/forget-me", "/memories", "quit"),
            out.append,
        )
        text = "\n".join(out)
        assert "Buddy: Laptop on its way!" in text
        assert it_systems.orders["sara-laptop"]["model"] == "MacBook Pro 16"
        assert "diet: vegetarian" in text
        assert "Remaining:" in text
        assert "Erased 1 memories" in text and "(no memories)" in text

    def test_resumes_leftover_approval_on_start(self, make_session, it_systems):
        session, _ = make_session(
            [call("ask_it_specialist", call_id="s1", request="order"), AIMessage("Done.")],
            [call("order_laptop", model="ThinkPad T14"), AIMessage("IT: ok.")],
        )
        assert session.send("laptop please").pending_approvals
        out = []
        run_chat(session, feeder("approve", "quit"), out.append)
        assert "Buddy: Done." in out
        assert it_systems.orders["sara-laptop"]["model"] == "ThinkPad T14"

    def test_agent_errors_do_not_end_the_chat(self):
        class Broken:
            def pending_approvals(self):
                return []

            def send(self, text):
                raise ConnectionError("Ollama is not running")

        out = []
        run_chat(Broken(), feeder("hi", "quit"), out.append)
        assert "  Error: Ollama is not running" in out

    def test_end_of_input_propagates_so_main_can_exit(self):
        class Session:
            def pending_approvals(self):
                return []

        def eof(_prompt):
            raise EOFError

        with pytest.raises(EOFError):
            run_chat(Session(), eof, lambda *_: None)


def test_reply_defaults():
    assert Reply("hi").pending_approvals == []
