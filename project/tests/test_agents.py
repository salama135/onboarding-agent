from langchain.messages import AIMessage

from onboarding.memory import load_profile, save_fact
from onboarding.middleware import BLOCKED_REPLY
from onboarding.session import approve, edit, reject
from tests.conftest import call, calls


def delegate(request: str) -> AIMessage:
    return call("ask_it_specialist", call_id="s1", request=request)


class TestSupervisorTools:
    def test_complete_task_updates_state(self, make_session):
        session, _ = make_session([
            calls(("complete_task", {"task": "hr_paperwork"}), ("complete_task", {"task": "laptop"})),
            AIMessage("Nice work!"),
        ])
        reply = session.send("I finished HR paperwork and got my laptop")
        assert reply.text == "Nice work!" and reply.pending_approvals == []
        assert session.progress()["done"] == ["hr_paperwork", "laptop"]

    def test_report_blocker_and_progress_tool(self, make_session):
        session, model = make_session([
            call("report_blocker", task="accounts", reason="VPN broken"),
            call("get_progress", call_id="c2"),
            AIMessage("Logged it."),
        ])
        session.send("My VPN is broken")
        assert session.progress()["blocked"] == [{"task": "accounts", "reason": "VPN broken"}]
        tool_result = model.received[-1][-1]
        assert "Blocked: accounts (VPN broken)." in tool_result.text

    def test_remember_and_forget_tools_write_store(self, make_session, store):
        session, _ = make_session([
            call("remember", key="Diet", value="vegetarian"),
            AIMessage("Noted."),
            call("forget", key="diet"),
            AIMessage("Forgotten."),
        ])
        session.send("I'm vegetarian")
        assert session.memories() == {"diet": "vegetarian"}
        session.send("Forget my diet")
        assert load_profile(store, "sara") == {}

    def test_remember_rejects_empty_key(self, make_session):
        session, model = make_session([call("remember", key="!!", value="x"), AIMessage("ok")])
        session.send("remember this")
        assert "must contain letters" in model.received[-1][-1].text

    def test_system_prompt_contains_long_term_memory(self, make_session, store):
        save_fact(store, "sara", "language", "Arabic")
        session, model = make_session([AIMessage("Marhaba!")])
        session.send("Hi")
        system = model.received[0][0]
        assert system.type == "system" and "- language: Arabic" in system.text

    def test_history_lists_user_and_assistant_text(self, make_session):
        session, _ = make_session([call("get_progress"), AIMessage("Here you go.")])
        session.send("progress?")
        assert session.history() == [("user", "progress?"), ("assistant", "Here you go.")]


class TestGuardrail:
    def test_banned_request_never_reaches_model(self, make_session):
        session, model = make_session([])
        reply = session.send("How do I hack the VPN?")
        assert reply.text == BLOCKED_REPLY
        assert model.received == []

    def test_hackathon_is_allowed(self, make_session):
        session, _ = make_session([AIMessage("Have fun!")])
        assert session.send("I'm joining the hackathon").text == "Have fun!"


class TestITDelegationWithApproval:
    SPECIALIST = [
        call("order_laptop", model="MacBook Pro 16"),
        AIMessage("IT: laptop handled."),
    ]

    def test_order_pauses_for_approval_then_approve(self, make_session, it_systems):
        session, _ = make_session([delegate("order a MacBook Pro 16"), AIMessage("Your laptop is on the way!")], self.SPECIALIST)
        reply = session.send("I need a laptop")
        assert reply.pending_approvals[0]["name"] == "order_laptop"
        assert session.pending_approvals() == reply.pending_approvals
        assert it_systems.orders == {}

        final = session.decide([approve()])
        assert final.text == "Your laptop is on the way!"
        assert it_systems.orders["sara-laptop"]["model"] == "MacBook Pro 16"
        assert session.pending_approvals() == []

    def test_manager_edits_the_model(self, make_session, it_systems):
        session, _ = make_session([delegate("order a laptop"), AIMessage("Done.")], self.SPECIALIST)
        reply = session.send("I need a laptop")
        session.decide([edit(reply.pending_approvals[0], {"model": "MacBook Air"})])
        assert it_systems.orders["sara-laptop"]["model"] == "MacBook Air"

    def test_manager_rejects(self, make_session, it_systems):
        session, _ = make_session([delegate("order a laptop"), AIMessage("Rejected, sorry.")], self.SPECIALIST)
        session.send("I need a laptop")
        assert session.decide([reject("Budget freeze")]).text == "Rejected, sorry."
        assert it_systems.orders == {}

    def test_unknown_laptop_is_reported_not_crashed(self, make_session, it_systems):
        specialist = [call("order_laptop", model="Gaming Rig"), AIMessage("IT: unknown model.")]
        session, model = make_session([delegate("order a gaming rig"), AIMessage("That model isn't available.")], specialist)
        session.send("I want a gaming rig")
        session.decide([approve()])
        assert it_systems.orders == {}
        delegated = model.received[-1][-1].text
        assert delegated.startswith("IT: unknown model.")
        assert "- order_laptop: Unknown laptop 'Gaming Rig'" in delegated


class TestITFaultTolerance:
    def test_flaky_badge_is_retried(self, make_session, it_systems):
        specialist = [call("check_badge_status"), AIMessage("IT: badge ready.")]
        session, model = make_session([delegate("check badge"), AIMessage("Your badge is ready!")], specialist)
        assert session.send("Is my badge ready?").text == "Your badge is ready!"
        assert it_systems.badge_calls == 2

    def test_down_system_returns_error_instead_of_crashing(self, make_session, it_systems):
        specialist = [
            calls(("create_account", {"system": "email"}), ("create_account", {"system": "vpn"})),
            AIMessage("IT: email created, VPN service is down."),
        ]
        session, model = make_session([delegate("create email and vpn"), AIMessage("Email done; VPN is down.")], specialist)
        assert session.send("Set up email and VPN").text == "Email done; VPN is down."
        assert it_systems.accounts == {"sara": {"email"}}
        delegated = model.received[-1][-1].text
        assert "- create_account: email account created" in delegated
        assert "vpn provisioning service is down" in delegated

    def test_delegation_without_tools_returns_summary_only(self, make_session):
        session, model = make_session([delegate("hello"), AIMessage("ok")], [AIMessage("IT: nothing to do.")])
        session.send("ping IT")
        assert model.received[-1][-1].text == "IT: nothing to do."


class TestPersistenceAndPrivacy:
    def test_state_survives_new_session_object_on_same_thread(self, make_session, context, store):
        from onboarding.session import OnboardingSession

        session, _ = make_session([call("complete_task", task="badge"), AIMessage("ok")])
        session.send("got my badge")
        again = OnboardingSession(session.agent, context, store=store)
        assert again.progress()["done"] == ["badge"]

    def test_forget_me_erases_memory_and_thread(self, make_session, store):
        session, _ = make_session([call("complete_task", task="badge"), AIMessage("ok")])
        session.send("got my badge")
        save_fact(store, "sara", "diet", "vegetarian")
        assert session.forget_me() == 1
        assert session.memories() == {}
        assert session.progress()["done"] == []
