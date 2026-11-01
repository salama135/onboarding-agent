from pathlib import Path

import pytest
import streamlit as st
from langchain.messages import AIMessage
from streamlit.testing.v1 import AppTest

import onboarding.agents
from onboarding.agents import build_it_specialist, build_supervisor
from tests.conftest import call, scripted

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture(autouse=True)
def fresh_app(tmp_path, monkeypatch):
    monkeypatch.setenv("ONBOARDING_DATA_DIR", str(tmp_path))
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def start(at: AppTest, user: str = "sara") -> AppTest:
    at.sidebar.text_input[0].set_value(user)
    return at.sidebar.button[0].click().run()


def test_app_starts_and_creates_session():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert "Start session" in at.info[0].value

    start(at)
    assert not at.exception
    assert at.session_state.session.context.user_id == "sara"
    assert any("hr_paperwork" in m.value for m in at.sidebar.markdown)


def test_invalid_user_id_shows_error():
    at = AppTest.from_file(APP, default_timeout=30).run()
    start(at, "../etc")
    assert at.sidebar.error[0].value.startswith("user_id must be")


def test_chat_approval_and_forget_flow(monkeypatch, settings):
    def fake_agent(_settings, checkpointer, store):
        specialist = build_it_specialist(
            scripted(call("order_laptop", model="MacBook Pro 16"), AIMessage("IT: ordered.")), settings, store
        )
        supervisor_model = scripted(
            call("ask_it_specialist", call_id="s1", request="order laptop"),
            AIMessage("Your laptop is on the way!"),
            call("remember", key="diet", value="vegetarian"),
            AIMessage("Noted!"),
        )
        return build_supervisor(supervisor_model, specialist, settings, checkpointer=checkpointer, store=store)

    monkeypatch.setattr(onboarding.agents, "build_onboarding_agent", fake_agent)
    at = start(AppTest.from_file(APP, default_timeout=30).run())

    at.chat_input[0].set_value("I need a laptop").run()
    assert not at.exception
    assert "Manager approval needed" in at.warning[0].value
    assert at.chat_input[0].disabled

    at.text_input(key="arg_0_model").set_value("MacBook Air")
    at.segmented_control(key="choice_0").set_value("edit")
    at.button(key="submit_decision").click().run()
    assert not at.exception
    assert at.session_state.session.context.it_systems.orders["sara-laptop"]["model"] == "MacBook Air"
    assert any("Your laptop is on the way!" in m.value for m in at.markdown)

    at.chat_input[0].set_value("I'm vegetarian").run()
    assert at.sidebar.dataframe[0].value["value"].tolist() == ["vegetarian"]

    at.button(key="forget_me").click().run()
    assert not at.exception
    assert "Nothing remembered yet." in at.sidebar.caption[0].value
