from contextlib import ExitStack

import streamlit as st

from onboarding.agents import build_onboarding_agent
from onboarding.config import load_settings
from onboarding.context import OnboardingContext, validate_user_id
from onboarding.session import OnboardingSession, build_decision
from onboarding.storage import open_storage

st.set_page_config(page_title="Onboarding Buddy", page_icon=":material/badge:")


@st.cache_resource
def get_runtime():
    settings = load_settings()
    stack = ExitStack()  # returned below so it isn't garbage-collected, which would close the DB connections
    checkpointer, store = stack.enter_context(open_storage(settings.data_dir))
    return build_onboarding_agent(settings, checkpointer, store), store, settings, stack


def start_session(user: str, name: str, role: str) -> None:
    agent, store, *_ = get_runtime()
    context = OnboardingContext(user_id=validate_user_id(user), employee_name=name.strip()[:60], role=role.strip()[:60])
    st.session_state.session = OnboardingSession(agent, context, store=store)


def render_sidebar(session: OnboardingSession | None) -> None:
    with st.sidebar:
        st.header("Employee")
        with st.form("employee"):
            user = st.text_input("User id", value="sara")
            name = st.text_input("Name", value="Sara")
            role = st.text_input("Role", value="data engineer")
            if st.form_submit_button("Start session", type="primary"):
                try:
                    start_session(user, name, role)
                    st.rerun()
                except ValueError as err:
                    st.error(str(err))

        if session is None:
            return

        progress = session.progress()
        st.subheader("Checklist")
        st.progress(len(progress["done"]) / (len(progress["done"]) + len(progress["remaining"])))
        for task in progress["done"]:
            st.markdown(f":green[:material/check_circle:] {task}")
        for task in progress["remaining"]:
            st.markdown(f":gray[:material/radio_button_unchecked:] {task}")
        for blocker in progress["blocked"]:
            st.markdown(f":red[:material/block:] {blocker['task']}: {blocker['reason']}")

        st.subheader("Long-term memory")
        memories = session.memories()
        if memories:
            st.dataframe({"fact": list(memories), "value": list(memories.values())}, hide_index=True)
        else:
            st.caption("Nothing remembered yet.")

        if st.button("Forget me", icon=":material/delete:", key="forget_me"):
            session.forget_me()
            st.rerun()


def render_approvals(session: OnboardingSession, actions: list[dict]) -> None:
    with st.chat_message("assistant", avatar=":material/approval:"):
        st.warning("Manager approval needed before IT can continue.")
        with st.form("approvals"):
            choices = []
            for i, action in enumerate(actions):
                st.markdown(f"**{action['name']}**")
                choice = st.segmented_control("Decision", ["approve", "edit", "reject"], default="approve", key=f"choice_{i}")
                edited = {k: st.text_input(f"{k} (for edit)", value=str(v), key=f"arg_{i}_{k}") for k, v in action["args"].items()}
                reason = st.text_input("Reason (for reject)", key=f"reason_{i}")
                changed = {k: v for k, v in edited.items() if v != str(action["args"][k])}
                choices.append((action, choice or "approve", changed, reason))
            if st.form_submit_button("Submit decision", type="primary", key="submit_decision"):
                with st.spinner("Resuming the agent..."):
                    session.decide([build_decision(*c) for c in choices])
                st.rerun()


def main() -> None:
    session: OnboardingSession | None = st.session_state.get("session")
    render_sidebar(session)

    st.title(":material/badge: Onboarding Buddy")
    if session is None:
        st.info("Enter the employee details in the sidebar and click **Start session**.")
        return

    _, _, settings, _ = get_runtime()
    st.caption(f"{session.context.employee_name} · model `{settings.model}` · thread `{session.thread_id}`")

    for role, text in session.history():
        with st.chat_message(role):
            st.markdown(text)

    pending = session.pending_approvals()
    if pending:
        render_approvals(session, pending)

    if prompt := st.chat_input("Tell your onboarding buddy what you did or need", disabled=bool(pending)):
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            try:
                session.send(prompt)
            except Exception as err:  # UI boundary: e.g. Ollama not running
                st.error(f"Agent error: {err}")
                return
        st.rerun()


main()
