import argparse
from collections.abc import Callable

from onboarding.agents import build_onboarding_agent
from onboarding.config import load_settings
from onboarding.context import OnboardingContext, validate_user_id
from onboarding.session import OnboardingSession, build_decision
from onboarding.storage import open_storage

HELP = "Commands: /progress  /memories  /forget-me  quit"
DECISIONS = {"approve", "edit", "reject"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Onboarding buddy (terminal)")
    parser.add_argument("--user", required=True, help="employee id, e.g. sara")
    parser.add_argument("--name", required=True, help="display name, e.g. Sara")
    parser.add_argument("--role", default="new joiner")
    parser.add_argument("--thread", default=None, help="conversation id (default: <user>-onboarding)")
    return parser.parse_args(argv)


def ask_manager(actions: list[dict], input_fn: Callable[[str], str] = input, print_fn=print) -> list[dict]:
    decisions = []
    for action in actions:
        print_fn(f"\n  [MANAGER APPROVAL] {action['name']} {action['args']}")
        choice = ""
        while choice not in DECISIONS:
            choice = input_fn("  approve / edit / reject: ").strip().lower()
        new_args = {}
        if choice == "edit":
            for key, value in action["args"].items():
                if new_value := input_fn(f"  {key} [{value}]: ").strip():
                    new_args[key] = new_value
        reason = input_fn("  Reason (sent to the agent): ") if choice == "reject" else ""
        decisions.append(build_decision(action, choice, new_args, reason))
    return decisions


def print_progress(session: OnboardingSession, print_fn=print) -> None:
    progress = session.progress()
    print_fn(f"  Done:      {', '.join(progress['done']) or '-'}")
    print_fn(f"  Remaining: {', '.join(progress['remaining']) or '-'}")
    print_fn(f"  Blocked:   {'; '.join(f'{b['task']} ({b['reason']})' for b in progress['blocked']) or '-'}")


def respond(session: OnboardingSession, reply, input_fn, print_fn) -> None:
    while reply.pending_approvals:
        reply = session.decide(ask_manager(reply.pending_approvals, input_fn, print_fn))
    print_fn(f"Buddy: {reply.text}")


def run_chat(session: OnboardingSession, input_fn: Callable[[str], str] = input, print_fn=print) -> None:
    print_fn(HELP)
    if pending := session.pending_approvals():
        print_fn("  Resuming an approval left over from last time.")
        respond(session, session.decide(ask_manager(pending, input_fn, print_fn)), input_fn, print_fn)

    while True:
        text = input_fn("\nYou: ").strip()
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            return
        if text == "/progress":
            print_progress(session, print_fn)
            continue
        if text == "/memories":
            print_fn("\n".join(f"  {k}: {v}" for k, v in session.memories().items()) or "  (no memories)")
            continue
        if text == "/forget-me":
            print_fn(f"  Erased {session.forget_me()} memories and this conversation.")
            continue
        try:
            respond(session, session.send(text), input_fn, print_fn)
        except EOFError:
            raise
        except Exception as err:  # UI boundary: keep the chat alive (e.g. Ollama not running)
            print_fn(f"  Error: {err}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = load_settings()
    context = OnboardingContext(
        user_id=validate_user_id(args.user),
        employee_name=args.name.strip()[:60],
        role=args.role.strip()[:60],
    )
    with open_storage(settings.data_dir) as (checkpointer, store):
        agent = build_onboarding_agent(settings, checkpointer, store)
        session = OnboardingSession(agent, context, args.thread, store)
        print(f"Onboarding buddy | {context.employee_name} | model {settings.model} | thread {session.thread_id}")
        try:
            run_chat(session)
        except (EOFError, KeyboardInterrupt):
            print("\nBye! Your progress is saved.")


if __name__ == "__main__":
    main()
