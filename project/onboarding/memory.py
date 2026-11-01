import re

from langgraph.store.base import BaseStore

MAX_VALUE_LENGTH = 200


def profile_namespace(user_id: str) -> tuple[str, ...]:
    return ("users", user_id, "profile")


def normalize_key(key: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")[:40]
    if not cleaned:
        raise ValueError("Memory key must contain letters or numbers")
    return cleaned


def save_fact(store: BaseStore, user_id: str, key: str, value: str) -> str:
    clean_key = normalize_key(key)
    store.put(profile_namespace(user_id), clean_key, {"value": value.strip()[:MAX_VALUE_LENGTH]})
    return clean_key


def delete_fact(store: BaseStore, user_id: str, key: str) -> None:
    store.delete(profile_namespace(user_id), normalize_key(key))


def load_profile(store: BaseStore | None, user_id: str) -> dict[str, str]:
    if store is None:
        return {}
    return {item.key: item.value["value"] for item in store.search(profile_namespace(user_id), limit=100)}


def forget_user(store: BaseStore, user_id: str) -> int:
    items = store.search(("users", user_id), limit=1000)
    for item in items:
        store.delete(item.namespace, item.key)
    return len(items)
