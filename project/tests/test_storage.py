from onboarding.memory import load_profile, save_fact
from onboarding.storage import open_storage


def test_storage_persists_memory_across_reopen(tmp_path):
    with open_storage(tmp_path) as (_, store):
        save_fact(store, "sara", "diet", "vegetarian")
        assert load_profile(store, "sara") == {"diet": "vegetarian"}
    with open_storage(tmp_path) as (_, store):
        assert load_profile(store, "sara") == {"diet": "vegetarian"}
    assert (tmp_path / "checkpoints.db").exists() and (tmp_path / "memory.db").exists()
