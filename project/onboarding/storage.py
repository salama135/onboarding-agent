import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore


@contextmanager
def open_storage(data_dir: Path) -> Iterator[tuple[SqliteSaver, SqliteStore]]:
    """Checkpoints (per-thread) and long-term memory (per-user) live in separate SQLite files."""
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_conn = sqlite3.connect(data_dir / "checkpoints.db", check_same_thread=False)
    memory_conn = sqlite3.connect(data_dir / "memory.db", check_same_thread=False, isolation_level=None)  # store manages its own transactions
    try:
        store = SqliteStore(memory_conn)
        store.setup()
        yield SqliteSaver(checkpoint_conn), store
    finally:
        checkpoint_conn.close()
        memory_conn.close()
