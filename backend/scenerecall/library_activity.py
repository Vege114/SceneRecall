"""Coordinate offline maintenance with serving processes and each other."""
from contextlib import closing, contextmanager
import sqlite3


@contextmanager
def library_activity(root, *, maintenance=False):
    # SQLite supplies process-scoped shared/exclusive locks on Windows and
    # POSIX and releases them on process exit. No library data lives here.
    with closing(sqlite3.connect(root / "runtime" / "library-activity.sqlite", timeout=0)) as db:
        try:
            db.execute("BEGIN EXCLUSIVE" if maintenance else "BEGIN")
            db.execute("SELECT count(*) FROM sqlite_schema").fetchone()
        except sqlite3.OperationalError as exc:
            if exc.sqlite_errorcode not in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
                raise
            raise RuntimeError("资料库正在使用。请先停止服务或等待离线整理完成。") from exc
        yield
