"""PostgreSQL connection pool; loads .env locally via python-dotenv."""

from __future__ import annotations

import os
from typing import Any

# Load .env only when present (e.g. local dev); does not override existing env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import psycopg2
from psycopg2.pool import SimpleConnectionPool

_DATABASE_URL = os.getenv("DATABASE_URL")
_pool: SimpleConnectionPool | None = None
if _DATABASE_URL:
    _pool = SimpleConnectionPool(minconn=1, maxconn=10, dsn=_DATABASE_URL)


def get_conn() -> Any:
    """Get a connection from the pool. Raises if DATABASE_URL is not set or pool unavailable."""
    if _pool is None:
        raise RuntimeError("DATABASE_URL not set")
    return _pool.getconn()


def put_conn(conn: Any) -> None:
    """Return a connection to the pool."""
    if conn is not None and _pool is not None:
        _pool.putconn(conn)
