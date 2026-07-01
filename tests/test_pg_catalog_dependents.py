"""Regression tests for ``sqlalchemy_utils.alembic.pg_catalog.get_dependent_views``.

These tests exercise real PostgreSQL catalog queries and require a live PG
instance. They are marked ``@pytest.mark.infrastructure`` and skip gracefully
when PG is unavailable.

BUG-1 regression: ``get_dependent_views`` previously referenced a nonexistent
``pg_depend.refobjname`` column, crashing with ``UndefinedColumn`` at runtime.
Additionally, the dict key was the *referenced* view's name rather than the
*dependent* view's name. These tests lock both behaviours.
"""
from __future__ import annotations

import os
import socket

import pytest
import sqlalchemy as sa
from sqlalchemy import create_engine

from sqlalchemy_utils.alembic.pg_catalog import (
    get_database_views,
    get_dependent_views,
)


PG_HOST = os.environ.get("SQLALCHEMY_UTILS_TEST_POSTGRESQL_HOST", "localhost")
PG_PORT = int(os.environ.get("SQLALCHEMY_UTILS_TEST_POSTGRESQL_PORT", "55432"))
PG_USER = os.environ.get("SQLALCHEMY_UTILS_TEST_POSTGRESQL_USER", "postgres")
PG_PASSWORD = os.environ.get("SQLALCHEMY_UTILS_TEST_POSTGRESQL_PASSWORD", "")
PG_DB = os.environ.get("SQLALCHEMY_UTILS_TEST_DB", "sqlalchemy_utils_test")

# psycopg2 connection string; the task spec mandates this driver.
DSN = f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}"


def _pg_available() -> bool:
    """Return True if a TCP connection to the PG port succeeds."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1.0)
    try:
        sock.connect((PG_HOST, PG_PORT))
    except OSError:
        return False
    finally:
        sock.close()
    return True


PG_AVAILABLE = _pg_available()


pytestmark = pytest.mark.infrastructure


@pytest.fixture
def connection():
    """Yield a SQLAlchemy Connection against the live PG instance."""
    if not PG_AVAILABLE:
        pytest.skip(f"PostgreSQL not reachable at {PG_HOST}:{PG_PORT}")
    engine = create_engine(DSN, future=True)
    conn = engine.connect()
    try:
        yield conn
    finally:
        conn.close()
        engine.dispose()


def _setup_schema(connection):
    """Create base table + base view + dependent view. Idempotent."""
    connection.execute(
        sa.text(
            "CREATE TABLE IF NOT EXISTS _bug1_base (id SERIAL PRIMARY KEY)"
        )
    )
    connection.execute(
        sa.text(
            "CREATE OR REPLACE VIEW _bug1_base_view AS "
            "SELECT id FROM _bug1_base"
        )
    )
    connection.execute(
        sa.text(
            "CREATE OR REPLACE VIEW _bug1_dep_view AS "
            "SELECT * FROM _bug1_base_view"
        )
    )
    connection.commit()


def _teardown_schema(connection):
    """Drop the dependent view, base view, and base table."""
    try:
        connection.execute(sa.text("DROP VIEW IF EXISTS _bug1_dep_view CASCADE"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    try:
        connection.execute(sa.text("DROP VIEW IF EXISTS _bug1_base_view CASCADE"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    try:
        connection.execute(sa.text("DROP TABLE IF EXISTS _bug1_base CASCADE"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    connection.commit()


def test_get_dependent_views_returns_correct_name(connection):
    """``get_dependent_views`` returns ``{dependent_name: definition}``.

    Regression for BUG-1: the broken query referenced
    ``pg_depend.refobjname`` (nonexistent column) and used the *referenced*
    view's name as the dict key. After the fix, the key MUST be the
    *dependent* view's name (``_bug1_dep_view``), not the referenced view's
    name (``_bug1_base_view``).
    """
    _setup_schema(connection)
    try:
        # Sanity check: base view is registered in pg_views.
        db_views = get_database_views(connection)
        assert "_bug1_base_view" in db_views, (
            "base view should exist in pg_views; got keys: "
            f"{sorted(db_views.keys())}"
        )

        dependents = get_dependent_views(connection, "_bug1_base_view")

        # The dependent view's name must be the dict key.
        assert "_bug1_dep_view" in dependents, (
            "expected dependent key '_bug1_dep_view' in result, got: "
            f"{sorted(dependents.keys())}"
        )

        # The referenced view's name must NOT be the dict key.
        assert "_bug1_base_view" not in dependents, (
            "referenced view name leaked into dependent dict keys: "
            f"{sorted(dependents.keys())}"
        )

        # The value must be the dependent view's definition SQL.
        dep_definition = dependents["_bug1_dep_view"]
        assert isinstance(dep_definition, str)
        assert dep_definition.strip(), "dependent definition must be non-empty"
        # The definition should reference the base view.
        assert "_bug1_base_view" in dep_definition, (
            "dependent view definition should reference the base view; got: "
            f"{dep_definition!r}"
        )
    finally:
        _teardown_schema(connection)
