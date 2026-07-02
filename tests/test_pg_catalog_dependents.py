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
from sqlalchemy_utils.alembic.pg_catalog import (
    get_database_materialized_views,
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


def _setup_bug13(connection):
    """Create a base view and a materialized view that depends on it."""
    connection.execute(
        sa.text(
            "CREATE TABLE IF NOT EXISTS _bug13_base (id SERIAL PRIMARY KEY)"
        )
    )
    connection.execute(
        sa.text(
            "CREATE OR REPLACE VIEW _bug13_base_view AS "
            "SELECT id FROM _bug13_base"
        )
    )
    connection.execute(
        sa.text(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS _bug13_mv AS "
            "SELECT id FROM _bug13_base_view"
        )
    )
    connection.commit()


def _teardown_bug13(connection):
    try:
        connection.execute(sa.text("DROP MATERIALIZED VIEW IF EXISTS _bug13_mv"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    try:
        connection.execute(sa.text("DROP VIEW IF EXISTS _bug13_base_view CASCADE"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    try:
        connection.execute(sa.text("DROP TABLE IF EXISTS _bug13_base CASCADE"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    connection.commit()


def test_get_dependent_views_includes_materialized_views(connection):
    """``get_dependent_views`` must include materialized views.

    The query must join both ``pg_views`` and ``pg_matviews``; materialized
    views depending on a regular view must be returned.
    """
    _setup_bug13(connection)
    try:
        mvs = get_database_materialized_views(connection)
        assert "_bug13_mv" in mvs, (
            "MV _bug13_mv should be in pg_matviews; got: "
            f"{sorted(mvs.keys())}"
        )

        dependents = get_dependent_views(connection, "_bug13_base_view")

        assert "_bug13_mv" in dependents, (
            "expected materialized view '_bug13_mv' as dependent of "
            "'_bug13_base_view'; got: "
            f"{sorted(dependents.keys())}"
        )
        assert "_bug13_base_view" not in dependents
    finally:
        _teardown_bug13(connection)


def _setup_bug14(connection):
    """Real dependent in schema A; same-named view with different body in B.

    A name-only join (``c.relname = v.viewname``) cannot tell the two apart
    and would surface B's definition for A's dependent. Schema-qualified
    joins must return A's definition only.
    """
    for schema in ("_bug14_a", "_bug14_b", "_bug_a", "_bug_b"):
        connection.execute(sa.text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
    connection.execute(sa.text("CREATE SCHEMA _bug14_a"))
    connection.execute(sa.text("CREATE SCHEMA _bug14_b"))
    connection.execute(sa.text("CREATE SCHEMA _bug_a"))
    connection.execute(sa.text("CREATE SCHEMA _bug_b"))

    connection.execute(
        sa.text("CREATE TABLE _bug14_a.t (id SERIAL PRIMARY KEY)")
    )
    connection.execute(
        sa.text(
            "CREATE VIEW _bug14_a.base_view AS SELECT id FROM _bug14_a.t"
        )
    )
    connection.execute(
        sa.text(
            "CREATE VIEW _bug_a.dep_view AS "
            "SELECT id FROM _bug14_a.base_view"
        )
    )
    connection.execute(
        sa.text("CREATE TABLE _bug_b.other (id INT)"))
    connection.execute(
        sa.text(
            "CREATE OR REPLACE VIEW _bug_b.dep_view AS "
            "SELECT id FROM _bug_b.other"
        )
    )
    connection.commit()


def _teardown_bug14(connection):
    for schema in ("_bug14_a", "_bug14_b", "_bug_a", "_bug_b"):
        try:
            connection.execute(sa.text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        except sa.exc.SQLAlchemyError:
            connection.rollback()
    connection.commit()


def test_get_dependent_views_no_cross_schema_false_matches(connection):
    """No false positive when same view name exists in another schema.

    ``_bug_a.dep_view`` is a real dependent of ``_bug14_a.base_view``.
    ``_bug_b.dep_view`` has the same name but a different body. A name-only
    join would conflate them; schema-qualified joins must return only
    ``_bug_a.dep_view``'s definition (referencing ``_bug14_a.base_view``).
    """
    _setup_bug14(connection)
    try:
        dependents = get_dependent_views(connection, "base_view", schema="_bug_a")

        assert "dep_view" in dependents, (
            "expected dependent 'dep_view'; got: "
            f"{sorted(dependents.keys())}"
        )
        definition = dependents["dep_view"]
        assert "_bug14_a.base_view" in definition, (
            "dependent definition must reference _bug14_a.base_view; got: "
            f"{definition!r}"
        )
        assert "_bug_b.other" not in definition, (
            "cross-schema definition leaked into dependent; got: "
            f"{definition!r}"
        )

        unscoped = get_dependent_views(connection, "base_view")
        assert "_bug_b.other" not in unscoped.get("dep_view", ""), (
            "cross-schema definition leaked into unscoped dependent; got: "
            f"{unscoped.get('dep_view', '')!r}"
        )
    finally:
        _teardown_bug14(connection)
