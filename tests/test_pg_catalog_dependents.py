"""Regression tests for ``sqlalchemy_utils.alembic.pg_catalog.get_dependent_views``.

These tests exercise real PostgreSQL catalog queries and require a live PG
instance. They are marked ``@pytest.mark.infrastructure`` and skip gracefully
when PG is unavailable.

Regression: ``get_dependent_views`` previously referenced a nonexistent
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
            "CREATE TABLE IF NOT EXISTS _dep_test_base (id SERIAL PRIMARY KEY)"
        )
    )
    connection.execute(
        sa.text(
            "CREATE OR REPLACE VIEW _dep_test_base_view AS "
            "SELECT id FROM _dep_test_base"
        )
    )
    connection.execute(
        sa.text(
            "CREATE OR REPLACE VIEW _dep_test_dep_view AS "
            "SELECT * FROM _dep_test_base_view"
        )
    )
    connection.commit()


def _teardown_schema(connection):
    """Drop the dependent view, base view, and base table."""
    try:
        connection.execute(sa.text("DROP VIEW IF EXISTS _dep_test_dep_view CASCADE"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    try:
        connection.execute(sa.text("DROP VIEW IF EXISTS _dep_test_base_view CASCADE"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    try:
        connection.execute(sa.text("DROP TABLE IF EXISTS _dep_test_base CASCADE"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    connection.commit()


def test_get_dependent_views_returns_correct_name(connection):
    """``get_dependent_views`` returns ``{dependent_name: definition}``.

    Regression: the broken query referenced
    ``pg_depend.refobjname`` (nonexistent column) and used the *referenced*
    view's name as the dict key. After the fix, the key MUST be the
    *dependent* view's name (``_dep_test_dep_view``), not the referenced view's
    name (``_dep_test_base_view``).
    """
    _setup_schema(connection)
    try:
        # Sanity check: base view is registered in pg_views.
        db_views = get_database_views(connection)
        assert "_dep_test_base_view" in db_views, (
            "base view should exist in pg_views; got keys: "
            f"{sorted(db_views.keys())}"
        )

        dependents = get_dependent_views(connection, "_dep_test_base_view")

        # The dependent view's name must be the dict key.
        assert "_dep_test_dep_view" in dependents, (
            "expected dependent key '_dep_test_dep_view' in result, got: "
            f"{sorted(dependents.keys())}"
        )

        # The referenced view's name must NOT be the dict key.
        assert "_dep_test_base_view" not in dependents, (
            "referenced view name leaked into dependent dict keys: "
            f"{sorted(dependents.keys())}"
        )

        # The value must be the dependent view's definition SQL.
        dep_definition = dependents["_dep_test_dep_view"]
        assert isinstance(dep_definition, str)
        assert dep_definition.strip(), "dependent definition must be non-empty"
        # The definition should reference the base view.
        assert "_dep_test_base_view" in dep_definition, (
            "dependent view definition should reference the base view; got: "
            f"{dep_definition!r}"
        )
    finally:
        _teardown_schema(connection)


def _setup_mv_dependent(connection):
    """Create a base view and a materialized view that depends on it."""
    connection.execute(
        sa.text(
            "CREATE TABLE IF NOT EXISTS _mv_dep_base (id SERIAL PRIMARY KEY)"
        )
    )
    connection.execute(
        sa.text(
            "CREATE OR REPLACE VIEW _mv_dep_base_view AS "
            "SELECT id FROM _mv_dep_base"
        )
    )
    connection.execute(
        sa.text(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS _mv_dep_mv AS "
            "SELECT id FROM _mv_dep_base_view"
        )
    )
    connection.commit()


def _teardown_mv_dependent(connection):
    try:
        connection.execute(sa.text("DROP MATERIALIZED VIEW IF EXISTS _mv_dep_mv"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    try:
        connection.execute(sa.text("DROP VIEW IF EXISTS _mv_dep_base_view CASCADE"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    try:
        connection.execute(sa.text("DROP TABLE IF EXISTS _mv_dep_base CASCADE"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    connection.commit()


def test_get_dependent_views_includes_materialized_views(connection):
    """``get_dependent_views`` must include materialized views.

    The query must join both ``pg_views`` and ``pg_matviews``; materialized
    views depending on a regular view must be returned.
    """
    _setup_mv_dependent(connection)
    try:
        mvs = get_database_materialized_views(connection)
        assert "_mv_dep_mv" in mvs, (
            "MV _mv_dep_mv should be in pg_matviews; got: "
            f"{sorted(mvs.keys())}"
        )

        dependents = get_dependent_views(connection, "_mv_dep_base_view")

        assert "_mv_dep_mv" in dependents, (
            "expected materialized view '_mv_dep_mv' as dependent of "
            "'_mv_dep_base_view'; got: "
            f"{sorted(dependents.keys())}"
        )
        assert "_mv_dep_base_view" not in dependents
    finally:
        _teardown_mv_dependent(connection)


def _setup_cross_schema(connection):
    """Real dependent in schema A; same-named view with different body in B.

    A name-only join (``c.relname = v.viewname``) cannot tell the two apart
    and would surface B's definition for A's dependent. Schema-qualified
    joins must return A's definition only.
    """
    for schema in ("_cross_schema_a", "_cross_schema_b"):
        connection.execute(sa.text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
    connection.execute(sa.text("CREATE SCHEMA _cross_schema_a"))
    connection.execute(sa.text("CREATE SCHEMA _cross_schema_b"))

    connection.execute(
        sa.text("CREATE TABLE _cross_schema_a.t (id SERIAL PRIMARY KEY)")
    )
    connection.execute(
        sa.text(
            "CREATE VIEW _cross_schema_a.base_view AS SELECT id FROM _cross_schema_a.t"
        )
    )
    connection.execute(
        sa.text(
            "CREATE VIEW _cross_schema_a.dep_view AS "
            "SELECT id FROM _cross_schema_a.base_view"
        )
    )

    connection.execute(
        sa.text("CREATE TABLE _cross_schema_b.other (id INT)"))
    connection.execute(
        sa.text(
            "CREATE VIEW _cross_schema_b.base_view AS SELECT id FROM _cross_schema_b.other"
        )
    )
    connection.execute(
        sa.text(
            "CREATE OR REPLACE VIEW _cross_schema_b.dep_view AS "
            "SELECT id FROM _cross_schema_b.other"
        )
    )
    connection.commit()


def _teardown_cross_schema(connection):
    for schema in ("_cross_schema_a", "_cross_schema_b"):
        try:
            connection.execute(sa.text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        except sa.exc.SQLAlchemyError:
            connection.rollback()
    connection.commit()


def test_get_dependent_views_no_cross_schema_false_matches(connection):
    """No false positive when same view name exists in another schema.

    ``_cross_schema_a.dep_view`` is a real dependent of ``_cross_schema_a.base_view``.
    ``_cross_schema_b.dep_view`` has the same name but a different body. A name-only
    join would conflate them; schema-qualified joins must return only
    ``_cross_schema_a.dep_view``'s definition (referencing ``_cross_schema_a.base_view``).
    The ``schema`` argument filters the referenced view's namespace
    (``_cross_schema_a``), so dependents of same-named views in other schemas
    are excluded.
    """
    _setup_cross_schema(connection)
    try:
        dependents = get_dependent_views(
            connection, "base_view", schema="_cross_schema_a"
        )

        assert "dep_view" in dependents, (
            "expected dependent 'dep_view'; got: "
            f"{sorted(dependents.keys())}"
        )
        definition = dependents["dep_view"]
        assert "_cross_schema_a.base_view" in definition, (
            "dependent definition must reference _cross_schema_a.base_view; got: "
            f"{definition!r}"
        )
        assert "_cross_schema_b.other" not in definition, (
            "cross-schema definition leaked into dependent; got: "
            f"{definition!r}"
        )

        unscoped = get_dependent_views(connection, "base_view")
        assert "_cross_schema_b.other" not in unscoped.get("dep_view", ""), (
            "cross-schema definition leaked into unscoped dependent; got: "
            f"{unscoped.get('dep_view', '')!r}"
        )
    finally:
        _teardown_cross_schema(connection)


def _setup_union_all(connection):
    """Create a regular view and a materialized view as dependents.

    Both depend on ``_union_base_view``. PostgreSQL forbids a regular view
    and a materialized view sharing a name in the same schema, so the two
    dependents have distinct names. The two subqueries in
    ``get_dependent_views`` are disjoint by ``relkind`` (``v`` vs ``m``);
    ``UNION ALL`` preserves all rows from both, while a plain ``UNION``
    would dedupe identical rows. This setup verifies both dependents are
    returned and locks the use of ``UNION ALL`` in the generated SQL.
    """
    connection.execute(
        sa.text(
            "CREATE TABLE IF NOT EXISTS _union_base (id SERIAL PRIMARY KEY)"
        )
    )
    connection.execute(
        sa.text(
            "CREATE OR REPLACE VIEW _union_base_view AS "
            "SELECT id FROM _union_base"
        )
    )
    connection.execute(
        sa.text(
            "CREATE OR REPLACE VIEW _union_dep AS "
            "SELECT id FROM _union_base_view"
        )
    )
    connection.execute(
        sa.text(
            "CREATE MATERIALIZED VIEW IF NOT EXISTS _union_mv AS "
            "SELECT id FROM _union_base_view"
        )
    )
    connection.commit()


def _teardown_union_all(connection):
    try:
        connection.execute(sa.text("DROP MATERIALIZED VIEW IF EXISTS _union_mv"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    try:
        connection.execute(sa.text("DROP VIEW IF EXISTS _union_dep CASCADE"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    try:
        connection.execute(sa.text("DROP VIEW IF EXISTS _union_base_view CASCADE"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    try:
        connection.execute(sa.text("DROP TABLE IF EXISTS _union_base CASCADE"))
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    connection.commit()


@pytest.mark.infrastructure
def test_get_dependent_views_union_all_keeps_both_regular_and_mv(connection):
    """UNION ALL must not dedupe regular + MV dependents.

    The two subqueries in ``get_dependent_views`` are disjoint by
    ``relkind`` (``v`` for regular views via ``pg_views``, ``m`` for
    materialized views via ``pg_matviews``). A plain ``UNION`` would
    dedupe identical rows; ``UNION ALL`` preserves all rows from both
    subqueries. This test verifies:

    1. A regular view dependent (``_union_dep``) and a materialized view
       dependent (``_union_mv``) are BOTH returned for the same referenced
       view.
    2. The generated SQL uses ``UNION ALL`` (not plain ``UNION``), so
       future regressions to ``UNION`` are caught even though the
       disjoint-by-relkind property means the two are functionally
       equivalent for distinct-named dependents.
    """
    _setup_union_all(connection)
    try:
        dependents = get_dependent_views(connection, "_union_base_view")
        assert "_union_dep" in dependents, (
            "expected regular view dependent '_union_dep'; got: "
            f"{sorted(dependents.keys())}"
        )
        assert "_union_mv" in dependents, (
            "expected materialized view dependent '_union_mv'; got: "
            f"{sorted(dependents.keys())}"
        )

        from sqlalchemy_utils.alembic import pg_catalog as pg_catalog_mod

        src = pg_catalog_mod.get_dependent_views.__code__.co_consts
        joined = " ".join(str(c) for c in src)
        assert "UNION ALL" in joined, (
            "get_dependent_views SQL must use UNION ALL (not plain UNION) "
            "to avoid deduping disjoint regular/MV rows; source constants: "
            f"{joined!r}"
        )
    finally:
        _teardown_union_all(connection)


def _setup_schema_filter(connection):
    """Two schemas, each with a same-named referenced view ``base_view``.

    ``_schema_filter_a.base_view`` has a real dependent ``_schema_filter_a.dep_view``.
    ``_schema_filter_b.base_view`` ALSO has a dependent ``_schema_filter_b.dep_view`` (same
    dependent name, different body referencing ``_schema_filter_b.base_view``).
    Without a schema filter on the referenced view's namespace
    (``refn.nspname``), querying dependents of ``base_view`` with
    ``schema='_schema_filter_a'`` would match BOTH ``base_view`` rows (the WHERE
    clause only filtered ``ref.relname = :view_name``) and erroneously
    surface ``_schema_filter_b.dep_view`` as a false positive.
    """
    for schema in ("_schema_filter_a", "_schema_filter_b"):
        connection.execute(sa.text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
    connection.execute(sa.text("CREATE SCHEMA _schema_filter_a"))
    connection.execute(sa.text("CREATE SCHEMA _schema_filter_b"))

    connection.execute(
        sa.text("CREATE TABLE _schema_filter_a.t (id SERIAL PRIMARY KEY)")
    )
    connection.execute(
        sa.text(
            "CREATE VIEW _schema_filter_a.base_view AS SELECT id FROM _schema_filter_a.t"
        )
    )
    connection.execute(
        sa.text(
            "CREATE VIEW _schema_filter_a.dep_view AS "
            "SELECT id FROM _schema_filter_a.base_view"
        )
    )

    # Second schema: same-named base_view WITH a dependent.
    connection.execute(
        sa.text("CREATE TABLE _schema_filter_b.t (id SERIAL PRIMARY KEY)")
    )
    connection.execute(
        sa.text(
            "CREATE VIEW _schema_filter_b.base_view AS SELECT id FROM _schema_filter_b.t"
        )
    )
    connection.execute(
        sa.text(
            "CREATE VIEW _schema_filter_b.dep_view AS "
            "SELECT id FROM _schema_filter_b.base_view"
        )
    )
    connection.commit()


def _teardown_schema_filter(connection):
    for schema in ("_schema_filter_a", "_schema_filter_b"):
        try:
            connection.execute(sa.text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        except sa.exc.SQLAlchemyError:
            connection.rollback()
    connection.commit()


@pytest.mark.infrastructure
def test_get_dependent_views_schema_filter_excludes_other_schema(connection):
    """Schema filter must constrain the referenced view's namespace.

    When ``schema='_schema_filter_a'`` is passed, ``get_dependent_views`` must only
    return dependents of ``_schema_filter_a.base_view``. The referenced view's
    namespace (``refn.nspname``) must be filtered, not just the dependent
    view's schema. Without the ``refn.nspname = :schema`` filter, the
    query matches ``base_view`` in ANY schema and produces false positives
    (e.g. ``_schema_filter_b.dep_view`` which references ``_schema_filter_b.base_view``).
    """
    _setup_schema_filter(connection)
    try:
        dependents = get_dependent_views(
            connection, "base_view", schema="_schema_filter_a"
        )
        assert "dep_view" in dependents, (
            "expected dependent 'dep_view' in _schema_filter_a; got: "
            f"{sorted(dependents.keys())}"
        )
        # Every returned dependent must reference _schema_filter_a.base_view, proving
        # the schema filter constrained the referenced view's namespace.
        for name, definition in dependents.items():
            assert "_schema_filter_a.base_view" in definition, (
                f"dependent {name!r} should reference _schema_filter_a.base_view; "
                f"got definition: {definition!r}"
            )
            assert "_schema_filter_b" not in definition, (
                f"dependent {name!r} leaked cross-schema reference to "
                f"_schema_filter_b; definition: {definition!r}"
            )
    finally:
        _teardown_schema_filter(connection)
