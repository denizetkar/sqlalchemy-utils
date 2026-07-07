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
    get_database_materialized_views,
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
    """``get_dependent_views`` returns ``{(dependent_name, schema): definition}``.

    Regression: the broken query referenced
    ``pg_depend.refobjname`` (nonexistent column) and used the *referenced*
    view's name as the dict key. After the fix, the key MUST be the
    *dependent* view's name (``_dep_test_dep_view``), not the referenced view's
    name (``_dep_test_base_view``). The key is now a ``(name, schema)``
    tuple to avoid cross-schema name collisions.
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

        dep_keys = [k for k in dependents.keys() if k[0] == "_dep_test_dep_view"]
        assert dep_keys, (
            "expected dependent key ('_dep_test_dep_view', schema) in result, got: "
            f"{sorted(dependents.keys())}"
        )

        # The referenced view's name must NOT appear as a dependent key.
        ref_keys = [k for k in dependents.keys() if k[0] == "_dep_test_base_view"]
        assert not ref_keys, (
            "referenced view name leaked into dependent dict keys: "
            f"{sorted(dependents.keys())}"
        )

        # The value must be the dependent view's definition SQL.
        dep_definition = dependents[dep_keys[0]]
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

        mv_keys = [k for k in dependents.keys() if k[0] == "_mv_dep_mv"]
        assert mv_keys, (
            "expected materialized view '_mv_dep_mv' as dependent of "
            "'_mv_dep_base_view'; got: "
            f"{sorted(dependents.keys())}"
        )
        ref_keys = [k for k in dependents.keys() if k[0] == "_mv_dep_base_view"]
        assert not ref_keys
    finally:
        _teardown_mv_dependent(connection)


def _setup_cross_schema(connection, schema_a: str, schema_b: str):
    """Two schemas, each with a same-named referenced view ``base_view``.

    ``<schema_a>.base_view`` has a real dependent ``<schema_a>.dep_view``.
    ``<schema_b>.base_view`` ALSO has a same-named dependent in its own
    schema (different body). Without a schema filter on the referenced
    view's namespace (``refn.nspname``), querying dependents of
    ``base_view`` with ``schema='<schema_a>'`` would match BOTH
    ``base_view`` rows and erroneously surface ``<schema_b>.dep_view``
    as a false positive.
    """
    for schema in (schema_a, schema_b):
        connection.execute(sa.text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
    connection.execute(sa.text(f"CREATE SCHEMA {schema_a}"))
    connection.execute(sa.text(f"CREATE SCHEMA {schema_b}"))

    connection.execute(
        sa.text(f"CREATE TABLE {schema_a}.t (id SERIAL PRIMARY KEY)"))
    connection.execute(
        sa.text(
            f"CREATE VIEW {schema_a}.base_view AS SELECT id FROM {schema_a}.t"
        )
    )
    connection.execute(
        sa.text(
            f"CREATE VIEW {schema_a}.dep_view AS "
            f"SELECT id FROM {schema_a}.base_view"
        )
    )

    # Second schema: same-named base_view WITH a dependent.
    connection.execute(
        sa.text(f"CREATE TABLE {schema_b}.t (id SERIAL PRIMARY KEY)"))
    connection.execute(
        sa.text(
            f"CREATE VIEW {schema_b}.base_view AS SELECT id FROM {schema_b}.t"
        )
    )
    connection.execute(
        sa.text(
            f"CREATE VIEW {schema_b}.dep_view AS "
            f"SELECT id FROM {schema_b}.base_view"
        )
    )
    connection.commit()


def _teardown_cross_schema(connection, schema_a: str, schema_b: str):
    for schema in (schema_a, schema_b):
        try:
            connection.execute(sa.text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        except sa.exc.SQLAlchemyError:
            connection.rollback()
    connection.commit()


@pytest.mark.infrastructure
@pytest.mark.parametrize(
    "schema_a,schema_b",
    [("_cross_schema_a", "_cross_schema_b"),
     ("_schema_filter_a", "_schema_filter_b")],
    ids=["cross_schema", "schema_filter"],
)
def test_get_dependent_views_schema_filter_excludes_other_schema(
    connection, schema_a, schema_b
):
    """Schema filter must constrain the referenced view's namespace.

    When ``schema='<schema_a>'`` is passed, ``get_dependent_views`` must
    only return dependents of ``<schema_a>.base_view``. The referenced
    view's namespace (``refn.nspname``) must be filtered, not just the
    dependent view's schema. Without the ``refn.nspname = :schema``
    filter, the query matches ``base_view`` in ANY schema and produces
    false positives (e.g. ``<schema_b>.dep_view`` which references
    ``<schema_b>.base_view``).
    """
    _setup_cross_schema(connection, schema_a, schema_b)
    try:
        dependents = get_dependent_views(
            connection, "base_view", schema=schema_a
        )
        dep_names = {k[0] for k in dependents.keys()}
        assert "dep_view" in dep_names, (
            f"expected dependent 'dep_view' in {schema_a}; got: "
            f"{sorted(dependents.keys())}"
        )
        # Every returned dependent must reference <schema_a>.base_view,
        # proving the schema filter constrained the referenced view's
        # namespace.
        for key, definition in dependents.items():
            assert f"{schema_a}.base_view" in definition, (
                f"dependent {key!r} should reference {schema_a}.base_view; "
                f"got definition: {definition!r}"
            )
            assert schema_b not in definition, (
                f"dependent {key!r} leaked cross-schema reference to "
                f"{schema_b}; definition: {definition!r}"
            )
    finally:
        _teardown_cross_schema(connection, schema_a, schema_b)


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
        dep_names = {k[0] for k in dependents.keys()}
        assert "_union_dep" in dep_names, (
            "expected regular view dependent '_union_dep'; got: "
            f"{sorted(dependents.keys())}"
        )
        assert "_union_mv" in dep_names, (
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


def _setup_same_name_dependents(connection):
    """Two dependent views sharing a name in different schemas.

    ``_same_name_a.shared_dep`` and ``_same_name_b.shared_dep`` both depend
    on a same-named referenced view ``base_view`` in their respective
    schemas. With a name-only dict key the second overwrites the first;
    the result must contain both.
    """
    for schema in ("_same_name_a", "_same_name_b"):
        connection.execute(sa.text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
    connection.execute(sa.text("CREATE SCHEMA _same_name_a"))
    connection.execute(sa.text("CREATE SCHEMA _same_name_b"))

    connection.execute(
        sa.text("CREATE TABLE _same_name_a.t (id SERIAL PRIMARY KEY)")
    )
    connection.execute(
        sa.text(
            "CREATE VIEW _same_name_a.base_view AS SELECT id FROM _same_name_a.t"
        )
    )
    connection.execute(
        sa.text(
            "CREATE VIEW _same_name_a.shared_dep AS "
            "SELECT id FROM _same_name_a.base_view"
        )
    )

    connection.execute(
        sa.text("CREATE TABLE _same_name_b.t (id SERIAL PRIMARY KEY)")
    )
    connection.execute(
        sa.text(
            "CREATE VIEW _same_name_b.base_view AS SELECT id FROM _same_name_b.t"
        )
    )
    connection.execute(
        sa.text(
            "CREATE VIEW _same_name_b.shared_dep AS "
            "SELECT id FROM _same_name_b.base_view"
        )
    )
    connection.commit()


def _teardown_same_name_dependents(connection):
    for schema in ("_same_name_a", "_same_name_b"):
        try:
            connection.execute(sa.text(f"DROP SCHEMA IF EXISTS {schema} CASCADE"))
        except sa.exc.SQLAlchemyError:
            connection.rollback()
    connection.commit()


@pytest.mark.infrastructure
def test_get_dependent_views_same_name_across_schemas(connection):
    """Two dependents with the same name in different schemas must coexist.

    ``get_dependent_views`` returns a dict; with a name-only key a second
    dependent sharing a name in another schema overwrites the first. The
    dict must be keyed by ``(dependent_name, dependent_schema)`` so both
    are preserved.
    """
    _setup_same_name_dependents(connection)
    try:
        dependents = get_dependent_views(connection, "base_view")

        assert isinstance(dependents, dict)
        # Both same-named dependents must be present (no overwrite).
        assert len(dependents) >= 2, (
            f"Expected at least 2 dependents (one per schema); got "
            f"{len(dependents)}: {sorted(dependents.keys())}"
        )

        # Each key must be a (name, schema) tuple.
        for key in dependents.keys():
            assert isinstance(key, tuple) and len(key) == 2, (
                f"dependent dict keys must be (name, schema) tuples; got "
                f"{key!r} (type {type(key).__name__})"
            )

        keys = set(dependents.keys())
        assert ("shared_dep", "_same_name_a") in keys, (
            f"missing ('shared_dep', '_same_name_a'); got {sorted(keys)}"
        )
        assert ("shared_dep", "_same_name_b") in keys, (
            f"missing ('shared_dep', '_same_name_b'); got {sorted(keys)}"
        )

        a_def = dependents[("shared_dep", "_same_name_a")]
        b_def = dependents[("shared_dep", "_same_name_b")]
        assert "_same_name_a.base_view" in a_def, (
            f"A dependent must reference _same_name_a.base_view; got {a_def!r}"
        )
        assert "_same_name_b.base_view" in b_def, (
            f"B dependent must reference _same_name_b.base_view; got {b_def!r}"
        )
    finally:
        _teardown_same_name_dependents(connection)

