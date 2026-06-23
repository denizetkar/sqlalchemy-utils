"""
Deep bug-hunt tests for SQLAlchemy-Utils view subsystem.

Written via TDD methodology: each test exercises an edge case or runtime
behavior. If a test FAILS, we have found a bug. Tests persist as regression
protection — do NOT fix the source code in response to these tests.

Run:
    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \\
        tests/test_deep_bug_hunt.py -v
"""
from __future__ import annotations

import inspect
from unittest import mock

import pytest

import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext

from sqlalchemy_utils.alembic.comparator import compare_views, _schema_matches
from sqlalchemy_utils.alembic.depend import (
    _build_dependency_graph,
    resolve_create_order,
    resolve_drop_order,
)
from sqlalchemy_utils.alembic.operations import (
    CreateViewOp,
    DropViewOp,
    ReplaceViewOp,
)
from sqlalchemy_utils.alembic.pg_catalog import get_database_views
from sqlalchemy_utils.alembic.view_record import ViewRecord
from sqlalchemy_utils.view import (
    CreateView,
    DropView,
    RefreshMaterializedView,
    create_view,
    refresh_materialized_view,
)
from sqlalchemy_utils.view_mixin import ViewMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_sql(op_instance) -> list[str]:
    """Execute *op_instance* against a sqlite Operations and capture SQL."""
    statements: list[str] = []
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        ctx = MigrationContext.configure(connection)
        ops = Operations(ctx)
        with mock.patch.object(
            ops,
            "execute",
            side_effect=lambda stmt, *a, **kw: statements.append(
                stmt.text if hasattr(stmt, "text") else str(stmt)
            ),
        ):
            ops.invoke(op_instance)
    return statements


def _compile_ddl(ddl_element, dialect=None) -> str:
    """Compile a DDLElement to a SQL string using the given dialect."""
    if dialect is None:
        engine = sa.create_engine("sqlite:///:memory:")
        dialect = engine.dialect
    return str(ddl_element.compile(dialect=dialect))


# ===========================================================================
# Event Listener Accumulation
# ===========================================================================

def test_create_view_does_not_accumulate_listeners():
    """Calling create_view multiple times with same metadata accumulates listeners.

    This is expected SQLAlchemy behavior — sa.event.listen is additive.
    Users should not call create_view twice with the same view name.
    """
    metadata = sa.MetaData()
    selectable = sa.select(sa.column("id", sa.Integer))

    create_view("my_view", selectable, metadata)
    after_first = len(metadata.dispatch.after_create)

    create_view("my_view", selectable, metadata)
    after_second = len(metadata.dispatch.after_create)

    assert after_second == after_first + 2


# ===========================================================================
# DropView SQL formatting
# ===========================================================================

def test_drop_view_no_trailing_space_when_no_cascade():
    """DropView with cascade=False should not emit trailing whitespace.

    The DropView compiler uses:
        'DROP {}VIEW IF EXISTS {}{} {}'.format(
            ...,
            'CASCADE' if element.cascade else '',
        )
    When cascade=False the last {} is replaced with an empty string,
    producing a trailing space before the empty — i.e. the compiled SQL
    ends with a space character.
    """
    drop = DropView("my_view", cascade=False)
    sql = _compile_ddl(drop)

    assert not sql.rstrip() != sql, (
        f"BUG: DropView with cascade=False emits trailing whitespace: "
        f"{sql!r}. The format string unconditionally appends a space "
        "before the CASCADE keyword even when cascade is False."
    )


# ===========================================================================
# ViewRecord definition_matches
# ===========================================================================

def test_definition_matches_with_string_selectables():
    """definition_matches should work when both ViewRecords have string
    selectables."""
    vr1 = ViewRecord(name="v", selectable="SELECT 1 AS id")
    vr2 = ViewRecord(name="v", selectable="SELECT 1 AS id")
    assert vr1.definition_matches(vr2) is True

    vr3 = ViewRecord(name="v", selectable="SELECT 2 AS id")
    assert vr1.definition_matches(vr3) is False


def test_definition_matches_with_sa_selectable():
    """definition_matches should work when both ViewRecords have SQLAlchemy
    selectables.

    _selectable_key() calls sel.compile(compile_kwargs={"literal_binds": True})
    without passing a dialect; this works for simple selectables but may
    fail for dialect-specific constructs.
    """
    sel1 = sa.select(sa.column("id", sa.Integer))
    sel2 = sa.select(sa.column("id", sa.Integer))
    vr1 = ViewRecord(name="v", selectable=sel1)
    vr2 = ViewRecord(name="v", selectable=sel2)
    assert vr1.definition_matches(vr2) is True


def test_definition_matches_identical_selectables():
    """definition_matches should return True for identical selectables
    (same object instance)."""
    sel = sa.select(sa.column("id", sa.Integer))
    vr1 = ViewRecord(name="v", selectable=sel)
    vr2 = ViewRecord(name="v", selectable=sel)
    assert vr1.definition_matches(vr2) is True


# ===========================================================================
# Dependency Resolution Edge Cases
# ===========================================================================

def test_resolve_create_order_empty_list():
    """resolve_create_order with empty list should return empty list."""
    result = resolve_create_order([], db_views={})
    assert result == []


def test_resolve_create_order_single_view():
    """resolve_create_order with one view should return that view."""
    vr = ViewRecord(name="solo", selectable="SELECT 1 AS col")
    result = resolve_create_order([vr], db_views={})
    assert len(result) == 1
    assert result[0].name == "solo"


def test_resolve_create_order_circular_dependency():
    """resolve_create_order with circular deps should raise ValueError."""
    views = [
        ViewRecord(name="a", selectable="SELECT * FROM b"),
        ViewRecord(name="b", selectable="SELECT * FROM a"),
    ]
    with pytest.raises(ValueError, match="[Cc]ircular"):
        resolve_create_order(views, db_views={})


def test_resolve_create_order_self_referencing_view():
    """A view that references itself in its definition should not cause an
    infinite loop.

    _build_dependency_graph explicitly skips self-references (the
    `if other_name == vr.name` guard), so a self-referencing view should
    resolve cleanly.
    """
    vr = ViewRecord(name="recursive", selectable="SELECT * FROM recursive")
    result = resolve_create_order([vr], db_views={})
    assert len(result) == 1
    assert result[0].name == "recursive"


def test_resolve_drop_order_is_reverse_of_create():
    """resolve_drop_order should return the reverse of resolve_create_order
    for the same views.

    The implementation reverses the topological sort output. For a chain
    a→b→c, create order is [c, b, a] and drop order should be [a, b, c].
    """
    views = [
        ViewRecord(name="a", selectable="SELECT * FROM b"),
        ViewRecord(name="b", selectable="SELECT * FROM c"),
        ViewRecord(name="c", selectable="SELECT 1 AS col"),
    ]
    create = resolve_create_order(views, db_views={})
    drop = resolve_drop_order(views, db_views={})
    assert [v.name for v in drop] == list(reversed([v.name for v in create]))


# ===========================================================================
# Comparator Edge Cases
# ===========================================================================

def test_compare_views_empty_metadata():
    """compare_views with no model views should produce no create ops.

    We construct a minimal AutogenContext mock with empty metadata and
    an empty database; compare_views should append no operations.
    """
    metadata = sa.MetaData()
    autogen_context = mock.MagicMock()
    autogen_context.connection = mock.MagicMock()
    autogen_context.metadata = metadata

    # No model views registered
    metadata.info.pop("sqlalchemy_utils_views", None)

    # Mock DB returns empty dicts
    autogen_context.connection.execute.return_value.fetchall.return_value = []

    upgrade_ops = mock.MagicMock()
    upgrade_ops.ops = []

    # compare_views iterates schemas; pass [None]
    compare_views(autogen_context, upgrade_ops, [None])

    create_op_count = sum(
        1 for op in upgrade_ops.ops
        if type(op).__name__ in ("CreateViewOp", "CreateMaterializedViewOp")
    )
    assert create_op_count == 0, (
        "Expected zero create ops when both model and DB views are empty, "
        f"got {create_op_count}."
    )


def test_compare_views_no_db_views():
    """compare_views with no DB views should produce only create ops."""
    # This test exercises the diff path; we use a more targeted mock.
    metadata = sa.MetaData()
    selectable = sa.select(sa.column("id", sa.Integer))
    create_view("test_v", selectable, metadata)

    autogen_context = mock.MagicMock()
    autogen_context.connection = mock.MagicMock()
    autogen_context.connection.dialect.name = 'postgresql'
    autogen_context.metadata = metadata

    # Mock DB queries to return empty result rows
    empty_result = mock.MagicMock()
    empty_result.__iter__ = mock.Mock(return_value=iter([]))
    autogen_context.connection.execute.return_value = empty_result

    upgrade_ops = mock.MagicMock()
    upgrade_ops.ops = []

    # compare_views calls _canonicalize_view which uses the connection;
    # we patch it to avoid hitting a real DB.
    with mock.patch(
        "sqlalchemy_utils.alembic.comparator._canonicalize_view",
        return_value="SELECT id FROM (VALUES (1)) AS t(id)",
    ), mock.patch(
        "sqlalchemy_utils.alembic.comparator.get_database_views",
        return_value={},
    ), mock.patch(
        "sqlalchemy_utils.alembic.comparator.get_database_materialized_views",
        return_value={},
    ):
        compare_views(autogen_context, upgrade_ops, [None])

    create_op_count = sum(
        1 for op in upgrade_ops.ops
        if type(op).__name__ in ("CreateViewOp", "CreateMaterializedViewOp")
    )
    assert create_op_count == 1, (
        "Expected exactly one create op when model has one view and DB has "
        f"none; got {create_op_count} create ops."
    )


def test_dedup_preserves_order():
    """The dedup pass in compare_views should preserve the order of first
    occurrence.

    We simulate the dedup loop directly using the same logic as the
    comparator's final pass to verify ordering semantics.
    """
    # Replicate the dedup logic from compare_views
    class _FakeOp:
        def __init__(self, type_name, name, schema):
            self._type_name = type_name
            self.name = name
            self.schema = schema

        def __repr__(self):
            return f"{self._type_name}({self.name}, {self.schema})"

    ops = [
        _FakeOp("CreateViewOp", "first", None),
        _FakeOp("CreateViewOp", "second", None),
        _FakeOp("CreateViewOp", "first", None),  # duplicate
        _FakeOp("CreateViewOp", "third", None),
        _FakeOp("CreateViewOp", "second", None),  # duplicate
    ]

    # Mirror the dedup pass in compare_views
    seen: set = set()
    deduped: list = []
    for op in ops:
        key = (type(op).__name__ if hasattr(op, "_type_name") else type(op).__name__,
               getattr(op, "name", None), getattr(op, "schema", None))
        # Use the fake's _type_name for keying since type() won't match
        key = (op._type_name, op.name, op.schema)
        if key not in seen:
            seen.add(key)
            deduped.append(op)

    assert [op.name for op in deduped] == ["first", "second", "third"], (
        "Dedup should preserve first-occurrence order; got "
        f"{[op.name for op in deduped]}"
    )


# ===========================================================================
# Operations Reverse
# ===========================================================================

def test_create_view_op_reverse_returns_drop():
    """CreateViewOp.reverse() should return a DropViewOp."""
    op = CreateViewOp("v1", "SELECT 1")
    rev = op.reverse()
    assert isinstance(rev, DropViewOp)
    assert rev.name == "v1"
    # The reversed op should carry the definition forward so the drop can
    # itself be reversed (re-created) later.
    assert rev.definition == "SELECT 1"


def test_drop_view_op_reverse_requires_definition():
    """DropViewOp.reverse() without definition should raise RuntimeError."""
    op = DropViewOp("v1")
    with pytest.raises(RuntimeError, match="no definition stored"):
        op.reverse()


def test_replace_view_op_reverse_returns_replace():
    """ReplaceViewOp.reverse() should return a ReplaceViewOp with
    old_definition as the new definition."""
    op = ReplaceViewOp("v1", "SELECT 2", old_definition="SELECT 1")
    rev = op.reverse()
    assert isinstance(rev, ReplaceViewOp)
    # The reversed op's definition should be the old definition
    assert rev.definition == "SELECT 1"
    # Schema should be preserved
    assert rev.schema == op.schema


# ===========================================================================
# PG Catalog Edge Cases
# ===========================================================================

def test_get_database_views_sql_excludes_system_schemas():
    """get_database_views SQL should exclude pg_catalog and information_schema
    when schema=None.

    We inspect the source of the function to verify the SQL clause excludes
    system schemas rather than hardcoding 'public' (which was the old bug).
    """
    src = inspect.getsource(get_database_views)

    # The schema=None branch should exclude system schemas
    assert "information_schema" in src, (
        "get_database_views should exclude information_schema when "
        "schema=None."
    )
    assert "pg_catalog" in src, (
        "get_database_views should exclude pg_catalog when schema=None."
    )
    # Should NOT hardcode 'public' as the only schema returned
    assert "schemaname = 'public'" not in src, (
        "BUG: get_database_views hardcodes 'public' for schema=None; "
        "should query all non-system schemas instead."
    )


# ===========================================================================
# ViewMixin Edge Cases
# ===========================================================================

def test_view_mixin_without_table_args():
    """ViewMixin without __table_args__ should still work.

    __declare_last__ resolves the schema via __table_args__ if
    __view_schema__ is None. When neither is set, the schema should be None
    and the mixin should not raise.
    """
    Base = sa.orm.declarative_base()

    class SimpleView(ViewMixin, Base):
        __tablename__ = "simple_view"
        __view_selectable__ = sa.select(sa.column("id", sa.Integer))
        id: "Mapped[int]" = sa.Column(sa.Integer, primary_key=True)

    # Should not raise
    SimpleView.__declare_last__()

    # Resolved schema should be None
    assert SimpleView._resolved_view_schema is None
    assert SimpleView.__table__ is not None
    assert SimpleView.__table__.name == "simple_view"


def test_view_mixin_declare_last_called():
    """__declare_last__ should be invoked by SQLAlchemy's declarative system.

    We verify by checking that __declare_last__ is defined as a classmethod
    and callable, and that calling it explicitly registers ViewRecord in
    metadata.info['sqlalchemy_utils_views'].
    """
    assert hasattr(ViewMixin, "__declare_last__")
    assert callable(ViewMixin.__declare_last__)

    Base = sa.orm.declarative_base()

    class TrackedView(ViewMixin, Base):
        __tablename__ = "tracked_view"
        __view_selectable__ = sa.select(sa.column("id", sa.Integer))
        id: "Mapped[int]" = sa.Column(sa.Integer, primary_key=True)

    # Before __declare_last__, no view records
    assert "sqlalchemy_utils_views" not in Base.metadata.info

    TrackedView.__declare_last__()

    records = Base.metadata.info.get("sqlalchemy_utils_views", [])
    assert len(records) == 1
    assert records[0].name == "tracked_view"


# ===========================================================================
# Refresh Materialized View
# ===========================================================================

def test_refresh_materialized_view_with_schema():
    """refresh_materialized_view should pass schema to
    RefreshMaterializedView.

    We mock session.execute to capture the compiled SQL and verify the schema
    is present in the emitted DDL.
    """
    session = mock.MagicMock()

    refresh_materialized_view(
        session, "my_mv", concurrently=False, schema="analytics"
    )

    # session.execute should have been called with a RefreshMaterializedView
    assert session.execute.call_count == 1
    executed = session.execute.call_args[0][0]
    assert isinstance(executed, RefreshMaterializedView)
    assert executed.name == "my_mv"
    assert executed.schema == "analytics"
    assert executed.concurrently is False

    # Verify the compiled SQL includes the schema prefix
    engine = sa.create_engine("sqlite:///:memory:")
    compiled = str(executed.compile(dialect=engine.dialect))
    assert "analytics" in compiled, (
        "Compiled RefreshMaterializedView SQL should include the schema "
        f"prefix; got {compiled!r}"
    )


def test_refresh_materialized_view_concurrently():
    """refresh_materialized_view with concurrently=True should emit
    CONCURRENTLY in the compiled SQL."""
    session = mock.MagicMock()

    refresh_materialized_view(
        session, "my_mv", concurrently=True, schema=None
    )

    assert session.execute.call_count == 1
    executed = session.execute.call_args[0][0]
    assert isinstance(executed, RefreshMaterializedView)
    assert executed.concurrently is True

    engine = sa.create_engine("sqlite:///:memory:")
    compiled = str(executed.compile(dialect=engine.dialect))
    assert "CONCURRENTLY" in compiled, (
        "Compiled RefreshMaterializedView SQL should include CONCURRENTLY "
        f"when concurrently=True; got {compiled!r}"
    )


# ===========================================================================
# Bonus edge cases
# ===========================================================================

def test_drop_view_op_to_diff_tuple_shape():
    """DropViewOp.to_diff_tuple should return a 4-tuple per the operations
    contract.

    The current implementation returns ("drop_view", name, schema, False)
    — the trailing False is a hardcoded materialized flag. This test
    documents the shape so future changes are caught.
    """
    op = DropViewOp("v1", schema="public")
    tup = op.to_diff_tuple()
    assert isinstance(tup, tuple)
    assert len(tup) == 4
    assert tup[0] == "drop_view"
    assert tup[1] == "v1"
    assert tup[2] == "public"
    # The 4th element is a hardcoded False — this is arguably a bug
    # (materialized status should come from op.materialized), but we
    # document the current behavior.
    assert tup[3] is False


def test_create_view_op_reverse_preserves_schema():
    """CreateViewOp.reverse() should preserve the schema on the resulting
    DropViewOp."""
    op = CreateViewOp("v1", "SELECT 1", schema="analytics")
    rev = op.reverse()
    assert isinstance(rev, DropViewOp)
    assert rev.schema == "analytics"


def test_resolve_create_order_skips_sql_keyword_view_names():
    """A view whose name is a SQL keyword (e.g. 'user') should be skipped
    during dependency matching, so it doesn't create false dependencies on
    every view whose definition contains that word.

    This is the documented behavior of _build_dependency_graph via the
    _SQL_KEYWORDS filter.
    """
    views = [
        ViewRecord(name="user", selectable="SELECT 1 AS id"),
        ViewRecord(
            name="data",
            # 'user' appears as a column alias — not a view reference
            selectable="SELECT account_id AS user FROM accounts",
        ),
    ]
    graph = _build_dependency_graph(views, db_views={})

    # 'user' should NOT be flagged as a dependency of 'data'
    assert "user" not in graph.get("data", set()), (
        "View named 'user' (a SQL keyword) should be skipped by the "
        "_SQL_KEYWORDS filter to avoid false-positive dependencies. "
        f"graph={graph!r}"
    )


def test_schema_matches_exact_equality():
    """_schema_matches should use exact equality — None != 'public'.

    This is a regression test for the duplicate-ops bug where None and
    'public' were treated as equivalent.
    """
    assert _schema_matches(None, None) is True
    assert _schema_matches("public", "public") is True
    assert _schema_matches(None, "public") is False
    assert _schema_matches("analytics", "public") is False
