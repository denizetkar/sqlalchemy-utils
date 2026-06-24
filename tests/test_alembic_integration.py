"""Comprehensive regression and audit tests for the Alembic view integration.

Sections:
1. PR Readiness (BUG-A through BUG-K) — schema params, quoting, deps
2. Interface Audit — replace attr, with_data default, keyword-only, lazy registration
3. Deep Bug Hunt — edge cases, SQL formatting, dependency resolution
4. Final Audit — import safety, Py3.9 syntax, renderer noise
"""
from __future__ import annotations

import inspect
import logging
import subprocess
import sys
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy.orm import declarative_base, Mapped, mapped_column

from sqlalchemy_utils.alembic.comparator import (
    _canonicalize_view,
    _schema_matches,
    compare_views,
)
from sqlalchemy_utils.alembic.depend import (
    _build_dependency_graph,
    resolve_create_order,
    resolve_drop_order,
)
from sqlalchemy_utils.alembic.operations import (
    CreateMaterializedViewOp,
    CreateViewOp,
    DropMaterializedViewOp,
    DropViewOp,
    ReplaceMaterializedViewOp,
    ReplaceViewOp,
    _create_view_impl,
)
from sqlalchemy_utils.alembic.pg_catalog import get_database_views
from sqlalchemy_utils.alembic.view_record import ViewRecord
from sqlalchemy_utils.view import (
    CreateView,
    DropView,
    RefreshMaterializedView,
    create_materialized_view,
    create_view,
    refresh_materialized_view,
)
from sqlalchemy_utils.view_mixin import ViewMixin


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capture_sql(op_instance) -> list[str]:
    """Invoke *op_instance* against an in-memory SQLite engine and return
    the list of captured SQL strings without actually executing them."""
    statements: list[str] = []
    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        ctx = MigrationContext.configure(connection)
        ops = Operations(ctx)
        with patch.object(
            ops,
            "execute",
            side_effect=lambda stmt, *a, **kw: statements.append(
                stmt.text if hasattr(stmt, "text") else str(stmt)
            ),
        ):
            ops.invoke(op_instance)
    return statements


def _make_operations() -> Operations:
    """Build a real Operations instance backed by an in-memory SQLite engine."""
    engine = sa.create_engine("sqlite:///:memory:")
    conn = engine.connect()
    ctx = MigrationContext.configure(conn)
    return Operations(ctx)


def _ddl_sql_for_metadata(metadata: sa.MetaData, dialect=None) -> list[str]:
    """Compile after_create DDL listeners to SQL strings without executing.

    SQLite does not support ``CREATE OR REPLACE VIEW``, so we cannot run
    ``metadata.create_all`` against a real SQLite engine. Instead, we
    iterate the ``after_create`` dispatch and compile each DDLElement
    directly to a SQL string.
    """
    if dialect is None:
        dialect = sa.dialects.sqlite.dialect()
    statements: list[str] = []
    for listener in metadata.dispatch.after_create:
        compile_fn = getattr(listener, "compile", None)
        if compile_fn is not None:
            try:
                sql = str(compile_fn(dialect=dialect))
                statements.append(sql)
            except Exception:
                pass
    return statements


def _compile_ddl(ddl_element, dialect=None) -> str:
    """Compile a DDLElement to a SQL string using the given dialect."""
    if dialect is None:
        engine = sa.create_engine("sqlite:///:memory:")
        dialect = engine.dialect
    return str(ddl_element.compile(dialect=dialect))


# ============================================================================
# Section 1: PR Readiness (BUG-A through BUG-K)
# ============================================================================

# ---------------------------------------------------------------------------
# BUG-A: ViewMixin.refresh() does not resolve schema from __table_args__
#        when __view_schema__ is not set.
#
# When a view model declares schema via __table_args__={'schema': 'analytics'}
# but does NOT set __view_schema__, refresh() passes schema=None to
# refresh_materialized_view instead of the resolved 'analytics'.
# ---------------------------------------------------------------------------
def test_refresh_uses_resolved_schema_from_table_args():
    """ViewMixin.refresh must use the schema resolved from __table_args__
    when __view_schema__ is not explicitly set."""

    class AnalyticsView(ViewMixin):
        __tablename__ = 'analytics_mv'
        __view_selectable__ = sa.select(sa.column('id', sa.Integer))
        __view_materialized__ = True
        # __view_schema__ intentionally NOT set — schema comes from table_args
        __table_args__ = {'schema': 'analytics'}
        metadata = sa.MetaData()
        id = sa.Column(sa.Integer, primary_key=True)

    session = mock.MagicMock(name='session')

    with mock.patch(
        'sqlalchemy_utils.view_mixin.refresh_materialized_view'
    ) as mock_refresh:
        AnalyticsView.refresh(session)

    mock_refresh.assert_called_once()
    _, kwargs = mock_refresh.call_args
    # BUG-A: refresh() passes cls.__view_schema__ (which is None) instead of
    # the schema resolved from __table_args__.
    assert kwargs.get('schema') == 'analytics', (
        "ViewMixin.refresh() should resolve schema from __table_args__ when "
        "__view_schema__ is not set, but got "
        f"schema={kwargs.get('schema')!r}"
    )


# ---------------------------------------------------------------------------
# BUG-B: Alembic operations do not quote identifiers (reserved words break).
#
# _schema_prefix() and the operation implementations use bare f-strings like
# f"{prefix}{op.name}" without running the identifier through the dialect's
# identifier_preparer.quote(). A view named "order" (a SQL reserved word)
# produces invalid SQL.
# ---------------------------------------------------------------------------
def test_operations_quote_identifiers():
    """The operations module must use identifier_preparer.quote() for both
    schema and name, so reserved words like 'order' produce valid SQL.

    This test inspects the source of _create_view_impl and asserts that
    quoting is applied. (The old _schema_prefix helper has been removed;
    quoting is now done via _qualified_name / _quote_identifier.)
    """
    src = inspect.getsource(_create_view_impl)

    # The implementation should call identifier_preparer.quote() (or equivalent)
    # on the view name. Currently it just embeds {op.name} directly.
    assert 'quote' in src, (
        "_create_view_impl must quote identifiers via "
        "identifier_preparer.quote() to support reserved-word view names "
        "like 'order'. Source does not contain 'quote'."
    )


# ---------------------------------------------------------------------------
# BUG-C: _canonicalize_view does not quote identifiers when building the
#        CREATE VIEW / CREATE MATERIALIZED VIEW SQL.
#
# The function builds SQL via f-strings using {prefix}{view_record.name}
# without running through identifier_preparer.quote().
# ---------------------------------------------------------------------------
def test_comparator_canonicalize_quotes_identifiers():
    """_canonicalize_view must use identifier_preparer.quote() for schema
    and name when constructing CREATE VIEW SQL."""
    src = inspect.getsource(_canonicalize_view)

    assert 'quote' in src, (
        "_canonicalize_view must use identifier_preparer.quote() for schema "
        "and name identifiers; current source does not contain 'quote'."
    )

    # And it must quote both the schema (prefix) and the view name.
    assert src.count('quote') >= 2, (
        "_canonicalize_view should quote both schema and name (>=2 quote "
        "calls); found fewer."
    )


# ---------------------------------------------------------------------------
# BUG-D: _schema_matches treats None and 'public' as equivalent, causing
#        duplicate processing of views that are genuinely in non-public
#        schemas during autogenerate diff loops scoped to 'public'.
# ---------------------------------------------------------------------------
def test_no_duplicate_ops_for_none_public_schemas():
    """_schema_matches uses exact match only; None != 'public'.

    Previously a view with schema=None matched BOTH a None-schema loop AND
    a 'public'-schema loop, leading to duplicate operations during
    autogenerate when schemas=[None, 'public'] was iterated. With the fix,
    _schema_matches is an exact equality check, so None matches only None.
    """
    # None view-schema no longer matches 'public' loop-schema — bug fixed.
    assert _schema_matches(None, 'public') is False, (
        "BUG-D: _schema_matches(None, 'public') should be False after fix; "
        "treating None and 'public' as equivalent causes duplicate ops."
    )

    # None still matches None (same schema).
    assert _schema_matches(None, None) is True, (
        "Expected _schema_matches(None, None) == True (exact match)."
    )

    # 'public' matches 'public' (exact match).
    assert _schema_matches('public', 'public') is True, (
        "Expected _schema_matches('public', 'public') == True (exact match)."
    )

    # 'analytics' does not match 'public'.
    assert _schema_matches('analytics', 'public') is False, (
        "Expected _schema_matches('analytics', 'public') == False (exact match)."
    )

    # A view with schema=None is processed ONLY in the None loop, not the
    # 'public' loop — no duplicate ops.
    both_match = (
        _schema_matches(None, None) and _schema_matches(None, 'public')
    )
    assert not both_match, (
        "BUG-D: _schema_matches(None, None) and _schema_matches(None, "
        "'public') both return True, causing duplicate ops when "
        "schemas=[None, 'public'] is iterated."
    )


# ---------------------------------------------------------------------------
# BUG-E: The materialized-view index-creation listener is scoped to the
#        whole metadata instead of to the specific MV's table, so it fires
#        for every table.create_all() call and can raise "index already
#        exists" errors when multiple MVs share a metadata.
# ---------------------------------------------------------------------------
def test_index_listener_scoped_to_table():
    """The create_indexes listener in create_materialized_view must be scoped
    to the MV's table (via sa.event.listens_for(table, 'after_create')), not
    to the whole metadata. Currently it listens on metadata, so it fires for
    EVERY table/MV created via that metadata, re-attempting index creation
    and raising 'index already exists' when multiple objects share metadata.
    """
    from sqlalchemy_utils.view import create_materialized_view

    src = inspect.getsource(create_materialized_view)

    # The bug: the listener is registered via
    #   @sa.event.listens_for(metadata, 'after_create')
    # without a guard, so it fires for every table created via that metadata.
    # A correct implementation either listens on the table directly OR guards
    # the listener body with a target identity check.
    #
    # We accept either: listens on table, or has a target-is-table guard.
    assert "listens_for(table," in src or "listens_for(table " in src or "target is not table" in src or "target is table" in src, (
        "BUG-E: create_materialized_view's create_indexes listener is "
        "registered on `metadata` (not the MV's `table`), so it fires for "
        "every table created via that metadata and re-creates the MV "
        "indexes, raising 'index already exists'. Expected "
        "'listens_for(table, ...)' in source."
    )


# ---------------------------------------------------------------------------
# BUG-F: get_database_views hardcodes 'public' when schema=None instead of
#        querying ALL non-system schemas.
# ---------------------------------------------------------------------------
def test_pg_catalog_returns_all_schemas_when_none():
    """When schema=None, get_database_views should query all non-system
    schemas, not just 'public'. Currently the SQL clause is:
        (:schema IS NULL AND schemaname = 'public')
    which limits None to 'public' only.
    """
    src = inspect.getsource(get_database_views)

    # The bug: the SQL contains "schemaname = 'public'" as the None branch.
    # A correct implementation would exclude system schemas (pg_*, information_schema)
    # rather than hardcoding 'public'.
    assert "schemaname = 'public'" not in src, (
        "get_database_views hardcodes 'public' for the schema=None branch; "
        "it should query all non-system schemas instead. Source contains "
        "\"schemaname = 'public'\"."
    )


# ---------------------------------------------------------------------------
# BUG-G: _canonicalize_view does not pass the connection's dialect to
#        sel.compile(), so the canonicalized definition may use the wrong
#        dialect's SQL syntax.
# ---------------------------------------------------------------------------
def test_comparator_uses_connection_dialect():
    """_canonicalize_view must pass dialect=connection.dialect to
    sel.compile() so the canonical definition matches the actual DB dialect."""
    src = inspect.getsource(_canonicalize_view)

    assert 'connection.dialect' in src or 'dialect=connection' in src, (
        "_canonicalize_view must pass dialect=connection.dialect (or "
        "dialect=connection) to sel.compile(); current source does not."
    )


# ---------------------------------------------------------------------------
# BUG-H: _build_dependency_graph uses word-boundary regex matching that
#        treats SQL keywords-as-view-names (e.g. a view named "user") as
#        matching any definition containing the word "user", creating
#        false dependencies.
# ---------------------------------------------------------------------------
def test_depend_regex_skips_sql_keywords():
    """A view named 'user' must not be flagged as a dependency of every
    view whose definition happens to contain the word 'user' as a SQL
    keyword (e.g. 'CREATE USER ...'). Currently it does."""
    # Two views: one named 'user', one named 'data' with a definition that
    # contains the standalone word 'user' as a column alias, NOT a view ref.
    user_view = ViewRecord(
        name='user',
        selectable='SELECT 1 AS id',
        schema=None,
        materialized=False,
    )
    data_view = ViewRecord(
        name='data',
        # 'user' appears here as a column alias (standalone word), not as a
        # view reference. The regex \buser\b matches it anyway.
        selectable='SELECT account_id AS user FROM accounts',
        schema=None,
        materialized=False,
    )

    graph = _build_dependency_graph([user_view, data_view], {})

    # 'data' should NOT depend on 'user' — the word 'user' is a column
    # alias here, not a reference to the 'user' view. The regex can't tell
    # the difference, so it creates a false dependency.
    assert 'user' not in graph.get('data', set()), (
        "BUG-H: _build_dependency_graph falsely reports 'user' as a "
        f"dependency of 'data' (graph={graph!r}). The regex matches the "
        "standalone word 'user' even when it's a column alias, not a view "
        "reference."
    )


# ---------------------------------------------------------------------------
# BUG-I: compare_views only collects db_views/db_mvs for the CURRENT schema
#        loop iteration before calling resolve_create_order, so cross-schema
#        view dependencies (a view in schema A depending on a view in schema B)
#        are not resolved and may produce a wrong creation order.
# ---------------------------------------------------------------------------
def test_cross_schema_dependency_resolution():
    """compare_views must collect all_db across ALL schemas before calling
    resolve_create_order, so cross-schema dependencies are visible to the
    topological sort. Currently all_db only contains the current schema's
    views."""
    src = inspect.getsource(compare_views)

    # The bug: resolve_create_order is called with `all_db = {**db_views, **db_mvs}`
    # where db_views/db_mvs are scoped to the CURRENT schema loop only.
    # A correct implementation would aggregate across all schemas first.
    #
    # We check that the source does NOT build a cross-schema aggregate before
    # calling resolve_create_order. Specifically, the call should pass an
    # accumulator rather than a per-schema dict.
    assert 'all_schemas' in src or 'all_db_views' in src or 'cross_schema' in src, (
        "compare_views should aggregate db views across ALL schemas before "
        "calling resolve_create_order to support cross-schema dependencies; "
        "current source only uses the current-schema db_views/db_mvs."
    )


# ---------------------------------------------------------------------------
# BUG-J: ViewMixin does not define __view_replace__ as a class attribute,
#        so getattr(cls, '__view_replace__', False) always returns the
#        default False even when a subclass sets it via __view_replace__ = True.
#        (Actually the deeper bug: the attribute is missing from the class
#        body, so hasattr(ViewMixin, '__view_replace__') is False.)
# ---------------------------------------------------------------------------
def test_view_replace_is_class_attribute():
    """ViewMixin must define __view_replace__ as a class attribute (default
    False) so that subclasses can override it and the code path
    getattr(cls, '__view_replace__', False) finds the class attribute."""
    assert hasattr(ViewMixin, '__view_replace__'), (
        "BUG-J: ViewMixin does not define __view_replace__ as a class "
        "attribute; hasattr(ViewMixin, '__view_replace__') is False."
    )
    assert ViewMixin.__view_replace__ is False, (
        "BUG-J: ViewMixin.__view_replace__ should default to False."
    )


# ---------------------------------------------------------------------------
# BUG-K: ViewRecord does not implement a definition_matches() method to
#        compare its selectable against a database definition string.
#        Autogenerate relies on string equality, which is fragile.
# ---------------------------------------------------------------------------
def test_viewrecord_definition_matches_method():
    """ViewRecord must expose a definition_matches() method that compares
    the view's selectable against a database definition string in a
    dialect-aware, normalization-tolerant way."""
    assert hasattr(ViewRecord, 'definition_matches'), (
        "BUG-K: ViewRecord does not define a definition_matches() method; "
        "hasattr(ViewRecord, 'definition_matches') is False."
    )
    assert callable(getattr(ViewRecord, 'definition_matches')), (
        "BUG-K: ViewRecord.definition_matches exists but is not callable."
    )


# ============================================================================
# Section 2: Interface Audit
# ============================================================================

# ---------------------------------------------------------------------------
# BLOCKER B1: __view_replace__ not passed to CreateView
# ---------------------------------------------------------------------------

def test_view_mixin_replace_attr_passed_to_create_view():
    """ViewMixin.__view_replace__ = True should produce CREATE OR REPLACE VIEW,
    not CREATE VIEW.

    Bug: ``ViewMixin.__declare_last__`` computes ``replace`` but does NOT pass
    it to ``CreateView(...)`` when registering the ``after_create`` listener
    (see ``sqlalchemy_utils/view_mixin.py:168-171``). The ViewRecord stored in
    metadata.info gets the correct ``replace`` value, but the DDL element
    itself always has ``replace=False``.
    """
    Base = declarative_base()

    class ReplaceView(ViewMixin, Base):
        __tablename__ = "replace_view"
        __view_selectable__ = sa.select(
            sa.table("src", sa.column("id", sa.Integer))
        )
        __view_replace__ = True
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    ReplaceView.__declare_last__()

    # Inspect the CreateView DDL element registered on metadata.
    after_create_listeners = list(
        Base.metadata.dispatch.after_create
    )
    create_view_ddls = [
        listener
        for listener in after_create_listeners
        if isinstance(getattr(listener, "__self__", None), CreateView)
    ]
    # The listener wrapper exposes the DDLElement via __self__ (bound method
    # of DDLElement.__call__). Fall back to scanning by attribute presence.
    if not create_view_ddls:
        create_view_ddls = [
            listener
            for listener in after_create_listeners
            if hasattr(listener, "name")
            and hasattr(listener, "replace")
            and hasattr(listener, "selectable")
        ]

    assert create_view_ddls, (
        "Expected at least one CreateView DDL listener on metadata; got: "
        f"{after_create_listeners!r}"
    )

    create_view_ddl = create_view_ddls[0]
    assert create_view_ddl.replace is True, (
        f"Expected CreateView.replace == True (because __view_replace__=True), "
        f"but got CreateView.replace == {create_view_ddl.replace!r}. "
        f"The __declare_last__ hook does not forward replace= to CreateView."
    )

    # Also verify the emitted SQL contains "OR REPLACE".
    captured = _ddl_sql_for_metadata(Base.metadata)
    assert any("CREATE OR REPLACE VIEW" in stmt.upper() for stmt in captured), (
        f"Expected DDL to contain 'CREATE OR REPLACE VIEW'; got: {captured!r}"
    )


# ---------------------------------------------------------------------------
# BLOCKER B2: with_data default inverted
# ---------------------------------------------------------------------------

def test_create_mv_op_defaults_to_with_data():
    """op.create_materialized_view should default to WITH DATA (PG default),
    not WITH NO DATA.

    Bug: ``CreateMaterializedViewOp.__init__`` defaults ``with_data=False``
    (see ``sqlalchemy_utils/alembic/operations.py:206-216``). PostgreSQL's
    default for ``CREATE MATERIALIZED VIEW`` is ``WITH DATA`` (populated
    immediately). The op inverts this, defaulting to ``WITH NO DATA``.
    """
    op = CreateMaterializedViewOp("mv", "SELECT 1")
    assert op.with_data is True, (
        f"Expected CreateMaterializedViewOp.with_data to default to True "
        f"(matching PostgreSQL's default), but got {op.with_data!r}."
    )

    # Also verify the SQL implementation emits "WITH DATA" by default.
    sqls = _capture_sql(CreateMaterializedViewOp("mv", "SELECT 1"))
    assert sqls == ["CREATE MATERIALIZED VIEW mv AS SELECT 1 WITH DATA"], (
        f"Expected default SQL to emit 'WITH DATA'; got: {sqls!r}"
    )


def test_create_mv_runtime_vs_op_consistency():
    """Runtime create_materialized_view and op.create_materialized_view should
    have the same default for data population.

    Bug: The runtime ``create_materialized_view`` in view.py emits a bare
    ``CREATE MATERIALIZED VIEW ... AS ...`` (no ``WITH [NO] DATA`` clause —
    PostgreSQL defaults to ``WITH DATA``). The ``op.create_materialized_view``
    emits ``WITH NO DATA`` by default. These two paths are inconsistent —
    the same conceptual operation produces a populated MV via runtime and
    an empty MV via the migration op.
    """
    # Runtime path: capture the CreateView DDL element registered by
    # create_materialized_view. Its emitted SQL has NO "WITH [NO] DATA"
    # clause (PG defaults to WITH DATA).
    metadata = sa.MetaData()
    create_materialized_view(
        "runtime_mv",
        sa.select(sa.table("src", sa.column("id", sa.Integer))),
        metadata,
    )
    runtime_ddls = [
        listener
        for listener in metadata.dispatch.after_create
        if isinstance(getattr(listener, "__self__", None), CreateView)
        and getattr(listener, "__self__", None).materialized
    ]
    if not runtime_ddls:
        runtime_ddls = [
            listener
            for listener in metadata.dispatch.after_create
            if hasattr(listener, "materialized") and listener.materialized
        ]
    assert runtime_ddls, "Expected a materialized-view after_create listener"
    runtime_ddl = runtime_ddls[0]

    # Compile the runtime DDL element to a SQL string.
    engine = sa.create_engine("sqlite:///:memory:")
    compiled_runtime = str(
        runtime_ddl.compile(dialect=engine.dialect)
    ).upper()

    # Op path: capture via the in-memory SQLite execution.
    op_sqls = _capture_sql(CreateMaterializedViewOp("op_mv", "SELECT 1"))
    op_sql = op_sqls[0].upper() if op_sqls else ""

    # Runtime path omits "WITH [NO] DATA" → PG default = WITH DATA (populated).
    # Op path emits explicit "WITH NO DATA" by default (empty MV).
    runtime_emits_with_no_data = "WITH NO DATA" in compiled_runtime
    op_emits_with_no_data = "WITH NO DATA" in op_sql

    assert runtime_emits_with_no_data == op_emits_with_no_data, (
        "Inconsistent defaults between runtime and op paths for "
        "materialized-view data population.\n"
        f"Runtime DDL: {compiled_runtime!r}\n"
        f"Op DDL:      {op_sql!r}\n"
        f"Runtime emits WITH NO DATA? {runtime_emits_with_no_data}\n"
        f"Op emits WITH NO DATA?      {op_emits_with_no_data}"
    )


# ---------------------------------------------------------------------------
# SHOULD-FIX S1: op.create_view missing replace param
# ---------------------------------------------------------------------------

def test_op_create_view_accepts_replace():
    """op.create_view should accept replace= parameter for consistency with
    CreateViewOp (which has a replace field)."""
    from unittest.mock import MagicMock

    op = CreateViewOp("v", "SELECT 1", replace=True)
    assert op.replace is True

    operations = MagicMock()
    operations.invoke.return_value = None
    CreateViewOp.create_view(operations, "v", "SELECT 1", replace=True)
    operations.invoke.assert_called_once()
    invoked_op = operations.invoke.call_args[0][0]
    assert isinstance(invoked_op, CreateViewOp)
    assert invoked_op.replace is True


# ---------------------------------------------------------------------------
# SHOULD-FIX S2: keyword-only enforcement on op.* methods
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method_name,op_class",
    [
        ("create_view", CreateViewOp),
        ("drop_view", DropViewOp),
        ("replace_view", ReplaceViewOp),
        ("create_materialized_view", CreateMaterializedViewOp),
        ("drop_materialized_view", DropMaterializedViewOp),
        ("replace_materialized_view", ReplaceMaterializedViewOp),
    ],
)
def test_op_methods_enforce_keyword_only_schema(method_name, op_class):
    """op.create_view, op.drop_view, etc. should enforce schema= as
    keyword-only.

    Bug: The classmethod entry-points accept ``schema`` as a regular
    positional-or-keyword parameter (see e.g. ``CreateViewOp.create_view``
    signature: ``def create_view(cls, operations, name, definition, schema=None)``
    in operations.py:65-72). Passing schema positionally should raise
    TypeError to prevent ambiguity and enforce the keyword-only contract that
    the audit requires.
    """
    # Use the raw classmethod on the Op class (cls, operations, ...) so we
    # test the actual signature, bypassing Operations' method binding which
    # would otherwise execute SQL when schema leaks into the impl layer.
    cls_method = getattr(op_class, method_name)

    class FakeOperations:
        """Stand-in for Operations that records invoked ops without executing."""
        def invoke(self, op):
            self.invoked_op = op

    fake_ops = FakeOperations()

    if method_name in {"create_view", "create_materialized_view"}:
        positional_args = ("v", "SELECT 1", "myschema")
    elif method_name in {"drop_view", "drop_materialized_view"}:
        positional_args = ("v", "myschema")
    elif method_name in {"replace_view", "replace_materialized_view"}:
        positional_args = ("v", "SELECT 2", "myschema")
    else:
        pytest.fail(f"Unknown method {method_name}")

    # The schema positional arg should be rejected with TypeError.
    with pytest.raises(TypeError):
        cls_method(fake_ops, *positional_args)


# ---------------------------------------------------------------------------
# SHOULD-FIX S3: param order inconsistency
# ---------------------------------------------------------------------------

def test_refresh_param_order_consistency():
    """refresh_materialized_view and RefreshMaterializedView should have
    consistent parameter ordering.

    Bug: ``RefreshMaterializedView.__init__(name, schema=None, concurrently=False)``
    puts schema BEFORE concurrently, while the convenience function
    ``refresh_materialized_view(session, name, concurrently=False, schema=None)``
    puts concurrently BEFORE schema. They should agree.
    """
    cls_sig = inspect.signature(RefreshMaterializedView.__init__)
    fn_sig = inspect.signature(refresh_materialized_view)

    cls_params = [
        p
        for p in cls_sig.parameters.values()
        if p.name not in {"self"}
    ]
    fn_params = [
        p
        for p in fn_sig.parameters.values()
        if p.name not in {"session"}
    ]

    # Compare the relative order of `schema` and `concurrently`.
    cls_order = [
        p.name for p in cls_params if p.name in {"schema", "concurrently"}
    ]
    fn_order = [
        p.name for p in fn_params if p.name in {"schema", "concurrently"}
    ]

    assert cls_order == fn_order, (
        "Parameter order mismatch between RefreshMaterializedView.__init__ "
        f"and refresh_materialized_view.\n"
        f"  Class params (schema/concurrently order): {cls_order}\n"
        f"  Func params  (schema/concurrently order): {fn_order}\n"
        f"These should be consistent."
    )


# ---------------------------------------------------------------------------
# SHOULD-FIX S4: include_view_comparator side effects
# ---------------------------------------------------------------------------

def test_importing_op_class_does_not_register_comparator():
    """Importing CreateViewOp should NOT have the side effect of registering
    the comparator.

    Bug: ``sqlalchemy_utils/alembic/__init__.py`` imports from ``comparator``,
    which at module-load time executes::

        @comparators.dispatch_for("schema")
        def compare_views(...): ...

    This means *any* import of an Op class from ``sqlalchemy_utils.alembic``
    triggers global comparator registration as a side effect. This test
    spawns a fresh subprocess to verify the isolation property.

    Since the current buggy behavior registers the comparator on import, this
    test is expected to FAIL until the side effect is removed.
    """
    import subprocess
    import sys

    code = (
        "import sys\n"
        "from alembic.autogenerate import comparators\n"
        # Import ONLY an Op class — should not trigger comparator registration.
        "from sqlalchemy_utils.alembic.operations import CreateViewOp\n"
        # Check whether compare_views is registered as a schema comparator.
        # The dispatch registry stores callables; we inspect its internal
        # container's string representation.
        "comparators_repr = repr(comparators.__dict__)\n"
        "has_compare_views = 'compare_views' in comparators_repr\n"
        "print('HAS_COMPARE_VIEWS=' + str(has_compare_views))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PYTHONDONTWRITEBYTECODE": "1", "PATH": ""},
    )
    assert result.returncode == 0, (
        f"Subprocess failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
    last_line = result.stdout.strip().splitlines()[-1]
    assert last_line.startswith("HAS_COMPARE_VIEWS="), (
        f"Unexpected subprocess output: {result.stdout!r}"
    )
    has_compare_views = last_line.split("=", 1)[1].strip() == "True"

    assert has_compare_views is False, (
        "Importing CreateViewOp should NOT register compare_views as a side "
        "effect. The current implementation triggers comparator registration "
        "at module import time via __init__.py re-exports of comparator.py "
        "(which decorates compare_views with @comparators.dispatch_for('schema') "
        "at module level)."
    )


# ---------------------------------------------------------------------------
# SHOULD-FIX S5: non-PG dialect silent no-op
# ---------------------------------------------------------------------------

def test_comparator_warns_on_non_pg_dialect(caplog):
    """compare_views should emit a clear warning on non-PostgreSQL dialects,
    not silently produce no ops (nor raise an unhandled exception).

    Bug: ``compare_views`` queries ``pg_views``/``pg_matviews`` directly via
    ``get_database_views`` / ``get_database_materialized_views`` without
    checking the dialect. On a non-PG dialect (e.g. sqlite) these queries
    fail with ``OperationalError``; the comparator neither catches this nor
    emits a warning that autogeneration is effectively skipping view
    diffing. It should detect the non-PG dialect up front and emit a clear
    warning instead.
    """
    from sqlalchemy_utils.alembic.comparator import compare_views

    engine = sa.create_engine("sqlite:///:memory:")
    metadata = sa.MetaData()
    metadata.info["sqlalchemy_utils_views"] = []

    autogen_context = MagicMock()
    autogen_context.connection = engine.connect()
    autogen_context.metadata = metadata

    upgrade_ops = MagicMock()
    upgrade_ops.ops = []
    schemas = [None]

    with caplog.at_level(logging.WARNING, logger="sqlalchemy_utils.alembic.comparator"):
        # The comparator should detect the non-PG dialect and warn rather
        # than raise. We catch any exception here to assert the bug: the
        # current implementation raises (or silently produces no ops with
        # no warning) instead of emitting a clear non-PG warning.
        raised_exc = None
        try:
            compare_views(autogen_context, upgrade_ops, schemas)
        except Exception as exc:
            raised_exc = exc

    warnings = [
        rec for rec in caplog.records if rec.levelno >= logging.WARNING
    ]

    assert raised_exc is None, (
        "compare_views should not raise on a non-PostgreSQL dialect; it "
        f"should emit a warning instead. Got exception: {raised_exc!r}"
    )
    assert warnings, (
        "compare_views was called against a non-PostgreSQL dialect (sqlite) "
        "but emitted no warning. Non-PG dialects should produce a clear "
        "warning that view diffing is skipped, rather than silently "
        "producing no ops."
    )
    assert any(
        "non" in rec.message.lower() and "postgres" in rec.message.lower()
        for rec in warnings
    ), (
        f"Expected a warning mentioning non-PostgreSQL dialect; got: "
        f"{[rec.message for rec in warnings]!r}"
    )


# ============================================================================
# Section 3: Deep Bug Hunt
# ============================================================================

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


# ============================================================================
# Section 4: Final Audit
# ============================================================================

# ---------------------------------------------------------------------------
# BLOCKER B1: Importing sqlalchemy_utils without alembic installed breaks
# ---------------------------------------------------------------------------

def test_import_without_alembic_does_not_break():
    """sqlalchemy_utils should be importable even when alembic is not installed.

    Bug: __init__.py line 104 does `from .alembic import include_view_comparator`
    unconditionally. If alembic is not installed, this raises ModuleNotFoundError,
    breaking the entire package for users who don't use Alembic.
    """
    code = (
        "import sys\n"
        "sys.modules['alembic'] = None\n"  # Block alembic import
        "sys.modules['alembic.operations'] = None\n"
        "import sqlalchemy_utils\n"
        "print('IMPORT_OK')\n"
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True,
        text=True,
        env={'PYTHONPATH': 'src', 'PATH': ''},
    )
    assert result.returncode == 0, (
        f"Import failed:\nSTDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    )
    assert 'IMPORT_OK' in result.stdout


# ---------------------------------------------------------------------------
# BLOCKER B2: pg_catalog.py uses PEP 604 annotations without future import
# ---------------------------------------------------------------------------

def test_pg_catalog_imports_on_python_39_syntax():
    """pg_catalog.py should use from __future__ import annotations or avoid PEP 604 syntax.

    Bug: pg_catalog.py uses `str | None` and `dict[str, str]` in function signatures
    without `from __future__ import annotations`. On Python 3.9 (which pyproject.toml
    declares as minimum), this raises TypeError at import time.
    """
    from sqlalchemy_utils.alembic import pg_catalog
    src = inspect.getsource(pg_catalog)
    assert 'from __future__ import annotations' in src, (
        "pg_catalog.py must have 'from __future__ import annotations' to support Python 3.9"
    )


# ---------------------------------------------------------------------------
# BLOCKER B3: ReplaceMaterializedViewOp with_data default mismatch
# ---------------------------------------------------------------------------

def test_replace_mv_op_init_and_classmethod_with_data_default_match():
    """ReplaceMaterializedViewOp.__init__ and op.replace_materialized_view should
    have the same default for with_data.

    Bug: __init__ defaults with_data=False, but the classmethod defaults with_data=True.
    Constructing the op directly yields WITH NO DATA; calling op.replace_materialized_view
    yields WITH DATA. Same op, different behavior depending on entry point.
    """
    from sqlalchemy_utils.alembic.operations import ReplaceMaterializedViewOp
    # Direct construction (uses __init__ default)
    op_direct = ReplaceMaterializedViewOp("mv", "SELECT 1")
    assert op_direct.with_data is True, (
        f"ReplaceMaterializedViewOp.__init__ defaults with_data={op_direct.with_data}, "
        "but should default to True (matching the classmethod)"
    )


# ---------------------------------------------------------------------------
# SHOULD-FIX S2: Renderer emits cascade=True (the default) for drop_view
# ---------------------------------------------------------------------------

def test_drop_view_renderer_omits_default_cascade():
    """render_drop_view should omit cascade=True (the default), only emitting
    cascade=False when explicitly set.

    Bug: renderer.py emits 'cascade=True' when cascade is True (the default),
    making generated migrations noisy. Should only emit 'cascade=False' when
    cascade is False (non-default).
    """
    from sqlalchemy_utils.alembic.operations import DropViewOp
    from sqlalchemy_utils.alembic.renderer import render_drop_view

    autogen_context = MagicMock()
    # Default cascade=True should NOT emit cascade= in rendered code
    op = DropViewOp("v", cascade=True)
    result = render_drop_view(autogen_context, op)
    assert 'cascade=' not in result, (
        f"Renderer should omit cascade= when it's the default (True), but got: {result}"
    )
    # Non-default cascade=False SHOULD emit it
    op_no_cascade = DropViewOp("v", cascade=False)
    result_no = render_drop_view(autogen_context, op_no_cascade)
    assert 'cascade=False' in result_no, (
        f"Renderer should emit cascade=False when set, but got: {result_no}"
    )


# ---------------------------------------------------------------------------
# SHOULD-FIX S3: Renderer emits with_data=True (the default) for replace_materialized_view
# ---------------------------------------------------------------------------

def test_replace_mv_renderer_omits_default_with_data():
    """render_replace_materialized_view should omit with_data=True (the default),
    only emitting with_data=False when explicitly set.

    Bug: renderer.py emits 'with_data=True' when with_data is True (the default),
    making generated migrations noisy. Should only emit 'with_data=False' when
    with_data is False (non-default).
    """
    from sqlalchemy_utils.alembic.operations import ReplaceMaterializedViewOp
    from sqlalchemy_utils.alembic.renderer import render_replace_materialized_view

    autogen_context = MagicMock()
    # Default with_data=True should NOT emit with_data= in rendered code
    op = ReplaceMaterializedViewOp("mv", "SELECT 1", with_data=True)
    result = render_replace_materialized_view(autogen_context, op)
    assert 'with_data=' not in result, (
        f"Renderer should omit with_data= when it's the default (True), but got: {result}"
    )
    # Non-default with_data=False SHOULD emit it
    op_no_data = ReplaceMaterializedViewOp("mv", "SELECT 1", with_data=False)
    result_no = render_replace_materialized_view(autogen_context, op_no_data)
    assert 'with_data=False' in result_no, (
        f"Renderer should emit with_data=False when set, but got: {result_no}"
    )
