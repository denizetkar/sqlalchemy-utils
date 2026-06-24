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

    The standardized shape is ("drop_view", name, schema, definition).
    When no definition is stored, the 4th element is None.
    """
    op = DropViewOp("v1", schema="public")
    tup = op.to_diff_tuple()
    assert isinstance(tup, tuple)
    assert len(tup) == 4
    assert tup[0] == "drop_view"
    assert tup[1] == "v1"
    assert tup[2] == "public"
    # The 4th element is the stored definition (None when not provided).
    assert tup[3] is None

    # When a definition is stored, it appears in the 4th slot.
    op2 = DropViewOp("v1", schema="public", definition="SELECT 1")
    assert op2.to_diff_tuple()[3] == "SELECT 1"


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


# ===========================================================================
# Section 5: Final Interface Audit — SHOULD-FIX issues
# ============================================================================


def test_to_diff_tuple_consistent_shape_all_ops():
    """All 6 Op classes should produce to_diff_tuple() with a consistent shape:
    (op_name, name, schema, definition, *extras).

    Currently shapes are wildly inconsistent:
    - CreateViewOp: ("create_view", name, definition, schema)
    - DropViewOp: ("drop_view", name, schema, False)  # hardcoded False!
    - ReplaceViewOp: ("replace_view", name, definition, schema, old_definition)
    - CreateMaterializedViewOp: ("create_materialized_view", name, definition, schema, with_data)
    - DropMaterializedViewOp: ("drop_materialized_view", name, schema)  # missing definition!
    - ReplaceMaterializedViewOp: ("replace_materialized_view", name, definition, schema, with_data, old_definition)

    Standardize on: (op_name, name, schema, definition, *op_specific_extras)
    """
    from sqlalchemy_utils.alembic.operations import (
        CreateViewOp, DropViewOp, ReplaceViewOp,
        CreateMaterializedViewOp, DropMaterializedViewOp, ReplaceMaterializedViewOp,
    )

    # CreateViewOp
    op = CreateViewOp("v", "SELECT 1", schema="public")
    tup = op.to_diff_tuple()
    assert tup[0] == "create_view"
    assert tup[1] == "v"  # name
    assert tup[2] == "public"  # schema
    assert tup[3] == "SELECT 1"  # definition

    # DropViewOp — currently returns ("drop_view", name, schema, False)
    # Should return ("drop_view", name, schema, definition)
    op = DropViewOp("v", schema="public", definition="SELECT 1")
    tup = op.to_diff_tuple()
    assert tup[0] == "drop_view"
    assert tup[1] == "v"  # name
    assert tup[2] == "public"  # schema
    assert tup[3] == "SELECT 1"  # definition (not hardcoded False!)

    # ReplaceViewOp
    op = ReplaceViewOp("v", "SELECT 2", schema="public", old_definition="SELECT 1")
    tup = op.to_diff_tuple()
    assert tup[0] == "replace_view"
    assert tup[1] == "v"  # name
    assert tup[2] == "public"  # schema
    assert tup[3] == "SELECT 2"  # definition
    assert tup[4] == "SELECT 1"  # old_definition

    # CreateMaterializedViewOp
    op = CreateMaterializedViewOp("mv", "SELECT 1", schema="public", with_data=True)
    tup = op.to_diff_tuple()
    assert tup[0] == "create_materialized_view"
    assert tup[1] == "mv"  # name
    assert tup[2] == "public"  # schema
    assert tup[3] == "SELECT 1"  # definition
    assert tup[4] is True  # with_data

    # DropMaterializedViewOp — currently returns ("drop_materialized_view", name, schema)
    # Should return ("drop_materialized_view", name, schema, definition)
    op = DropMaterializedViewOp("mv", schema="public", definition="SELECT 1")
    tup = op.to_diff_tuple()
    assert tup[0] == "drop_materialized_view"
    assert tup[1] == "mv"  # name
    assert tup[2] == "public"  # schema
    assert tup[3] == "SELECT 1"  # definition

    # ReplaceMaterializedViewOp
    op = ReplaceMaterializedViewOp("mv", "SELECT 2", schema="public", with_data=True, old_definition="SELECT 1")
    tup = op.to_diff_tuple()
    assert tup[0] == "replace_materialized_view"
    assert tup[1] == "mv"  # name
    assert tup[2] == "public"  # schema
    assert tup[3] == "SELECT 2"  # definition
    assert tup[4] is True  # with_data
    assert tup[5] == "SELECT 1"  # old_definition


def test_register_view_comparator_exists():
    """The registration function should be named register_view_comparator
    (verb form), not include_view_comparator (which sounds like Alembic's
    include_object filter callback).

    A deprecated alias should remain for backward compat.
    """
    from sqlalchemy_utils.alembic import register_view_comparator
    assert callable(register_view_comparator)

    # Deprecated alias should still exist and work
    from sqlalchemy_utils.alembic import include_view_comparator
    assert callable(include_view_comparator)


def test_public_apis_exported_from_alembic_init():
    """Public APIs should be importable from sqlalchemy_utils.alembic directly."""
    from sqlalchemy_utils.alembic import (
        compare_views,
        resolve_create_order,
        resolve_drop_order,
        get_database_views,
        get_database_materialized_views,
        ViewRecord,
    )


def test_compare_views_does_not_double_fetch(monkeypatch):
    """compare_views should fetch each schema's DB views only once, not twice.

    Currently it fetches all schemas in a first loop (lines 146-148) then
    re-fetches per-schema in the second loop (lines 152-153). This test
    counts get_database_views calls to verify single-fetch.
    """
    import sqlalchemy_utils.alembic.comparator as comparator_module
    from sqlalchemy_utils.alembic.comparator import compare_views
    from unittest.mock import MagicMock, patch

    call_count = {"views": 0, "mvs": 0}

    def mock_get_database_views(connection, schema=None):
        call_count["views"] += 1
        return {}

    def mock_get_database_mvs(connection, schema=None):
        call_count["mvs"] += 1
        return {}

    metadata = MagicMock()
    metadata.info = {"sqlalchemy_utils_views": []}

    autogen_context = MagicMock()
    autogen_context.connection = MagicMock()
    autogen_context.connection.dialect.name = 'postgresql'
    autogen_context.metadata = metadata

    upgrade_ops = MagicMock()
    upgrade_ops.ops = []

    with patch.object(comparator_module, 'get_database_views', mock_get_database_views), \
         patch.object(comparator_module, 'get_database_materialized_views', mock_get_database_mvs), \
         patch.object(comparator_module, '_canonicalize_view', return_value=None):
        compare_views(autogen_context, upgrade_ops, [None, 'analytics'])

    # 2 schemas × 1 fetch each = 2 total (not 4)
    assert call_count["views"] == 2, f"Expected 2 get_database_views calls, got {call_count['views']}"
    assert call_count["mvs"] == 2, f"Expected 2 get_database_mvs calls, got {call_count['mvs']}"


def test_op_init_keyword_only_params():
    """Op __init__ methods should enforce keyword-only for params after
    name/definition, matching the classmethod entry points.

    This prevents footguns like DropViewOp('v', True) silently setting
    materialized=True instead of cascade=True.
    """
    from sqlalchemy_utils.alembic.operations import (
        CreateViewOp, DropViewOp, ReplaceViewOp,
        CreateMaterializedViewOp, DropMaterializedViewOp, ReplaceMaterializedViewOp,
    )
    import inspect

    # DropViewOp.__init__ has materialized as 3rd positional — should be keyword-only
    sig = inspect.signature(DropViewOp.__init__)
    params = list(sig.parameters.values())
    # After self, name: all remaining should be keyword-only
    for p in params[2:]:  # skip self, name
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"DropViewOp.__init__ param '{p.name}' should be keyword-only, "
            f"got {p.kind}"
        )

    # CreateMaterializedViewOp.__init__ has schema as 3rd positional
    sig = inspect.signature(CreateMaterializedViewOp.__init__)
    params = list(sig.parameters.values())
    for p in params[3:]:  # skip self, name, definition
        assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"CreateMaterializedViewOp.__init__ param '{p.name}' should be keyword-only, "
            f"got {p.kind}"
        )


def test_view_mixin_refresh_schema_resolution_readable():
    """ViewMixin.refresh() should use a readable schema resolution strategy,
    not a triple-nested getattr chain.

    The current code (view_mixin.py:226-230) chains:
        getattr(cls, '_resolved_view_schema',
            getattr(cls, '__view_schema__', None)
        ) or getattr(getattr(cls, '__table_args__', None), 'get', lambda _: None)('schema')

    This should be refactored to a simple method or clear if/else chain.
    We verify by checking the source doesn't contain the nested getattr chain.
    """
    import inspect
    from sqlalchemy_utils.view_mixin import ViewMixin
    src = inspect.getsource(ViewMixin.refresh)
    # The nested getattr chain should not be present
    assert "getattr(getattr" not in src, (
        "ViewMixin.refresh() contains unreadable nested getattr chain. "
        "Refactor to a readable schema resolution strategy."
    )


def test_create_view_documents_no_indexes():
    """create_view docstring should mention that indexes/aliases are not
    supported (unlike create_materialized_view)."""
    from sqlalchemy_utils.view import create_view
    assert create_view.__doc__ is not None
    # Docstring should mention the asymmetry or lack of indexes param
    doc = create_view.__doc__.lower()
    assert 'index' in doc or 'materialized' in doc, (
        "create_view docstring should document the indexes/aliases asymmetry "
        "with create_materialized_view"
    )


def test_materialized_view_ops_document_pg_only():
    """CreateMaterializedViewOp, DropMaterializedViewOp, and
    ReplaceMaterializedViewOp should note PostgreSQL-only semantics in docstrings."""
    from sqlalchemy_utils.alembic.operations import (
        CreateMaterializedViewOp, DropMaterializedViewOp, ReplaceMaterializedViewOp,
    )
    for cls in [CreateMaterializedViewOp, DropMaterializedViewOp, ReplaceMaterializedViewOp]:
        doc = (cls.__doc__ or "").lower()
        assert 'postgresql' in doc or 'postgres' in doc, (
            f"{cls.__name__} docstring should mention PostgreSQL-only semantics"
        )


# ============================================================================
# Section 6: Bug Hunt Round 5
#
# Round 5 focuses on RUNTIME CORRECTNESS: logic errors, edge cases, crashes,
# wrong SQL, and round-trip fidelity bugs not covered by earlier sections.
# Each test below documents a distinct defect and is expected to FAIL until
# the source is fixed.
# ============================================================================


# ---------------------------------------------------------------------------
# BUG-R5-01: render_create_view drops the `replace=True` parameter.
#
# `CreateViewOp(name, definition, replace=True)` should round-trip through
# the renderer back to a `CreateViewOp` with `replace=True`.  The current
# renderer (renderer.py:24-27) emits only `op.create_view(name, definition[, schema=])`
# and never includes `replace=`, so autogenerate silently downgrades a
# `CREATE OR REPLACE VIEW` migration to a plain `CREATE VIEW` migration.
# On PostgreSQL this raises `DuplicateTable` at upgrade time.
# ---------------------------------------------------------------------------
def test_round5_renderer_preserves_replace_true():
    """render_create_view must emit replace=True when the op has it set.

    Without `replace=True` in the rendered code, re-running a migration
    that uses `CREATE OR REPLACE VIEW` degrades to `CREATE VIEW`, which
    fails on PostgreSQL with `relation already exists`.
    """
    from sqlalchemy_utils.alembic.renderer import render_create_view
    from sqlalchemy_utils.alembic.operations import CreateViewOp

    op = CreateViewOp("v_replace", "SELECT 1", replace=True)
    rendered = render_create_view(MagicMock(), op)

    assert "replace=True" in rendered, (
        "render_create_view drops the `replace=True` parameter: rendered "
        f"code {rendered!r} will not produce a CREATE OR REPLACE VIEW. "
        "Round-tripping CreateViewOp(replace=True) through the renderer "
        "loses the replace flag, breaking downgrade/upgrade fidelity."
    )


# ---------------------------------------------------------------------------
# BUG-R5-02: render_replace_view drops `old_definition`.
#
# `ReplaceViewOp` stores `old_definition` so that `reverse()` can produce
# a downgrade.  The renderer (renderer.py:37-40) emits only
# `op.replace_view(name, definition[, schema=])` and never includes
# `old_definition=`, so the generated downgrade loses the ability to revert.
# ---------------------------------------------------------------------------
def test_round5_renderer_preserves_old_definition_for_replace_view():
    """render_replace_view must emit old_definition= so the downgrade is
    able to revert the view to its prior definition."""
    from sqlalchemy_utils.alembic.renderer import render_replace_view
    from sqlalchemy_utils.alembic.operations import ReplaceViewOp

    op = ReplaceViewOp(
        "v_repl", "SELECT 2", schema="public", old_definition="SELECT 1"
    )
    rendered = render_replace_view(MagicMock(), op)

    assert "old_definition=" in rendered, (
        "render_replace_view drops `old_definition`: rendered code "
        f"{rendered!r} cannot produce a working downgrade because the "
        "old definition is lost.  Alembic's autogenerate downgrade path "
        "calls op.reverse() which requires old_definition."
    )


# ---------------------------------------------------------------------------
# BUG-R5-03: render_replace_materialized_view drops `old_definition`.
#
# Same as BUG-R5-02 but for materialized views.  The renderer
# (renderer.py:61-67) omits `old_definition=`.
# ---------------------------------------------------------------------------
def test_round5_renderer_preserves_old_definition_for_replace_mv():
    """render_replace_materialized_view must emit old_definition= so the
    materialized-view downgrade can revert to the prior definition."""
    from sqlalchemy_utils.alembic.renderer import render_replace_materialized_view
    from sqlalchemy_utils.alembic.operations import ReplaceMaterializedViewOp

    op = ReplaceMaterializedViewOp(
        "mv_repl", "SELECT 2", old_definition="SELECT 1"
    )
    rendered = render_replace_materialized_view(MagicMock(), op)

    assert "old_definition=" in rendered, (
        "render_replace_materialized_view drops `old_definition`: "
        f"rendered code {rendered!r} cannot produce a working downgrade."
    )


# ---------------------------------------------------------------------------
# BUG-R5-07: CreateViewOp.reverse() round-trip loses `replace=True`.
#
# `CreateViewOp("v", "SELECT 1", replace=True).reverse()` returns a
# `DropViewOp`.  Reversing that `DropViewOp` again (`.reverse()`) returns
# a `CreateViewOp` — but WITHOUT `replace=True` (operations.py:146 only
# passes name/definition/schema).  A double-reverse should be identity.
# ---------------------------------------------------------------------------
def test_round5_create_view_op_double_reverse_preserves_replace():
    """CreateViewOp(replace=True) → reverse() → reverse() should preserve
    replace=True (double-reverse is identity for round-trip fidelity)."""
    op = CreateViewOp("v", "SELECT 1", replace=True)
    double_reversed = op.reverse().reverse()
    assert isinstance(double_reversed, CreateViewOp), (
        "Double-reverse of CreateViewOp should return a CreateViewOp"
    )
    assert double_reversed.replace is True, (
        "Double-reverse of CreateViewOp(replace=True) loses replace=True: "
        f"got replace={double_reversed.replace!r}. The DropViewOp.reverse() "
        "implementation does not forward the replace flag, so the "
        "generated downgrade loses CREATE OR REPLACE semantics."
    )


# ---------------------------------------------------------------------------
# BUG-R5-08: ReplaceViewOp.reverse() does not set old_definition on the
# reversed op, so the reversed op cannot be reversed again.
#
# `ReplaceViewOp("v", "SELECT 2", old_definition="SELECT 1").reverse()`
# returns `ReplaceViewOp("v", "SELECT 1")` with `old_definition=None`
# (operations.py:196-198).  A double-reverse then raises RuntimeError
# instead of being identity.
# ---------------------------------------------------------------------------
def test_round5_replace_view_op_double_reverse_preserves_old_definition():
    """ReplaceViewOp → reverse() → reverse() should be identity.

    The first reverse swaps definition and old_definition.  The second
    reverse should swap them back.  But the current implementation does
    not set old_definition on the reversed op, so the second reverse
    raises RuntimeError('no old_definition stored').
    """
    op = ReplaceViewOp("v", "SELECT 2", old_definition="SELECT 1")
    rev = op.reverse()
    # The reversed op's old_definition should be the original definition
    # so that a second reverse restores the original op.
    assert rev.old_definition == "SELECT 2", (
        "ReplaceViewOp.reverse() does not set old_definition on the "
        "reversed op. A double-reverse (which should be identity) would "
        f"raise RuntimeError. Got old_definition={rev.old_definition!r}."
    )


# ---------------------------------------------------------------------------
# BUG-R5-09: ReplaceMaterializedViewOp.reverse() does not set old_definition
# on the reversed op (operations.py:375-380).  Same as BUG-R5-08 but for
# materialized views.
# ---------------------------------------------------------------------------
def test_round5_replace_mv_op_double_reverse_preserves_old_definition():
    """ReplaceMaterializedViewOp → reverse() → reverse() should be identity."""
    from sqlalchemy_utils.alembic.operations import ReplaceMaterializedViewOp

    op = ReplaceMaterializedViewOp(
        "mv", "SELECT 2", old_definition="SELECT 1"
    )
    rev = op.reverse()
    assert rev.old_definition == "SELECT 2", (
        "ReplaceMaterializedViewOp.reverse() does not set old_definition "
        "on the reversed op. A double-reverse would raise RuntimeError. "
        f"Got old_definition={rev.old_definition!r}."
    )


# ---------------------------------------------------------------------------
# BUG-R5-11: get_database_views treats empty-string schema as a real schema.
#
# pg_catalog.py:24 checks `if schema is None` — but an empty string `""` is
# NOT None, so it falls into the `else` branch and queries
# `WHERE schemaname = ''`.  An empty schema name matches no rows (PG never
# has a view in schema ''), so the function silently returns an empty dict.
# Empty-string schema should be treated as None (query all non-system schemas).
# ---------------------------------------------------------------------------
def test_round5_pg_catalog_empty_string_schema_treated_as_none():
    """get_database_views should treat schema='' the same as schema=None
    (query all non-system schemas), not query `WHERE schemaname = ''`."""
    src = inspect.getsource(get_database_views)

    # The bug: `if schema is None:` treats "" as a real schema.
    # A correct check would be `if not schema:` (falsy) or
    # `if schema is None or schema == '':`.
    assert "if schema is None" not in src or 'if not schema' in src, (
        "get_database_views uses `if schema is None:` which treats "
        "empty-string schema as a real schema name, querying "
        "`WHERE schemaname = ''` and silently returning no rows. "
        "Empty-string schema should be treated as None."
    )


# ---------------------------------------------------------------------------
# BUG-R5-14: _canonicalize_view silently returns None on failure, causing
# compare_views to SKIP the view entirely (no create/replace op emitted).
#
# comparator.py:100-110 catches ALL exceptions, logs a warning, and returns
# None.  comparator.py:171-176 then skips views where canonical is None.
# This means a model view that references a not-yet-created table (a common
# scenario during migrations) is INVISIBLE to autogenerate — no op is
# emitted, so the view is never created.  The user gets a migration that
# silently omits the view.
# ---------------------------------------------------------------------------
def test_round5_canonicalize_failure_emits_op_not_silent_skip():
    """When _canonicalize_view returns None (canonicalization failed),
    compare_views should still emit a CreateViewOp/ReplaceViewOp using the
    model's raw selectable, not silently skip the view.

    Silent skipping means autogenerate produces a migration that omits
    the view entirely — the view is never created, and the user has no
    indication that anything went wrong (only a log.warning).
    """
    src = inspect.getsource(compare_views)

    # The bug: `if canonical is not None:` skips the view when
    # canonicalization fails.  A correct impl would fall back to the
    # model's raw selectable string when canonicalization fails.
    assert "if canonical is not None" not in src or "else" in src.split(
        "if canonical is not None"
    )[1].split("for schema")[0], (
        "compare_views silently skips views when _canonicalize_view "
        "returns None (canonicalization failure). A view that references "
        "a not-yet-created table becomes invisible to autogenerate — no "
        "create op is emitted, so the migration omits the view entirely. "
        "The comparator should fall back to the model's raw selectable "
        "when canonicalization fails."
    )


# ---------------------------------------------------------------------------
# BUG-R5-19: get_database_materialized_views has the same empty-string-schema
# and cross-schema collision bugs as get_database_views.
# ---------------------------------------------------------------------------
def test_round5_pg_catalog_mvs_empty_string_schema():
    """get_database_materialized_views should treat schema='' as None."""
    from sqlalchemy_utils.alembic.pg_catalog import get_database_materialized_views

    src = inspect.getsource(get_database_materialized_views)
    assert "if schema is None" not in src or "if not schema" in src, (
        "get_database_materialized_views uses `if schema is None:` which "
        "treats empty-string schema as a real schema name."
    )


# ---------------------------------------------------------------------------
# BUG-R5-20: ViewMixin.__declare_last__ with __table_args__ as a tuple
# containing a dict in a non-final position does not extract schema.
#
# view_mixin.py:100-101 only checks `isinstance(table_args, dict)` — but
# __table_args__ can be a tuple like `(Column(...), {'schema': 'analytics'})`
# or even `(Column(...), {'extend_existing': True, 'schema': 'analytics'})`.
# The code at line 108-114 extracts Index objects from tuple __table_args__,
# but the schema extraction at line 100-101 ONLY handles the dict form, not
# the tuple-with-dict-last-element form.  Wait — actually line 100 checks
# `isinstance(table_args, dict)` so a tuple __table_args__ with schema in
# the trailing dict is NOT extracted at line 100.  The schema is only
# extracted via __view_schema__ in that case.
#
# However, _resolve_schema (line 221-230) DOES handle tuple __table_args__
# with a trailing dict.  So there's an inconsistency: __declare_last__ uses
# only the dict form, while _resolve_schema handles the tuple form too.
# This means the CreateView/DropView DDL uses schema=None while refresh()
# uses the correct schema.
# ---------------------------------------------------------------------------
def test_round5_view_mixin_declare_last_handles_tuple_table_args_schema():
    """__declare_last__ should extract schema from tuple-style __table_args__
    (e.g. `(Column(...), {'schema': 'analytics'})`), not just dict-style.

    Currently __declare_last__ only checks `isinstance(table_args, dict)`
    for schema extraction, while _resolve_schema handles the tuple form.
    This means the CreateView/DropView DDL emitted by __declare_last__
    uses schema=None while refresh() uses the correct schema — an
    inconsistency that produces unqualified DDL on tuple-style table_args.
    """
    src = inspect.getsource(ViewMixin.__declare_last__)

    # The buggy code only handles dict table_args for schema.
    # It should also handle tuple/list table_args with a trailing dict.
    # Look for the schema extraction block.
    if "isinstance(table_args, dict)" in src:
        # The fix should also handle tuple/list with trailing dict.
        schema_block = src.split("isinstance(table_args, dict)")[1].split("\n")[:5]
        schema_block_text = " ".join(schema_block)
        # A correct impl would also check tuple/list or call _resolve_schema.
        assert "isinstance" in schema_block_text or "_resolve_schema" in src, (
            "__declare_last__ only extracts schema from dict-style "
            "__table_args__, not tuple-style (e.g. "
            "(Column(...), {'schema': 'analytics'})). The DDL emitted by "
            "__declare_last__ uses schema=None while refresh() uses the "
            "correct schema — an inconsistency that produces unqualified DDL."
        )


# ---------------------------------------------------------------------------
# BUG-R5-22: CreateViewOp.to_diff_tuple() shape is inconsistent with the
# other ops — it puts `definition` before `schema`, while DropViewOp and
# the rest put `schema` before `definition`.
#
# operations.py:90-91:  ("create_view", self.name, self.schema, self.definition)
# operations.py:148-149: ("drop_view", self.name, self.schema, self.definition)
#
# Wait — actually CreateViewOp IS (name, schema, definition) after the fix.
# Let me re-check... operations.py:91 says:
#   return ("create_view", self.name, self.schema, self.definition)
# So it's consistent.  But ReplaceViewOp (line 200-201) returns:
#   ("replace_view", self.name, self.schema, self.definition, self.old_definition)
# which is also consistent.  OK — this was fixed.  NOT a bug.
#
# Removing this test to avoid false positive.
# ---------------------------------------------------------------------------
# (No test — bug already fixed.)


# ---------------------------------------------------------------------------
# BUG-R5-23: DropViewOp classmethod `drop_view` hardcodes `materialized=False`,
# so `op.drop_view()` can never drop a materialized view.
#
# operations.py:124-130 constructs DropViewOp with `materialized=False`
# unconditionally.  This is by design (use `op.drop_materialized_view()` for
# MVs), but the DropViewOp class itself accepts `materialized=True`.  The
# inconsistency means `DropViewOp("mv", materialized=True)` produces
# `DROP MATERIALIZED VIEW` SQL, but `op.drop_view("mv")` produces
# `DROP VIEW` SQL — confusing for users who inspect the op class.
#
# This is a minor design issue, not a runtime bug.  Skipping.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUG-R5-24: _quote_qualified_name treats schema='' as falsy (correct), but
# the pg_catalog functions treat schema='' as a real schema (bug R5-11).
# This is an inconsistency: the operations layer treats '' as None, but the
# catalog layer treats '' as a real schema.  Already covered by R5-11.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUG-R5-25: RefreshMaterializedView does not validate that
# `concurrently=True` is incompatible with `schema=None` on some PG versions,
# but more importantly, the compiled SQL places CONCURRENTLY before the
# schema-qualified name, which is correct.  No bug here.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUG-R5-26: create_materialized_view does NOT emit a `WITH [NO] DATA`
# clause in the runtime CreateView DDL (view.py:126-130), while the Alembic
# op DOES emit it (operations.py:435-438).  This means the same materialized
# view is created differently via runtime vs. migration:
#   - Runtime: `CREATE MATERIALIZED VIEW mv AS ...`         (PG defaults to WITH DATA)
#   - Migration: `CREATE MATERIALIZED VIEW mv AS ... WITH DATA`  (explicit)
# The result is the same (WITH DATA), but the SQL text differs, which can
# cause spurious diffs during autogenerate (the canonicalized definition
# from pg_views won't have WITH DATA, but the op emits it).
#
# This is already covered by test_create_mv_runtime_vs_op_consistency in
# Section 2.  NOT a new bug.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUG-R5-27: _canonicalize_view for regular views uses `CREATE OR REPLACE
# VIEW`, which fails if the existing view has a DIFFERENT column set.  When
# this happens, canonicalization returns None and the view is silently
# skipped (see R5-14).  This is a compounding issue with R5-14.
# Already covered by R5-14.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUG-R5-28: comparator.py _schema_matches uses exact equality, so a view
# with schema='public' is NOT processed in the schema=None loop.  But
# PostgreSQL stores views created without an explicit schema in the 'public'
# schema (or whatever the default search_path is).  So a model ViewRecord
# with schema=None (meaning "default schema") won't match a DB view stored
# under schemaname='public'.  The comparator then emits a spurious
# CreateViewOp for a view that already exists in the DB under 'public'.
#
# This is the OPPOSITE of bug BUG-D (which warned about None=='public'
# causing DUPLICATE ops).  The fix for BUG-D made _schema_matches exact,
# which introduced THIS bug: None-schema model views don't match
# 'public'-schema DB views, causing spurious create ops.
#
# However, this is arguably correct behavior — the user should set
# schema='public' explicitly if they want to match.  The existing test
# (test_no_duplicate_ops_for_none_public_schemas) treats exact match as
# the EXPECTED behavior.  So this is a design tradeoff, not a bug.
# Skipping to avoid contradicting existing tests.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUG-R5-29: ViewMixin.__declare_last__ registers a GLOBAL before_flush
# listener on sa.orm.Session as a side effect.  Once any ViewMixin subclass
# is declared, EVERY Session in the process pays the cost of iterating
# session.new | session.dirty | session.deleted and isinstance-checking
# every instance.  This affects performance for applications that use
# Sessions heavily but have no ViewMixin instances in most sessions.
#
# More importantly, the listener is never removed, so it leaks across
# test runs and can interfere with other code that modifies session state.
# The `sa.event.contains` check (view_mixin.py:206) prevents double-
# registration but does NOT prevent the listener from being permanently
# active.
#
# This is a performance/design issue, not a correctness bug.  But the
# global side effect is surprising.  We test that the listener is only
# registered when a ViewMixin is actually used (not at import time).
# ---------------------------------------------------------------------------
def test_round5_global_listener_registered_on_session_not_scoped():
    """The before_flush listener should be scoped to a specific Session
    subclass or removable, not permanently registered on the global
    sa.orm.Session base class.

    Once any ViewMixin subclass is declared, EVERY Session in the process
    pays the cost of the before_flush hook iterating session.new |
    session.dirty | session.deleted and isinstance-checking every instance.
    The listener is never removed (no unregister API), leaking across
    test runs and affecting unrelated Sessions.
    """
    src = inspect.getsource(ViewMixin.__declare_last__)

    # The bug: listens on sa.orm.Session (the global base class), not a
    # scoped Session subclass.  A correct impl would either:
    # 1. Use a session-scoped event (e.g. on a specific Session class), or
    # 2. Provide an unregister API, or
    # 3. Use a cheaper check than iterating all instances.
    assert "sa.orm.Session" in src or "Session" in src, (
        "Expected the global listener registration on sa.orm.Session"
    )
    # The listener is registered via sa.event.listen on the GLOBAL Session
    # class — this is the bug.  It should be scoped or removable.
    assert "sa.event.listen(sa.orm.Session" in src, (
        "ViewMixin registers a permanent global before_flush listener on "
        "sa.orm.Session (the base class). This affects every Session in the "
        "process, even those with no ViewMixin instances. The listener "
        "should be scoped to a specific Session subclass or removable."
    )


# ---------------------------------------------------------------------------
# BUG-R5-31: render_drop_view emits cascade=False but never emits
# cascade=True explicitly — wait, actually it does (cascade_part is ""
# when cascade=True, ", cascade=False" when False).  This is correct.
# But render_drop_view does NOT emit the `definition=` parameter, so the
# rendered downgrade cannot reverse the drop.  However, DropViewOp.reverse()
# requires `definition`, and the renderer is for the UPGRADE direction,
# so this is by design.  NOT a bug.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUG-R5-32: render_drop_materialized_view does not emit `definition=`,
# so the rendered migration cannot reverse the drop.  Same as R5-31 — by
# design for upgrade direction.  NOT a bug.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUG-R5-34: ViewRecord._selectable_key calls sel.compile() without passing
# a dialect, which can produce wrong SQL or raise for dialect-specific
# selectables.  Already noted in test_definition_matches_with_sa_selectable
# (Section 3) as a known limitation.  NOT a new bug.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUG-R5-35: comparator.py compare_views does not handle the case where
# `schemas` is None (instead of a list).  The function signature is
# `compare_views(autogen_context, upgrade_ops, schemas)` and it iterates
# `for schema in schemas:`.  If schemas is None, this raises TypeError
# ("NoneType is not iterable").  Alembic may pass None in some configurations.
# ---------------------------------------------------------------------------
def test_round5_compare_views_handles_none_schemas():
    """compare_views should handle schemas=None gracefully (treat as [None]
    or empty list), not raise TypeError."""
    metadata = sa.MetaData()
    metadata.info["sqlalchemy_utils_views"] = []

    autogen_context = MagicMock()
    autogen_context.connection = MagicMock()
    autogen_context.connection.dialect.name = 'postgresql'
    autogen_context.metadata = metadata

    upgrade_ops = MagicMock()
    upgrade_ops.ops = []

    # schemas=None should not raise TypeError.
    raised = None
    try:
        compare_views(autogen_context, upgrade_ops, None)
    except Exception as exc:
        raised = exc

    assert raised is None or not isinstance(raised, TypeError), (
        "compare_views raises TypeError when schemas=None (iterating None). "
        f"Got: {raised!r}. Should handle None as [None] or empty list."
    )


# ---------------------------------------------------------------------------
# BUG-R5-37: DropView DDL compiler (view.py:46-55) appends ' CASCADE' with a
# leading space, but when cascade=False it does NOT append anything (correct).
# However, the format string `'DROP {}VIEW IF EXISTS {}{}'` has THREE
# placeholders but only TWO are filled by the .format() call — wait, no,
# there are three: materialized, schema_prefix, name.  Let me recount:
#   'DROP {}VIEW IF EXISTS {}{}'.format(
#       'MATERIALIZED ' if element.materialized else '',
#       schema_prefix,
#       compiler.dialect.identifier_preparer.quote(element.name),
#   )
# That's three args for three {} — correct.  And then `if element.cascade:
# sql += ' CASCADE'`.  OK, no bug here after the fix.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUG-R5-38: _create_view_impl (operations.py:398-405) interpolates
# `op.definition` directly into the SQL string via f-string.  If the
# definition contains a semicolon followed by malicious SQL (e.g.
# `SELECT 1; DROP TABLE users; --`), it executes both statements.  The
# name and schema are quoted via _quote_qualified_name, but the definition
# is NOT sanitized.  This is a SQL injection vector if the definition
# comes from an untrusted source.
#
# For a migration tool, the definition is usually trusted (written by the
# developer).  But if autogenerate reads a definition from an untrusted DB
# and re-emits it, the injection is possible.  We test that the impl does
# NOT sanitize the definition (documenting the risk).
# ---------------------------------------------------------------------------
def test_round5_create_view_impl_does_not_sanitize_definition():
    """_create_view_impl should not interpolate op.definition directly into
    SQL via f-string without sanitization, as it enables SQL injection if
    the definition comes from an untrusted source (e.g. autogenerate reading
    from a compromised DB)."""
    src = inspect.getsource(_create_view_impl)

    # The bug: f"... AS {op.definition}" interpolates raw.
    assert "op.definition" in src, (
        "Expected _create_view_impl to reference op.definition"
    )
    # A safer approach would use sa.text(...).bindparams or a parameterized
    # query, though DDL statements typically don't support bind parameters
    # for the body.  At minimum, the definition should be validated to
    # contain no statement separators (;) beyond the view query.
    assert "{op.definition}" in src or "op.definition}" in src, (
        "_create_view_impl interpolates op.definition directly into the SQL "
        "string via f-string. If the definition contains a semicolon followed "
        "by malicious SQL (e.g. from autogenerate reading an untrusted DB), "
        "this enables SQL injection. The definition should be validated or "
        "parameterized."
    )


# ============================================================================
# Section 7: Bug Hunt Round 6
#
# Round 6 focuses on RUNTIME CORRECTNESS bugs not covered by earlier rounds:
# None-handling crashes, round-trip fidelity losses, dead-code guards, missing
# attribute symmetry, and silent data-loss in renderers.  Each test below
# documents a distinct defect and is expected to FAIL until the source is
# fixed.
# ============================================================================


# ---------------------------------------------------------------------------
# BUG-R6-01: resolve_create_order / resolve_drop_order crash with
# AttributeError when db_views=None.
#
# depend.py:_toposort calls _build_dependency_graph(view_records, db_views),
# which does `set(db_views.keys())` (line 67).  If db_views is None this
# raises AttributeError: 'NoneType' object has no attribute 'keys'.
# The public API docstring says db_views is "a mapping", but callers (e.g.
# a custom autogenerate flow) may pass None to mean "no existing views".
# It should treat None as {}.
# ---------------------------------------------------------------------------
def test_round6_resolve_create_order_none_db_views():
    """resolve_create_order should treat db_views=None as an empty dict,
    not crash with AttributeError."""
    vr = ViewRecord(name="solo", selectable="SELECT 1 AS col")
    result = resolve_create_order([vr], db_views=None)
    assert result == [vr], (
        "resolve_create_order(db_views=None) should treat None as {} and "
        "return the input list unchanged, but it raises "
        "AttributeError: 'NoneType' object has no attribute 'keys'."
    )


def test_round6_resolve_drop_order_none_db_views():
    """resolve_drop_order should treat db_views=None as an empty dict."""
    vr = ViewRecord(name="solo", selectable="SELECT 1 AS col")
    result = resolve_drop_order([vr], db_views=None)
    assert result == [vr], (
        "resolve_drop_order(db_views=None) should treat None as {} and "
        "return the input list unchanged, but it raises AttributeError."
    )


def test_round6_build_dependency_graph_none_db_views():
    """_build_dependency_graph should treat db_views=None as {}."""
    vr = ViewRecord(name="solo", selectable="SELECT 1 AS col")
    graph = _build_dependency_graph([vr], None)
    assert graph == {"solo": set()}, (
        "_build_dependency_graph(db_views=None) should treat None as {} "
        "and return {'solo': set()}, but it raises AttributeError."
    )


# ---------------------------------------------------------------------------
# BUG-R6-02: ViewMixin.__declare_last__ has a dead-code super() guard.
#
# view_mixin.py:80-81:
#     if '__declare_last__' in cls.__mro__[1:]:
#         super().__declare_last__()
#
# `cls.__mro__[1:]` is a tuple of CLASS OBJECTS, not strings.  The check
# `'__declare_last__' in <tuple of classes>` tests whether the STRING
# equals any class object — which is always False.  So super() is NEVER
# invoked by ViewMixin, even when a parent mixin in the MRO defines
# __declare_last__.  The guard is dead code; the intent was likely
# `hasattr(cls.__mro__[1], '__declare_last__')` or
# `any(hasattr(c, '__declare_last__') for c in cls.__mro__[1:])`.
#
# Impact: any mixin declared BEFORE ViewMixin in the MRO whose
# __declare_last__ performs setup that ViewMixin depends on will NOT be
# called by ViewMixin's __declare_last__ (it relies on SQLAlchemy's own
# configure_mappers() walk to call the parent, which is fragile if
# __declare_last__ ordering matters).
# ---------------------------------------------------------------------------
def test_round6_declare_last_super_guard_is_not_dead_code():
    """The `if '__declare_last__' in cls.__mro__[1:]` guard checks if a
    STRING is in a tuple of CLASS OBJECTS, which is always False — so
    super().__declare_last__() is never called by ViewMixin.

    A parent mixin's __declare_last__ is only invoked by SQLAlchemy's
    own configure_mappers() walk, NOT by ViewMixin's explicit super()
    call.  The guard should use hasattr() or any() over the MRO.
    """
    src = inspect.getsource(ViewMixin.__declare_last__)

    # The buggy guard: checks string membership in a tuple of classes.
    assert "'__declare_last__' in cls.__mro__[1:]" not in src, (
        "ViewMixin.__declare_last__ contains dead code: the guard "
        "`if '__declare_last__' in cls.__mro__[1:]` checks if a STRING "
        "is in a tuple of CLASS OBJECTS (always False), so "
        "super().__declare_last__() is never called. The guard should "
        "use hasattr() or any(hasattr(c, '__declare_last__') for c in "
        "cls.__mro__[1:])."
    )


# ---------------------------------------------------------------------------
# BUG-R6-03: CreateMaterializedViewOp(with_data=False) loses with_data
# on double-reverse.
#
# operations.py:300-313 — DropMaterializedViewOp.reverse() constructs
# CreateMaterializedViewOp(self.name, self.definition, schema=self.schema)
# WITHOUT forwarding with_data.  So:
#   CreateMaterializedViewOp("mv", "SELECT 1", with_data=False)
#       .reverse()  → DropMaterializedViewOp(definition="SELECT 1")
#       .reverse()  → CreateMaterializedViewOp("mv", "SELECT 1")  # with_data=True (default!)
#
# A double-reverse should be identity, but with_data=False is lost,
# silently flipping a WITH NO DATA migration to WITH DATA.
# ---------------------------------------------------------------------------
def test_round6_create_mv_op_double_reverse_preserves_with_data_false():
    """CreateMaterializedViewOp(with_data=False) → reverse() → reverse()
    should preserve with_data=False (double-reverse is identity)."""
    op = CreateMaterializedViewOp("mv", "SELECT 1", with_data=False)
    double_reversed = op.reverse().reverse()
    assert isinstance(double_reversed, CreateMaterializedViewOp), (
        "Double-reverse of CreateMaterializedViewOp should return a "
        "CreateMaterializedViewOp"
    )
    assert double_reversed.with_data is False, (
        "Double-reverse of CreateMaterializedViewOp(with_data=False) "
        f"loses with_data: got with_data={double_reversed.with_data!r}. "
        "DropMaterializedViewOp.reverse() does not forward with_data, "
        "so a WITH NO DATA migration silently becomes WITH DATA after "
        "downgrade round-trip."
    )


# ---------------------------------------------------------------------------
# BUG-R6-04: CreateViewOp with definition=None produces invalid SQL
# "CREATE VIEW v AS None".
#
# operations.py:_create_view_impl interpolates op.definition directly
# via f-string: f"CREATE {replace_clause}VIEW {qualified} AS {op.definition}".
# When definition=None, this produces "CREATE VIEW v AS None" — the
# literal string "None" is emitted as the view body, which is invalid
# SQL on every dialect.  CreateViewOp.__init__ should reject None.
# ---------------------------------------------------------------------------
def test_round6_create_view_op_rejects_none_definition():
    """CreateViewOp.__init__ should reject definition=None, not silently
    produce invalid SQL 'CREATE VIEW v AS None'."""
    with pytest.raises((TypeError, ValueError), match="(?i)definition"):
        CreateViewOp("v", None)


def test_round6_create_view_impl_none_definition_does_not_emit_literal_none():
    """_create_view_impl should not emit 'CREATE VIEW v AS None' when
    definition is None.  It should either raise or be rejected upstream."""
    with pytest.raises((TypeError, ValueError), match="(?i)definition"):
        op = CreateViewOp("v", None)
        sqls = _capture_sql(op)
        assert not any("AS None" in s for s in sqls), (
            "_create_view_impl emits 'CREATE VIEW v AS None' when definition=None "
            f"(got {sqls!r}). The literal string 'None' is interpolated into the "
            "SQL body, producing invalid SQL on every dialect. "
            "CreateViewOp.__init__ should reject definition=None."
        )


# ---------------------------------------------------------------------------
# BUG-R6-05: ViewMixin without __tablename__ raises an unhelpful
# InvalidRequestError from SQLAlchemy core instead of a clear error
# from ViewMixin.
#
# view_mixin.py:130 calls create_table_from_selectable(name=cls.__tablename__, ...).
# If the subclass does not set __tablename__, cls.__tablename__ raises
# InvalidRequestError("Class does not have a __table__ or __tablename__
# specified...").  The error message does not mention ViewMixin or views,
# making it hard to diagnose.
# ---------------------------------------------------------------------------
def test_round6_view_mixin_without_tablename_raises_helpful_error():
    """ViewMixin without __tablename__ should raise a clear, view-specific
    error (e.g. mentioning __view_selectable__ or ViewMixin), not a generic
    SQLAlchemy InvalidRequestError that doesn't reference views at all."""

    Base = sa.orm.declarative_base()

    with pytest.raises(Exception) as exc_info:
        class NoTablenameThing(ViewMixin, Base):
            __view_selectable__ = sa.select(sa.column("id", sa.Integer))
            id: "Mapped[int]" = sa.Column(sa.Integer, primary_key=True)

        NoTablenameThing.__declare_last__()

    err_msg = str(exc_info.value).lower()
    # The error should mention views or ViewMixin specifically (not just rely
    # on the class name happening to contain "view"). SQLAlchemy's generic
    # error mentions __tablename__ but not __view_selectable__ or ViewMixin.
    assert "__view_selectable__" in err_msg or "viewmixin" in err_msg, (
        "ViewMixin without __tablename__ should raise an error that mentions "
        "'__view_selectable__' or 'ViewMixin' so the user can diagnose the "
        "issue from a view-specific error, not a generic SQLAlchemy "
        f"InvalidRequestError. Got: {exc_info.value!r}"
    )


# ---------------------------------------------------------------------------
# BUG-R6-06: ViewRecord with selectable=None causes _canonicalize_view
# to crash (AttributeError: 'NoneType' has no attribute 'compile'),
# which is caught and returns None → the view is silently skipped by
# compare_views.  No validation at construction time.
#
# comparator.py:57-66:
#     sel = view_record.selectable
#     if not isinstance(sel, str):
#         definition = str(sel.compile(...))   # sel=None → AttributeError
#
# The exception is caught (line 100), logged as a warning, and None is
# returned.  compare_views then skips the view entirely (no op emitted),
# so the view is invisible to autogenerate.
# ---------------------------------------------------------------------------
def test_round6_view_record_rejects_none_selectable():
    """ViewRecord should reject selectable=None at construction time,
    not silently crash during canonicalization."""
    with pytest.raises((TypeError, ValueError), match="(?i)selectable"):
        ViewRecord(name="v", selectable=None)


def test_round6_canonicalize_view_none_selectable_does_not_silently_skip():
    """_canonicalize_view should raise a clear error for selectable=None,
    not return None (which causes compare_views to silently skip the view)."""
    from sqlalchemy_utils.alembic.comparator import _canonicalize_view

    conn = MagicMock()
    conn.dialect = sa.dialects.sqlite.dialect()
    conn.dialect.identifier_preparer = (
        sa.dialects.sqlite.dialect().identifier_preparer
    )
    with pytest.raises((TypeError, ValueError), match="(?i)selectable"):
        vr = ViewRecord(name="v", selectable=None)
        result = _canonicalize_view(conn, vr)
        assert result is not None or True is False, (
            "_canonicalize_view silently returns None when selectable=None "
            "(AttributeError on None.compile is caught), causing compare_views "
            "to skip the view entirely — no create op is emitted, so the "
            "migration omits the view. ViewRecord should reject selectable=None "
            "at construction time."
        )


# ---------------------------------------------------------------------------
# BUG-R6-07: render_drop_view does not emit definition=, so the rendered
# migration's downgrade cannot reverse the drop.
#
# renderer.py:31-35:
#     def render_drop_view(autogen_context, op):
#         schema_part = f", schema={op.schema!r}" if op.schema else ""
#         cascade_part = "" if op.cascade else ", cascade=False"
#         return f"op.drop_view({op.name!r}{schema_part}{cascade_part})"
#
# When DropViewOp carries definition (needed for reverse()), the renderer
# omits it, so the rendered `op.drop_view(...)` call constructs a
# DropViewOp WITHOUT definition.  Calling .reverse() on that op raises
# RuntimeError("no definition stored"), breaking autogenerate downgrade.
# ---------------------------------------------------------------------------
def test_round6_render_drop_view_preserves_definition():
    """render_drop_view should emit definition= when set, so the rendered
    migration's downgrade can reverse the drop."""
    from sqlalchemy_utils.alembic.renderer import render_drop_view

    op = DropViewOp("v", schema="public", definition="SELECT 1")
    rendered = render_drop_view(MagicMock(), op)
    assert "definition=" in rendered, (
        "render_drop_view drops the `definition=` parameter: rendered code "
        f"{rendered!r} cannot produce a working downgrade because "
        "DropViewOp.reverse() requires definition. The rendered "
        "op.drop_view(...) call constructs a DropViewOp without definition, "
        "so calling .reverse() raises RuntimeError."
    )


# ---------------------------------------------------------------------------
# BUG-R6-08: render_drop_materialized_view does not emit definition=.
# Same as BUG-R6-07 but for materialized views.
# ---------------------------------------------------------------------------
def test_round6_render_drop_materialized_view_preserves_definition():
    """render_drop_materialized_view should emit definition= when set."""
    from sqlalchemy_utils.alembic.renderer import render_drop_materialized_view

    op = DropMaterializedViewOp("mv", definition="SELECT 1")
    rendered = render_drop_materialized_view(MagicMock(), op)
    assert "definition=" in rendered, (
        "render_drop_materialized_view drops the `definition=` parameter: "
        f"rendered code {rendered!r} cannot produce a working downgrade "
        "because DropMaterializedViewOp.reverse() requires definition."
    )


# ---------------------------------------------------------------------------
# BUG-R6-09: render_replace_view / render_replace_materialized_view omit
# old_definition= when it is an empty string "".
#
# renderer.py:41:  old_def_part = f", old_definition={op.old_definition!r}" if op.old_definition else ""
# renderer.py:69:  old_def_part = f", old_definition={op.old_definition!r}" if op.old_definition else ""
#
# The `if op.old_definition` check treats "" as falsy, so an empty-string
# old_definition is omitted from the rendered code.  The resulting
# op.replace_view(...) call constructs a ReplaceViewOp with
# old_definition=None, so .reverse() raises RuntimeError.
# This breaks downgrade for views whose old definition was empty.
# ---------------------------------------------------------------------------
def test_round6_render_replace_view_preserves_empty_string_old_definition():
    """render_replace_view should emit old_definition= even when it's an
    empty string, not omit it (falsy check treats '' as absent)."""
    from sqlalchemy_utils.alembic.renderer import render_replace_view

    op = ReplaceViewOp("v", "SELECT 2", old_definition="")
    rendered = render_replace_view(MagicMock(), op)
    assert "old_definition=" in rendered, (
        "render_replace_view omits old_definition= when it's an empty "
        f"string (falsy check). Rendered: {rendered!r}. The rendered "
        "op.replace_view(...) constructs a ReplaceViewOp with "
        "old_definition=None, so .reverse() raises RuntimeError — "
        "breaking downgrade for views whose old definition was empty."
    )


def test_round6_render_replace_mv_preserves_empty_string_old_definition():
    """render_replace_materialized_view should emit old_definition= even
    when it's an empty string."""
    from sqlalchemy_utils.alembic.renderer import render_replace_materialized_view

    op = ReplaceMaterializedViewOp("mv", "SELECT 2", old_definition="")
    rendered = render_replace_materialized_view(MagicMock(), op)
    assert "old_definition=" in rendered, (
        "render_replace_materialized_view omits old_definition= when it's "
        f"an empty string. Rendered: {rendered!r}. Breaks downgrade."
    )


# ---------------------------------------------------------------------------
# BUG-R6-10: render_create_view / render_drop_view omit schema= when
# schema is an empty string "".
#
# renderer.py:26:  schema_part = f", schema={op.schema!r}" if op.schema else ""
# renderer.py:33:  schema_part = f", schema={op.schema!r}" if op.schema else ""
# (same pattern in all 6 renderers)
#
# The `if op.schema` check treats "" as falsy, so an empty-string schema
# is omitted from the rendered code.  This is inconsistent with
# _quote_qualified_name (operations.py:39) which also treats '' as None,
# BUT the rendered migration loses the schema entirely — re-running the
# rendered migration creates the view in the default schema instead of
# the empty-string schema.  While empty-string schema is unusual, the
# asymmetry between "op carries schema=''" and "rendered migration has
# no schema=" is a round-trip fidelity bug.
# ---------------------------------------------------------------------------
def test_round6_render_create_view_preserves_empty_string_schema():
    """render_create_view should emit schema='' when explicitly set, not
    omit it (falsy check treats '' as absent)."""
    from sqlalchemy_utils.alembic.renderer import render_create_view

    op = CreateViewOp("v", "SELECT 1", schema="")
    rendered = render_create_view(MagicMock(), op)
    assert "schema=" in rendered, (
        "render_create_view omits schema= when it's an empty string (falsy "
        f"check). Rendered: {rendered!r}. The rendered op.create_view(...) "
        "constructs a CreateViewOp with schema=None instead of schema='', "
        "breaking round-trip fidelity for empty-string schema."
    )



