"""
TDD RED-phase tests for interface-audit findings.

These tests expose 2 blockers (B1, B2) and 5 should-fix issues (S1-S5)
identified in the interface audit. They are *expected to fail* against the
current (buggy) source so we can drive the fixes.

Run with:

    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
        tests/test_interface_audit.py -v
"""
from __future__ import annotations

import inspect
import logging
from unittest.mock import MagicMock

import pytest
import sqlalchemy as sa
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from sqlalchemy.orm import declarative_base, Mapped, mapped_column

from sqlalchemy_utils.alembic.operations import (
    CreateMaterializedViewOp,
    CreateViewOp,
    DropMaterializedViewOp,
    DropViewOp,
    ReplaceMaterializedViewOp,
    ReplaceViewOp,
)
from sqlalchemy_utils.view import (
    CreateView,
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
    from unittest.mock import patch

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
