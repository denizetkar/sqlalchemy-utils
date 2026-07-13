"""Tests for the SQLAlchemy-Utils Alembic view integration.

Covers ViewRecord, the 6 view Operations, renderers, comparators,
dependency resolution, pg_catalog helpers, autogenerate integration,
public API, import safety, DDL formatting, schema resolution and the
ViewMixin integration.
"""
from __future__ import annotations

import contextlib
import inspect
import logging
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import sqlalchemy as sa
from alembic import command, config
from alembic.autogenerate.api import AutogenContext
from alembic.operations import Operations, ops as alembic_ops
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Column, Integer, select
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

from sqlalchemy_utils import (
    create_materialized_view,
    create_view,
)
from sqlalchemy_utils.alembic.comparator import (
    _build_create_sql,
    _canonicalize_all_views,
    _reorder_cross_type_drops_before_creates,
    _safe_resolve,
    compare_views,
    register_view_comparator,
)
import sqlalchemy_utils.alembic.comparator as comparator_module
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
    RefreshMaterializedViewOp,
    ReplaceMaterializedViewOp,
    ReplaceViewOp,
)
from sqlalchemy_utils.alembic.pg_catalog import (
    get_database_materialized_views,
    get_database_views,
    get_dependent_views,
)
from sqlalchemy_utils.alembic.renderer import (
    render_create_materialized_view,
    render_create_view,
    render_drop_materialized_view,
    render_drop_view,
    render_refresh_materialized_view,
    render_replace_materialized_view,
    render_replace_view,
)
from sqlalchemy_utils.view_record import ViewRecord
from sqlalchemy_utils.view import (
    CreateView,
    DropView,
    RefreshMaterializedView,
    refresh_materialized_view,
)
from sqlalchemy_utils.view_mixin import ViewMixin


@pytest.fixture(autouse=True)
def _reset_registered():
    """Reset comparator _registered flag between tests."""
    saved = comparator_module._registered
    yield
    comparator_module._registered = saved


# ===========================================================================
# Shared helpers
# ===========================================================================

def _make_poisoned_connection(view_name, call_log):
    """Mock PG connection: ``CREATE VIEW <view_name>`` fails and ``SELECT 1``
    probe raises a poisoned-transaction error. All SQL is logged to *call_log*."""
    def _execute(stmt):
        text = getattr(stmt, "text", str(stmt))
        call_log.append(text)
        stripped = text.strip()

        # view_a CREATE fails — triggers the inner except block.
        if stripped.startswith(f"CREATE VIEW {view_name}"):
            raise sa.exc.ProgrammingError(
                statement=text,
                params=None,
                orig=Exception("relation does not exist"),
            )

        # The probe after the failure: poisoned transaction.
        if stripped == "SELECT 1":
            raise sa.exc.SQLAlchemyError(
                "current transaction is aborted, "
                "commands ignored until end of transaction block"
            )

        # All other statements succeed.
        return MagicMock()

    connection = MagicMock()
    connection.dialect = sa.dialects.postgresql.dialect()
    connection.execute.side_effect = _execute
    return connection


def _capture_sql(op_instance) -> list[str]:
    """Invoke *op_instance* against in-memory SQLite and capture SQL strings."""
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


def _compile_ddl(ddl_element, dialect=None) -> str:
    """Compile a DDLElement to a SQL string using the given dialect."""
    if dialect is None:
        engine = sa.create_engine("sqlite:///:memory:")
        dialect = engine.dialect
    return str(ddl_element.compile(dialect=dialect))


def _ddl_sql_for_metadata(metadata: sa.MetaData, dialect=None) -> list[str]:
    """Compile after_create DDL listeners to SQL strings without executing."""
    if dialect is None:
        dialect = sa.dialects.sqlite.dialect()
    statements: list[str] = []
    for listener in metadata.dispatch.after_create:
        compile_fn = getattr(listener, "compile", None)
        if compile_fn is not None:
            try:
                statements.append(str(compile_fn(dialect=dialect)))
            except Exception:
                pass
    return statements


_CMP_TEST_VIEW_NAMES = [
    "cmp_test_view", "cmp_test_mv", "cmp_test_view2",
    "cmp_test_changed", "cmp_test_mv_changed", "cmp_test_view_bad",
]


def _drop_views(connection, names) -> None:
    for view_name in names:
        try:
            connection.execute(
                sa.text(f"DROP MATERIALIZED VIEW IF EXISTS {view_name} CASCADE")
            )
        except sa.exc.ProgrammingError:
            connection.rollback()
        try:
            connection.execute(sa.text(f"DROP VIEW IF EXISTS {view_name} CASCADE"))
        except sa.exc.ProgrammingError:
            connection.rollback()
    connection.commit()


def _find_view_listener(metadata: sa.MetaData, materialized: bool | None = None):
    """Find a CreateView DDL element registered on *metadata*.

    *materialized*=None returns any CreateView listener; True/False
    filters on the listener's ``materialized`` attribute.
    """
    listeners = list(metadata.dispatch.after_create)

    def _listener_materialized(listener) -> bool | None:
        if isinstance(listener, CreateView):
            return listener.materialized
        return getattr(listener, "materialized", None)

    found = [
        listener
        for listener in listeners
        if isinstance(listener, CreateView)
        and (materialized is None or _listener_materialized(listener) is materialized)
    ]
    return found[0] if found else None


def _make_autogen_context() -> AutogenContext:
    """Create a minimal mock AutogenContext for renderer tests."""
    ctx = MagicMock(spec=AutogenContext)
    ctx.imports = set()
    return ctx


def _run_comparator(connection, metadata, schemas=None):
    """Run compare_views and return the generated UpgradeOps."""
    migration_ctx = MigrationContext.configure(connection)
    autogen_context = AutogenContext(migration_ctx, metadata=metadata)
    upgrade_ops = alembic_ops.UpgradeOps([])
    if schemas is None:
        schemas = [None]
    compare_views(autogen_context, upgrade_ops, schemas)
    return upgrade_ops


def _make_mock_autogen_context(model_views=None, existing_ops=None):
    """Build a mock AutogenContext + UpgradeOps for compare_views tests.

    *model_views* populates ``metadata.info['sqlalchemy_utils_views']``;
    *existing_ops* pre-populates ``upgrade_ops.ops`` (non-view ops that
    Alembic's own comparators may have already appended).
    """
    metadata = MagicMock()
    metadata.info = {"sqlalchemy_utils_views": model_views or []}

    autogen_context = MagicMock()
    autogen_context.connection = MagicMock()
    autogen_context.connection.dialect.name = "postgresql"
    autogen_context.metadata = metadata

    upgrade_ops = MagicMock()
    upgrade_ops.ops = list(existing_ops or [])
    return autogen_context, upgrade_ops


def _patch_comparator(
    db_views=None, db_mvs=None, canonical_return=None, dependent_views=None,
):
    """Return an ``ExitStack`` context manager applying all compare_views patches.

    Patches ``get_database_views``, ``get_database_materialized_views``,
    ``_canonicalize_all_views``, ``get_dependent_views``, and ``log`` on the
    comparator module. Yields the ``log`` mock so callers can assert on
    log calls. Use as ``with _patch_comparator(...) as mock_log:``.
    """
    if db_views is None:
        db_views = {}
    if db_mvs is None:
        db_mvs = {}
    if canonical_return is None:
        canonical_return = ({}, {}, set())
    if dependent_views is None:
        dependent_views = {}

    @contextlib.contextmanager
    def _ctx():
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(
                comparator_module, "get_database_views",
                return_value=db_views,
            ))
            stack.enter_context(patch.object(
                comparator_module, "get_database_materialized_views",
                return_value=db_mvs,
            ))
            stack.enter_context(patch.object(
                comparator_module, "_canonicalize_all_views",
                return_value=canonical_return,
            ))
            stack.enter_context(patch.object(
                comparator_module, "get_dependent_views",
                return_value=dependent_views,
            ))
            mock_log = stack.enter_context(
                patch.object(comparator_module, "log")
            )
            yield mock_log

    return _ctx()


# ===========================================================================
# ViewRecord
# ===========================================================================

class TestViewRecordCreation:

    def test_create_with_minimum_fields(self):
        record = ViewRecord(name="test_view", selectable="SELECT 1")
        assert record.name == "test_view"
        assert record.selectable == "SELECT 1"
        assert record.schema is None
        assert record.materialized is False
        assert record.replace is False
        assert record.cascade_on_drop is True

    def test_create_with_all_fields(self):
        record = ViewRecord(
            name="test_view",
            selectable="SELECT 1",
            schema="public",
            materialized=True,
            replace=True,
            cascade_on_drop=False,
        )
        assert record.name == "test_view"
        assert record.selectable == "SELECT 1"
        assert record.schema == "public"
        assert record.materialized is True
        assert record.replace is True
        assert record.cascade_on_drop is False

    def test_create_with_empty_string_schema_normalizes_to_none(self):
        """Empty-string schema is falsy and must be normalized to None.

        Without normalization, ``""`` is treated as "no schema" by
        ``_quote_qualified_name`` (view created in ``current_schema()``),
        but ``"" == None`` returns ``False``, causing the view to be
        dropped as a false ``DropViewOp``.
        """
        record = ViewRecord(name="test_view", selectable="SELECT 1", schema="")
        assert record.schema is None
        assert record == ViewRecord(name="test_view", selectable="SELECT 1", schema=None)
        assert hash(record) == hash(
            ViewRecord(name="test_view", selectable="SELECT 1", schema=None)
        )

    def test_rejects_none_selectable(self):
        """ViewRecord rejects selectable=None at construction time."""
        with pytest.raises((TypeError, ValueError), match="(?i)selectable"):
            ViewRecord(name="v", selectable=None)


class TestViewRecordEquality:

    def test_equal_records_with_same_fields(self):
        assert (
            ViewRecord(name="test_view", selectable="SELECT 1")
            == ViewRecord(name="test_view", selectable="SELECT 1")
        )

    def test_not_equal_with_different_name(self):
        assert ViewRecord(name="view1", selectable="SELECT 1") != ViewRecord(
            name="view2", selectable="SELECT 1"
        )

    def test_not_equal_with_different_schema(self):
        assert ViewRecord(
            name="v", selectable="SELECT 1", schema="schema1"
        ) != ViewRecord(name="v", selectable="SELECT 1", schema="schema2")

    def test_not_equal_with_different_materialized(self):
        assert ViewRecord(
            name="v", selectable="SELECT 1", materialized=True
        ) != ViewRecord(name="v", selectable="SELECT 1", materialized=False)

    def test_equality_without_optional_fields(self):
        record1 = ViewRecord(name="test_view", selectable="SELECT 1")
        record2 = ViewRecord(name="test_view", selectable="SELECT 1", schema=None)
        record3 = ViewRecord(
            name="test_view", selectable="SELECT 1", materialized=False
        )
        assert record1 == record2
        assert record1 == record3


class TestViewRecordHashing:

    def test_hash_consistent_with_equality(self):
        record1 = ViewRecord(name="test_view", selectable="SELECT 1")
        record2 = ViewRecord(name="test_view", selectable="SELECT 1")
        assert hash(record1) == hash(record2)
        assert record1 == record2

    def test_different_records_have_different_hashes(self):
        record1 = ViewRecord(name="view1", selectable="SELECT 1")
        record2 = ViewRecord(name="view2", selectable="SELECT 1")
        assert hash(record1) != hash(record2)


# ===========================================================================
# pg_catalog
# ===========================================================================

@pytest.mark.parametrize(
    "fetch_fn",
    [get_database_views, get_database_materialized_views],
    ids=["views", "materialized_views"],
)
class TestGetDatabaseViews:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_query_empty_database(self, connection, fetch_fn):
        assert fetch_fn(connection) == {}


class TestPgCatalogRejectsNonPostgresDialect:
    """All public pg_catalog functions must fail fast with a clear error
    when called with a non-PostgreSQL connection.

    These are public API (in ``__all__``) and should not silently emit
    broken SQL against a dialect that lacks ``pg_catalog``.
    """

    @pytest.mark.parametrize(
        "fn,fn_args,fn_kwargs",
        [
            (get_database_views, (), {}),
            (get_database_materialized_views, (), {}),
            (get_dependent_views, ("some_view",), {}),
        ],
        ids=["get_database_views", "get_database_materialized_views", "get_dependent_views"],
    )
    def test_non_pg_dialect_raises_not_implemented(self, fn, fn_args, fn_kwargs):
        engine = sa.create_engine("sqlite:///:memory:")
        with engine.connect() as connection:
            with pytest.raises(NotImplementedError, match="PostgreSQL"):
                fn(connection, *fn_args, **fn_kwargs)


# ===========================================================================
# Operations
# ===========================================================================

class TestCreateViewOp:

    def test_instantiation(self):
        op = CreateViewOp("v1", "SELECT 1")
        assert op.name == "v1"
        assert op.definition == "SELECT 1"
        assert op.schema is None

    def test_reverse_returns_drop_view(self):
        op = CreateViewOp("v1", "SELECT 1")
        rev = op.reverse()
        assert isinstance(rev, DropViewOp)
        assert rev.name == "v1"
        assert rev.definition == "SELECT 1"

    @pytest.mark.parametrize(
        "cascade_on_drop, expected_cascade",
        [(False, False), (None, True)],
        ids=["explicit_false", "default_true"],
    )
    def test_reverse_propagates_cascade_on_drop(
        self, cascade_on_drop, expected_cascade
    ):
        kwargs = {"cascade_on_drop": cascade_on_drop} if cascade_on_drop is not None else {}
        op = CreateViewOp("v1", "SELECT 1", **kwargs)
        rev = op.reverse()
        assert isinstance(rev, DropViewOp)
        assert rev.cascade is expected_cascade, f"got {rev.cascade!r}"

    def test_sql_without_replace(self):
        op = CreateViewOp("v1", "SELECT 1")
        sqls = _capture_sql(op)
        assert sqls == ["CREATE VIEW v1 AS SELECT 1"]

    def test_sql_with_schema(self):
        op = CreateViewOp("v1", "SELECT 1", schema="public")
        sqls = _capture_sql(op)
        assert sqls == ["CREATE VIEW public.v1 AS SELECT 1"]

    def test_replace_view_classmethod_invokes_replace_view_op(self):
        operations = MagicMock()
        operations.invoke.return_value = None
        ReplaceViewOp.replace_view(
            operations, "test_view", "SELECT 2", old_definition="SELECT 1"
        )
        operations.invoke.assert_called_once()
        invoked_op = operations.invoke.call_args[0][0]
        assert isinstance(invoked_op, ReplaceViewOp)
        assert invoked_op.definition == "SELECT 2"
        assert invoked_op.old_definition == "SELECT 1"


class TestDropViewOp:

    def test_instantiation(self):
        op = DropViewOp("v1", cascade=True)
        assert op.name == "v1"
        assert op.cascade is True

    def test_drop_view_rejects_materialized_kwarg(self):
        engine = sa.create_engine("sqlite:///:memory:")
        conn = engine.connect()
        ctx = MigrationContext.configure(conn)
        operations = Operations(ctx)
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            DropViewOp.drop_view(operations, "test_view", materialized=True)

    def test_reverse_returns_create_view(self):
        op = DropViewOp("v1", definition="SELECT 1")
        rev = op.reverse()
        assert isinstance(rev, CreateViewOp)
        assert rev.name == "v1"
        assert rev.definition == "SELECT 1"

    def test_reverse_without_definition_raises(self):
        op = DropViewOp("v1")
        with pytest.raises(NotImplementedError, match="no definition stored"):
            op.reverse()

    def test_sql_cascade(self):
        op = DropViewOp("v1", cascade=True)
        sqls = _capture_sql(op)
        assert sqls == ["DROP VIEW IF EXISTS v1 CASCADE"]

    def test_sql_no_cascade(self):
        op = DropViewOp("v1", cascade=False)
        sqls = _capture_sql(op)
        assert sqls == ["DROP VIEW IF EXISTS v1"]

    def test_sql_with_schema(self):
        op = DropViewOp("v1", schema="myschema")
        sqls = _capture_sql(op)
        assert sqls == ["DROP VIEW IF EXISTS myschema.v1 CASCADE"]


class TestReplaceViewOp:

    def test_instantiation(self):
        op = ReplaceViewOp("v1", "SELECT 2")
        assert op.name == "v1"
        assert op.definition == "SELECT 2"
        assert op.old_definition is None

    def test_reverse_returns_replace_view_with_old_def(self):
        op = ReplaceViewOp("v1", "SELECT 2", old_definition="SELECT 1")
        rev = op.reverse()
        assert isinstance(rev, ReplaceViewOp)
        assert rev.definition == "SELECT 1"

    def test_reverse_without_old_def_raises(self):
        op = ReplaceViewOp("v1", "SELECT 2")
        with pytest.raises(NotImplementedError, match="no old_definition stored"):
            op.reverse()

    def test_sql(self):
        op = ReplaceViewOp("v1", "SELECT 2")
        sqls = _capture_sql(op)
        assert len(sqls) == 2
        assert sqls[0] == "DROP VIEW IF EXISTS v1 CASCADE"
        assert sqls[1] == "CREATE VIEW v1 AS SELECT 2"

    @pytest.mark.parametrize(
        "cascade_value, expected",
        [
            pytest.param(False, False, id="cascade_false"),
            pytest.param(None, True, id="cascade_default"),
        ],
    )
    def test_replace_view_cascade_propagates_to_reverse(self, cascade_value, expected):
        """ReplaceViewOp.reverse() must propagate cascade to the reversed op."""
        kwargs = {"cascade": cascade_value} if cascade_value is not None else {}
        op = ReplaceViewOp(
            "v", "SELECT 2", old_definition="SELECT 1", **kwargs
        )
        rev = op.reverse()
        assert isinstance(rev, ReplaceViewOp)
        assert rev.cascade is expected, f"got {rev.cascade!r}"


class TestCreateMaterializedViewOp:

    def test_instantiation(self):
        op = CreateMaterializedViewOp("mv1", "SELECT 1")
        assert op.name == "mv1"
        assert op.with_data is False

    def test_reverse_returns_drop_mv(self):
        op = CreateMaterializedViewOp("mv1", "SELECT 1")
        rev = op.reverse()
        assert isinstance(rev, DropMaterializedViewOp)
        assert rev.name == "mv1"
        assert rev.definition == "SELECT 1"

    def test_sql_with_no_data(self):
        op = CreateMaterializedViewOp("mv1", "SELECT 1", with_data=False)
        sqls = _capture_sql(op)
        assert sqls == ["CREATE MATERIALIZED VIEW mv1 AS SELECT 1 WITH NO DATA"]

    def test_sql_with_data(self):
        op = CreateMaterializedViewOp("mv1", "SELECT 1", with_data=True)
        sqls = _capture_sql(op)
        assert sqls == ["CREATE MATERIALIZED VIEW mv1 AS SELECT 1 WITH DATA"]

    def test_sql_with_schema(self):
        op = CreateMaterializedViewOp("mv1", "SELECT 1", schema="analytics")
        sqls = _capture_sql(op)
        assert sqls == [
            "CREATE MATERIALIZED VIEW analytics.mv1 AS SELECT 1 WITH NO DATA"
        ]


class TestDropMaterializedViewOp:

    def test_instantiation(self):
        op = DropMaterializedViewOp("mv1", cascade=False)
        assert op.name == "mv1"
        assert op.cascade is False

    def test_reverse_returns_create_mv(self):
        op = DropMaterializedViewOp("mv1", definition="SELECT 1")
        rev = op.reverse()
        assert isinstance(rev, CreateMaterializedViewOp)
        assert rev.name == "mv1"
        assert rev.definition == "SELECT 1"

    def test_reverse_without_definition_raises(self):
        op = DropMaterializedViewOp("mv1")
        with pytest.raises(NotImplementedError, match="no definition stored"):
            op.reverse()

    def test_sql_cascade(self):
        op = DropMaterializedViewOp("mv1", cascade=True)
        sqls = _capture_sql(op)
        assert sqls == ["DROP MATERIALIZED VIEW IF EXISTS mv1 CASCADE"]

    def test_sql_no_cascade(self):
        op = DropMaterializedViewOp("mv1", cascade=False)
        sqls = _capture_sql(op)
        assert sqls == ["DROP MATERIALIZED VIEW IF EXISTS mv1"]


class TestReplaceMaterializedViewOp:

    def test_instantiation(self):
        op = ReplaceMaterializedViewOp("mv1", "SELECT 2")
        assert op.name == "mv1"
        assert op.definition == "SELECT 2"
        assert op.old_definition is None
        assert op.with_data is False

    def test_reverse_returns_replace_mv_with_old_def(self):
        op = ReplaceMaterializedViewOp(
            "mv1", "SELECT 2", old_definition="SELECT 1"
        )
        rev = op.reverse()
        assert isinstance(rev, ReplaceMaterializedViewOp)
        assert rev.definition == "SELECT 1"

    def test_reverse_without_old_def_raises(self):
        op = ReplaceMaterializedViewOp("mv1", "SELECT 2")
        with pytest.raises(NotImplementedError, match="no old_definition stored"):
            op.reverse()

    def test_sql_emits_drop_then_create(self):
        op = ReplaceMaterializedViewOp("mv1", "SELECT 2", with_data=True)
        sqls = _capture_sql(op)
        assert len(sqls) == 2
        assert sqls[0] == "DROP MATERIALIZED VIEW IF EXISTS mv1 CASCADE"
        assert sqls[1] == "CREATE MATERIALIZED VIEW mv1 AS SELECT 2 WITH DATA"

    def test_sql_with_no_data(self):
        op = ReplaceMaterializedViewOp("mv1", "SELECT 2", with_data=False)
        sqls = _capture_sql(op)
        assert sqls[1] == "CREATE MATERIALIZED VIEW mv1 AS SELECT 2 WITH NO DATA"

    def test_sql_with_schema(self):
        op = ReplaceMaterializedViewOp(
            "mv1", "SELECT 2", schema="analytics", with_data=True
        )
        sqls = _capture_sql(op)
        assert sqls[0] == "DROP MATERIALIZED VIEW IF EXISTS analytics.mv1 CASCADE"
        assert sqls[1] == (
            "CREATE MATERIALIZED VIEW analytics.mv1 AS SELECT 2 WITH DATA"
        )


class TestReplaceCascadeKwarg:

    @pytest.mark.parametrize(
        "op_class, view_name, drop_keyword, create_suffix",
        [
            (ReplaceViewOp, "v", "DROP VIEW IF EXISTS", ""),
            (ReplaceMaterializedViewOp, "mv", "DROP MATERIALIZED VIEW IF EXISTS", " WITH NO DATA"),
        ],
        ids=["view", "materialized_view"],
    )
    @pytest.mark.parametrize(
        "cascade_kwarg, expected_cascade_attr, drop_contains_cascade",
        [
            (None, True, True),
            (False, False, False),
        ],
        ids=["default_true", "explicit_false"],
    )
    def test_replace_cascade_kwarg(
        self,
        op_class,
        view_name,
        drop_keyword,
        create_suffix,
        cascade_kwarg,
        expected_cascade_attr,
        drop_contains_cascade,
    ):
        kwargs = {"cascade": cascade_kwarg} if cascade_kwarg is not None else {}
        op = op_class(view_name, "SELECT 1", **kwargs)
        assert op.cascade is expected_cascade_attr, f"got {op.cascade!r}"

        sqls = _capture_sql(op)
        assert len(sqls) == 2

        if drop_contains_cascade:
            expected_drop_sql = f"{drop_keyword} {view_name} CASCADE"
        else:
            expected_drop_sql = f"{drop_keyword} {view_name}"
        assert sqls[0] == expected_drop_sql, f"got {sqls[0]!r}"

        if op_class is ReplaceViewOp:
            expected_create = f"CREATE VIEW {view_name} AS SELECT 1"
        else:
            expected_create = f"CREATE MATERIALIZED VIEW {view_name} AS SELECT 1{create_suffix}"
        assert sqls[1] == expected_create


# ---------------------------------------------------------------------------
# Operations: keyword-only params (parametrized)
# ---------------------------------------------------------------------------

class TestOpKeywordOnlyParams:

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
    def test_classmethod_keyword_only_schema(self, method_name, op_class):
        cls_method = getattr(op_class, method_name)

        class FakeOperations:
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

        with pytest.raises(TypeError):
            cls_method(fake_ops, *positional_args)


# ---------------------------------------------------------------------------
# Operations: validation
# ---------------------------------------------------------------------------

class TestOpValidation:

    def test_create_view_rejects_none_definition(self):
        with pytest.raises((TypeError, ValueError), match="(?i)definition"):
            CreateViewOp("v", None)

    @pytest.mark.parametrize(
        "op_class,bad_name,extra_kwargs",
        [
            (CreateViewOp, "", {"definition": "SELECT 1"}),
            (CreateViewOp, None, {"definition": "SELECT 1"}),
            (DropViewOp, "", {}),
            (DropViewOp, None, {}),
            (ReplaceViewOp, "", {"definition": "SELECT 1"}),
            (ReplaceViewOp, None, {"definition": "SELECT 1"}),
            (CreateMaterializedViewOp, "", {"definition": "SELECT 1"}),
            (CreateMaterializedViewOp, None, {"definition": "SELECT 1"}),
            (DropMaterializedViewOp, "", {}),
            (DropMaterializedViewOp, None, {}),
            (ReplaceMaterializedViewOp, "", {"definition": "SELECT 1"}),
            (ReplaceMaterializedViewOp, None, {"definition": "SELECT 1"}),
            (RefreshMaterializedViewOp, "", {}),
            (RefreshMaterializedViewOp, None, {}),
        ],
        ids=[
            "CreateViewOp-empty", "CreateViewOp-none",
            "DropViewOp-empty", "DropViewOp-none",
            "ReplaceViewOp-empty", "ReplaceViewOp-none",
            "CreateMaterializedViewOp-empty", "CreateMaterializedViewOp-none",
            "DropMaterializedViewOp-empty", "DropMaterializedViewOp-none",
            "ReplaceMaterializedViewOp-empty", "ReplaceMaterializedViewOp-none",
            "RefreshMaterializedViewOp-empty", "RefreshMaterializedViewOp-none",
        ],
    )
    def test_op_rejects_empty_or_none_name(self, op_class, bad_name, extra_kwargs):
        """Every view op __init__ must reject None and empty-string name."""
        with pytest.raises(TypeError, match="(?i)name"):
            op_class(bad_name, **extra_kwargs)


# ===========================================================================
# Renderer
# ===========================================================================

@pytest.mark.parametrize(
    "renderer, op, expected_substring",
    [
        (
            render_replace_view,
            ReplaceViewOp("v", "SELECT 2", old_definition=""),
            "old_definition=",
        ),
        (
            render_replace_materialized_view,
            ReplaceMaterializedViewOp("mv", "SELECT 2", old_definition=""),
            "old_definition=",
        ),
    ],
    ids=["replace_view_old_def", "replace_mv_old_def"],
)
def test_renderer_preserves_falsy_non_none_values(
    renderer, op, expected_substring
):
    rendered = renderer(_make_autogen_context(), op)
    assert expected_substring in rendered


@pytest.mark.parametrize(
    "op_class,renderer",
    [
        (CreateViewOp, render_create_view),
        (CreateMaterializedViewOp, render_create_materialized_view),
    ],
    ids=["create_view", "create_materialized_view"],
)
class TestRendererCascadeOnDrop:

    def test_renders_cascade_on_drop_false(self, op_class, renderer):
        """``cascade_on_drop=False`` must be rendered so the autogenerated
        downgrade (reverse → DropViewOp/DropMaterializedViewOp) honors the
        non-cascading drop.

        Without rendering ``cascade_on_drop=False``, the rendered
        ``op.create_view(...)`` / ``op.create_materialized_view(...)``
        defaults to ``cascade_on_drop=True``, so the autogenerated downgrade
        incorrectly uses ``DROP ... CASCADE``.
        """
        op = op_class("v", "SELECT 1", cascade_on_drop=False)
        rendered = renderer(_make_autogen_context(), op)
        assert "cascade_on_drop=False" in rendered, f"got: {rendered!r}"

    def test_omits_cascade_on_drop_when_default_true(self, op_class, renderer):
        """``cascade_on_drop=True`` (the default) is NOT rendered."""
        op = op_class("v", "SELECT 1")
        rendered = renderer(_make_autogen_context(), op)
        assert "cascade_on_drop=" not in rendered, f"got: {rendered!r}"


class TestRendererCreateView:

    def test_produces_valid_python(self):
        op = CreateViewOp("my_view", "SELECT 1")
        result = render_create_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_schema_omitted_when_none(self):
        op = CreateViewOp("my_view", "SELECT 1", schema=None)
        result = render_create_view(_make_autogen_context(), op)
        assert "schema=" not in result

    def test_schema_included_when_provided(self):
        op = CreateViewOp("my_view", "SELECT 1", schema="public")
        result = render_create_view(_make_autogen_context(), op)
        assert "schema='public'" in result


class TestRendererDropView:

    def test_produces_valid_python(self):
        op = DropViewOp("my_view")
        result = render_drop_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_schema_omitted_when_none(self):
        op = DropViewOp("my_view", schema=None)
        result = render_drop_view(_make_autogen_context(), op)
        assert "schema=" not in result

    def test_schema_included_when_provided(self):
        op = DropViewOp("my_view", schema="analytics")
        result = render_drop_view(_make_autogen_context(), op)
        assert "schema='analytics'" in result

    def test_renders_definition(self):
        op = DropViewOp("v", schema="public", definition="SELECT 1")
        rendered = render_drop_view(_make_autogen_context(), op)
        assert "definition=" in rendered

    def test_omits_default_cascade(self):
        op = DropViewOp("v", cascade=True)
        result = render_drop_view(_make_autogen_context(), op)
        assert "cascade=" not in result

        op_no_cascade = DropViewOp("v", cascade=False)
        result_no = render_drop_view(_make_autogen_context(), op_no_cascade)
        assert "cascade=False" in result_no


class TestRendererReplaceView:

    def test_produces_valid_python(self):
        op = ReplaceViewOp("my_view", "SELECT 2")
        result = render_replace_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_schema_included_when_provided(self):
        op = ReplaceViewOp("my_view", "SELECT 2", schema="public")
        result = render_replace_view(_make_autogen_context(), op)
        assert "schema='public'" in result

    def test_renders_old_definition(self):
        op = ReplaceViewOp("v_repl", "SELECT 2", schema="public", old_definition="SELECT 1")
        rendered = render_replace_view(_make_autogen_context(), op)
        assert "old_definition=" in rendered

    def test_cascade_omitted_when_true(self):
        """Renderer omits cascade when True (the default)."""
        op = ReplaceViewOp("v", "SELECT 2")
        result = render_replace_view(_make_autogen_context(), op)
        assert "cascade=" not in result

    def test_cascade_included_when_false(self):
        """Renderer includes cascade=False when cascade is disabled."""
        op = ReplaceViewOp("v", "SELECT 2", cascade=False)
        result = render_replace_view(_make_autogen_context(), op)
        assert "cascade=False" in result


class TestRendererCreateMaterializedView:

    def test_produces_valid_python(self):
        op = CreateMaterializedViewOp("mv_stats", "SELECT count(*) FROM events")
        result = render_create_materialized_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_schema_included_when_provided(self):
        op = CreateMaterializedViewOp("mv_stats", "SELECT 1", schema="analytics")
        result = render_create_materialized_view(_make_autogen_context(), op)
        assert "schema='analytics'" in result


class TestRendererDropMaterializedView:

    def test_produces_valid_python(self):
        op = DropMaterializedViewOp("mv_stats")
        result = render_drop_materialized_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_cascade_omitted_when_true(self):
        op = DropMaterializedViewOp("mv_stats")
        result = render_drop_materialized_view(_make_autogen_context(), op)
        assert "cascade=" not in result

    def test_cascade_included_when_false(self):
        op = DropMaterializedViewOp("mv_stats", cascade=False)
        result = render_drop_materialized_view(_make_autogen_context(), op)
        assert "cascade=False" in result

    def test_renders_definition(self):
        op = DropMaterializedViewOp("mv", definition="SELECT 1")
        rendered = render_drop_materialized_view(_make_autogen_context(), op)
        assert "definition=" in rendered

    def test_renders_with_data_true(self):
        """``with_data=True`` must be preserved in the rendered downgrade.

        A manual ``op.drop_materialized_view(..., with_data=True)`` renders
        a downgrade that re-creates the materialized view. Without
        ``_with_data_part`` the rendered downgrade defaults to
        ``with_data=False``, silently changing the original ``WITH DATA``
        clause to ``WITH NO DATA``.
        """
        op = DropMaterializedViewOp("mv", definition="SELECT 1", with_data=True)
        rendered = render_drop_materialized_view(_make_autogen_context(), op)
        assert "with_data=True" in rendered, f"got: {rendered!r}"


class TestRendererReplaceMaterializedView:

    def test_produces_valid_python(self):
        op = ReplaceMaterializedViewOp("mv_stats", "SELECT count(*) FROM events")
        result = render_replace_materialized_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_schema_included_when_provided(self):
        op = ReplaceMaterializedViewOp("mv_stats", "SELECT 2", schema="analytics")
        result = render_replace_materialized_view(_make_autogen_context(), op)
        assert "schema='analytics'" in result

    def test_renders_old_definition(self):
        op = ReplaceMaterializedViewOp("mv_repl", "SELECT 2", old_definition="SELECT 1")
        rendered = render_replace_materialized_view(_make_autogen_context(), op)
        assert "old_definition=" in rendered


class TestRendererMaterializedViewWithDataDefault:

    @pytest.mark.parametrize(
        "factory, render_fn",
        [
            (lambda name, **kw: CreateMaterializedViewOp(name, "SELECT 1", **kw), render_create_materialized_view),
            (lambda name, **kw: DropMaterializedViewOp(name, definition="SELECT 1", **kw), render_drop_materialized_view),
            (lambda name, **kw: ReplaceMaterializedViewOp(name, "SELECT 1", **kw), render_replace_materialized_view),
        ],
        ids=["create", "drop", "replace"],
    )
    def test_omits_with_data_when_default(self, factory, render_fn):
        op = factory("mv")
        result = render_fn(_make_autogen_context(), op)
        assert "with_data=" not in result

        op_false = factory("mv", with_data=False)
        result_false = render_fn(_make_autogen_context(), op_false)
        assert "with_data=" not in result_false

        op_true = factory("mv", with_data=True)
        result_true = render_fn(_make_autogen_context(), op_true)
        assert "with_data=True" in result_true


# ===========================================================================
# Comparator
# ===========================================================================

def _create_base_table(connection):
    """Create a base table needed for view tests. Idempotent."""
    connection.execute(
        sa.text(
            "CREATE TABLE IF NOT EXISTS _cmp_test_base "
            "(id SERIAL PRIMARY KEY, name TEXT, value INTEGER)"
        )
    )
    connection.commit()


def _drop_base_table(connection):
    """Drop the base table used for view tests."""
    connection.execute(sa.text("DROP TABLE IF EXISTS _cmp_test_base CASCADE"))
    connection.commit()


@pytest.fixture
def cmp_test_base(connection):
    """Create ``_cmp_test_base`` and clean comparator test views before/after."""
    _drop_views(connection, _CMP_TEST_VIEW_NAMES)
    _create_base_table(connection)
    yield connection
    _drop_views(connection, _CMP_TEST_VIEW_NAMES)
    _drop_base_table(connection)


@pytest.fixture
def view_cleanup_factory(connection, request):
    """Factory fixture for per-test view/table setup+teardown.

    Returns a callable ``setup(view_names, create_base=False)`` that:
    - drops any leftover views from ``view_names`` (and optionally the base
      table) for a clean slate,
    - optionally creates ``_cmp_test_base``,
    - registers a finalizer that drops those views + the base table.

    Use in tests that manage their own view-name lists (distinct from the
    ``cmp_test_base`` / ``int_test_base`` fixtures which use fixed lists).
    """

    def setup(view_names, *, create_base=False):
        _drop_views(connection, view_names)
        if not create_base:
            _drop_base_table(connection)
        else:
            _create_base_table(connection)
        request.addfinalizer(lambda: (_drop_views(connection, view_names), _drop_base_table(connection)))
        return connection

    return setup


class TestComparatorCreateView:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_view_generates_create_view_op(self, cmp_test_base):
        connection = cmp_test_base
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="cmp_test_view",
                selectable="SELECT id, name FROM _cmp_test_base",
            )
        ]

        upgrade_ops = _run_comparator(connection, metadata)

        view_ops = [op for op in upgrade_ops.ops if isinstance(op, CreateViewOp)]
        assert len(view_ops) == 1
        assert view_ops[0].name == "cmp_test_view"


class TestComparatorDropView:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_view_generates_drop_view_op(self, cmp_test_base):
        connection = cmp_test_base
        connection.execute(
            sa.text(
                "CREATE VIEW cmp_test_view AS SELECT id, name FROM _cmp_test_base"
            )
        )
        connection.commit()

        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = []

        upgrade_ops = _run_comparator(connection, metadata)

        drop_ops = [op for op in upgrade_ops.ops if isinstance(op, DropViewOp)]
        assert len(drop_ops) == 1
        assert drop_ops[0].name == "cmp_test_view"


class TestComparatorReplaceView:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_view_generates_replace_view_op(self, cmp_test_base):
        connection = cmp_test_base
        connection.execute(
            sa.text(
                "CREATE VIEW cmp_test_changed AS "
                "SELECT id, name FROM _cmp_test_base WHERE value > 0"
            )
        )
        connection.commit()

        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="cmp_test_changed",
                selectable="SELECT id, name FROM _cmp_test_base",
            )
        ]

        upgrade_ops = _run_comparator(connection, metadata)

        replace_ops = [op for op in upgrade_ops.ops if isinstance(op, ReplaceViewOp)]
        assert len(replace_ops) == 1
        assert replace_ops[0].name == "cmp_test_changed"
        assert replace_ops[0].old_definition is not None


class TestComparatorCreateMV:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_mv_generates_create_mv_op(self, cmp_test_base):
        connection = cmp_test_base
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="cmp_test_mv",
                selectable="SELECT id, name FROM _cmp_test_base",
                materialized=True,
            )
        ]

        upgrade_ops = _run_comparator(connection, metadata)

        mv_ops = [
            op for op in upgrade_ops.ops if isinstance(op, CreateMaterializedViewOp)
        ]
        assert len(mv_ops) == 1
        assert mv_ops[0].name == "cmp_test_mv"
        assert mv_ops[0].with_data is False


class TestComparatorDropMV:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_mv_generates_drop_mv_op(self, cmp_test_base):
        connection = cmp_test_base
        connection.execute(
            sa.text(
                "CREATE MATERIALIZED VIEW cmp_test_mv AS "
                "SELECT id, name FROM _cmp_test_base WITH DATA"
            )
        )
        connection.commit()

        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = []

        upgrade_ops = _run_comparator(connection, metadata)

        drop_mv_ops = [
            op for op in upgrade_ops.ops if isinstance(op, DropMaterializedViewOp)
        ]
        assert len(drop_mv_ops) == 1
        assert drop_mv_ops[0].name == "cmp_test_mv"


class TestComparatorReplaceMV:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_mv_generates_replace_mv_op(self, cmp_test_base):
        connection = cmp_test_base
        connection.execute(
            sa.text(
                "CREATE MATERIALIZED VIEW cmp_test_mv_changed AS "
                "SELECT id, name FROM _cmp_test_base WITH DATA"
            )
        )
        connection.commit()

        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="cmp_test_mv_changed",
                selectable="SELECT id, value FROM _cmp_test_base",
                materialized=True,
            )
        ]

        upgrade_ops = _run_comparator(connection, metadata)

        replace_ops = [
            op for op in upgrade_ops.ops if isinstance(op, ReplaceMaterializedViewOp)
        ]
        assert len(replace_ops) == 1
        assert replace_ops[0].name == "cmp_test_mv_changed"
        assert replace_ops[0].with_data is False
        assert replace_ops[0].old_definition is not None


class TestComparatorNoChanges:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_no_changes_no_ops(self, cmp_test_base):
        connection = cmp_test_base
        connection.execute(
            sa.text(
                "CREATE VIEW cmp_test_view2 AS SELECT id, name FROM _cmp_test_base"
            )
        )
        connection.commit()

        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="cmp_test_view2",
                selectable="SELECT id, name FROM _cmp_test_base",
            )
        ]

        upgrade_ops = _run_comparator(connection, metadata)

        matching_view_ops = [
            op
            for op in upgrade_ops.ops
            if getattr(op, "name", None) == "cmp_test_view2"
        ]
        assert len(matching_view_ops) == 0, f"got: {matching_view_ops}"

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_no_change_materialized_view(self, cmp_test_base):
        connection = cmp_test_base
        connection.execute(
            sa.text(
                "CREATE MATERIALIZED VIEW cmp_test_mv AS "
                "SELECT id, name FROM _cmp_test_base WITH DATA"
            )
        )
        connection.commit()

        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="cmp_test_mv",
                selectable="SELECT id, name FROM _cmp_test_base",
                materialized=True,
            )
        ]

        upgrade_ops = _run_comparator(connection, metadata)

        mv_ops = [
            op
            for op in upgrade_ops.ops
            if isinstance(
                op,
                (
                    CreateMaterializedViewOp,
                    ReplaceMaterializedViewOp,
                    DropMaterializedViewOp,
                ),
            )
        ]
        assert len(mv_ops) == 0, f"got: {mv_ops}"


class TestComparatorSavepointRollback:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_canonicalized_view_does_not_persist(self, cmp_test_base):
        connection = cmp_test_base
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="cmp_test_view",
                selectable="SELECT id, name FROM _cmp_test_base",
            )
        ]

        _run_comparator(connection, metadata)

        db_views = get_database_views(connection)
        assert "cmp_test_view" not in db_views, f"got: {db_views}"


class TestComparatorDDLError:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_invalid_view_skipped_with_warning(self, connection):
        _drop_views(connection, _CMP_TEST_VIEW_NAMES)
        _drop_base_table(connection)
        try:

            metadata = sa.MetaData()
            metadata.info["sqlalchemy_utils_views"] = [
                ViewRecord(
                    name="cmp_test_view_bad",
                    selectable="SELECT id FROM nonexistent_table_xyz",
                )
            ]

            upgrade_ops = _run_comparator(connection, metadata)

            bad_view_ops = [
                op
                for op in upgrade_ops.ops
                if getattr(op, "name", None) == "cmp_test_view_bad"
            ]
            assert len(bad_view_ops) == 0, f"got ops: {bad_view_ops}"
        finally:
            _drop_views(connection, _CMP_TEST_VIEW_NAMES)
            _drop_base_table(connection)


# ===========================================================================
# Regression: programming errors must propagate, not be swallowed
# ===========================================================================

class _BreakingSelectable:
    """Raises *exc_type* when compiled against a dialect."""
    def __init__(self, exc_type=sa.exc.CompileError):
        self._exc_type = exc_type
    def compile(self, **kw):
        if "dialect" in kw:
            raise self._exc_type("Cannot compile clause element for this dialect")
        return "SELECT 1 AS id"


class TestProgrammingErrorPropagates:
    """Programming errors during canonicalization must propagate, not be
    swallowed by the broad except in ``_canonicalize_all_views``."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_programming_error_propagates(self, connection):
        _drop_views(connection, _CMP_TEST_VIEW_NAMES)
        _drop_base_table(connection)
        try:
            metadata = sa.MetaData()
            metadata.info["sqlalchemy_utils_views"] = [
                ViewRecord(
                    name="broken_view_for_dialect_test",
                    selectable=_BreakingSelectable(exc_type=TypeError),
                    schema=None,
                ),
            ]

            with pytest.raises(TypeError):
                _run_comparator(connection, metadata, schemas=[None])
        finally:
            _drop_views(connection, _CMP_TEST_VIEW_NAMES)
            _drop_base_table(connection)


# ===========================================================================
# Regression: canonicalization savepoint refactor
# ===========================================================================

# Distinct names so tests don't collide with other view fixtures.
_CLEANUP_VIEW_NAMES = ["failed_canon_view", "dep_chain_a", "dep_chain_b"]


class TestCanonicalizeViewOnViewDeps:
    """Regression: view-on-view dependencies must survive the savepoint.

    Two new model views that reference each other must both produce
    CreateViewOp (single outer savepoint keeps both alive during canonicalization).
    """

    @pytest.mark.infrastructure
    @pytest.mark.usefixtures("postgresql_dsn")
    def test_dependent_view_chain_both_created(self, view_cleanup_factory):
        connection = view_cleanup_factory(_CLEANUP_VIEW_NAMES)
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="dep_chain_a",
                selectable=sa.select(sa.text("1 AS id")),
                schema=None,
            ),
            ViewRecord(
                name="dep_chain_b",
                selectable=sa.select(sa.text("* FROM dep_chain_a")),
                schema=None,
            ),
        ]

        upgrade_ops = _run_comparator(connection, metadata, schemas=[None])

        create_ops = [
            op for op in upgrade_ops.ops if isinstance(op, CreateViewOp)
        ]
        created_names = {op.name for op in create_ops}
        assert "dep_chain_a" in created_names, f"got {sorted(created_names)}"
        assert "dep_chain_b" in created_names, f"got {sorted(created_names)}"


class TestCanonicalizeSkipDoesNotDrop:
    """Regression: a view whose canonicalization fails must be SKIPPED, not dropped."""

    @pytest.mark.infrastructure
    @pytest.mark.usefixtures("postgresql_dsn")
    def test_failing_canonicalization_does_not_emit_drop(self, view_cleanup_factory):
        connection = view_cleanup_factory(_CLEANUP_VIEW_NAMES)
        # Pre-create the view in the DB with an old, valid definition.
        connection.execute(
            sa.text("CREATE VIEW failed_canon_view AS SELECT 1 AS id")
        )
        connection.commit()

        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="failed_canon_view",
                selectable=sa.select(sa.text("* FROM nonexistent_table")),
                schema=None,
            ),
        ]

        upgrade_ops = _run_comparator(connection, metadata, schemas=[None])

        drop_ops = [
            op for op in upgrade_ops.ops if isinstance(op, DropViewOp)
        ]
        failed_canon_drops = [op for op in drop_ops if op.name == "failed_canon_view"]
        assert failed_canon_drops == [], f"got drop ops: {[(op.name, op.schema) for op in drop_ops]}"


# ===========================================================================
# Regression: savepoint name reuse after ROLLBACK TO skips later views
# ===========================================================================

# Distinct names so this regression test does not collide with other view fixtures.
_SAVEPOINT_TEST_VIEW_NAMES = ["savepoint_a", "savepoint_b", "savepoint_c"]


class TestCanonicalizeFailureDoesNotSkipSubsequentViews:
    """Regression: a failed view must not cascade-skip later views.

    After a view CREATE fails, ROLLBACK TO + RELEASE of the inner savepoint
    lets subsequent views (B, C) still produce CreateViewOp.
    """

    @pytest.mark.infrastructure
    @pytest.mark.usefixtures("postgresql_dsn")
    def test_failed_canonicalization_does_not_skip_subsequent_views(
        self, view_cleanup_factory
    ):
        connection = view_cleanup_factory(_SAVEPOINT_TEST_VIEW_NAMES)

        # view_a references a nonexistent table → CREATE fails inside the
        # canonicalization savepoint. view_b and view_c are trivially valid
        # (no table dependency) so they MUST still be canonicalized.
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="savepoint_a",
                selectable=sa.select(sa.text("* FROM nonexistent_table")),
                schema=None,
                materialized=False,
            ),
            ViewRecord(
                name="savepoint_b",
                selectable=sa.select(sa.text("1 AS col")),
                schema=None,
                materialized=False,
            ),
            ViewRecord(
                name="savepoint_c",
                selectable=sa.select(sa.text("2 AS col")),
                schema=None,
                materialized=False,
            ),
        ]

        upgrade_ops = _run_comparator(
            connection, metadata, schemas=[None]
        )

        create_ops = [
            op for op in upgrade_ops.ops if isinstance(op, CreateViewOp)
        ]
        created_names = {op.name for op in create_ops}

        # savepoint_a failed to canonicalize → must NOT appear.
        assert "savepoint_a" not in created_names, f"got {sorted(created_names)}"
        # savepoint_b and savepoint_c come after the failure; they MUST still be
        # canonicalized. Before the fix the reused savepoint name caused
        # both to be silently dropped.
        assert "savepoint_b" in created_names, f"got {sorted(created_names)}"
        assert "savepoint_c" in created_names, f"got {sorted(created_names)}"


# ===========================================================================
# Regression: outer savepoint poisoned by a DB-level error breaks early
# ===========================================================================


class TestAbortedTransactionBreaksEarly:
    """A poisoned outer savepoint must break the canonicalization loop.

    After a view CREATE fails and the inner savepoint is rolled back/released,
    the outer savepoint may be aborted. The ``SELECT 1`` probe detects this,
    logs a warning containing "aborted state", and breaks early so remaining
    views are added to ``skipped`` (not silently dropped). Mock-based, no PG.
    """

    def test_aborted_transaction_breaks_early(self, caplog):
        # Two views: view_a's CREATE fails, view_b should never be reached
        # because the probe after view_a's failure detects the poisoned
        # transaction and breaks the loop.
        view_records = [
            ViewRecord(
                name="abort_a",
                selectable="SELECT id FROM nonexistent_table_a",
                schema=None,
                materialized=False,
            ),
            ViewRecord(
                name="abort_b",
                selectable="SELECT 1 AS col",
                schema=None,
                materialized=False,
            ),
        ]

        # The mock connection captures every `execute(sa.text(...))` call and
        # returns a side effect based on the SQL text. The sequence of SQL
        # statements issued by _canonicalize_all_views is:
        #   1. SAVEPOINT su_view_cmp                       (outer)
        #   2. SAVEPOINT su_view_cmp_v                    (view_a inner)
        #   3. DROP VIEW IF EXISTS abort_a CASCADE        (succeeds)
        #   4. CREATE VIEW abort_a AS ...                 (FAILS — view_a)
        #   5. ROLLBACK TO SAVEPOINT su_view_cmp_v        (view_a cleanup)
        #   6. RELEASE SAVEPOINT su_view_cmp_v            (view_a cleanup)
        #   7. SELECT 1                                   (PROBE — FAILS)
        #   --- loop breaks; view_b is never touched ---
        #   8. ROLLBACK TO SAVEPOINT su_view_cmp          (finally)
        # After the loop, get_database_views / get_database_materialized_views
        # are called — stubbed via patch to return empty dicts.
        call_log: list[str] = []
        connection = _make_poisoned_connection("abort_a", call_log)

        with caplog.at_level(
            logging.WARNING, logger="sqlalchemy_utils.alembic.comparator"
        ), patch(
            "sqlalchemy_utils.alembic.comparator.get_database_views",
            return_value={},
        ), patch(
            "sqlalchemy_utils.alembic.comparator.get_database_materialized_views",
            return_value={},
        ):
            view_defs, mv_defs, skipped = _canonicalize_all_views(
                connection, view_records, db_views_for_deps=None
            )

        # 1. A warning containing "aborted state" was logged.
        abort_warnings = [
            rec
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
            and "aborted state" in rec.message
        ]
        assert abort_warnings, f"got log records: {[(rec.levelname, rec.message) for rec in caplog.records]}"

        # 2. The loop broke early — view_b (abort_b) was never canonicalized.
        #    Because the loop broke before reaching it, abort_b is added to
        #    `skipped` so drop detection does not emit a false DropViewOp for
        #    a view that is still modeled but merely un-processed.
        assert "abort_b" in skipped, f"skipped={skipped}"
        assert "abort_b" not in view_defs, f"view_defs={view_defs}"
        # abort_a WAS attempted and failed — it should be in skipped.
        assert "abort_a" in skipped, f"skipped={skipped}"

        # 3. No CREATE statement for abort_b was ever issued (loop broke).
        abort_b_creates = [
            sql for sql in call_log if "abort_b" in sql and "CREATE" in sql
        ]
        assert not abort_b_creates, f"executed: {abort_b_creates}"

    def test_aborted_transaction_no_false_drop_in_compare_views(self):
        """Regression: an aborted transaction must NOT cause a false DropViewOp
        for an un-processed view (it must be in ``skipped`` instead)."""
        # Two model views, both present in the DB mock.
        view_records = [
            ViewRecord(
                name="view_a",
                selectable="SELECT 1 AS col",
                schema=None,
                materialized=False,
            ),
            ViewRecord(
                name="view_b",
                selectable="SELECT 2 AS col",
                schema=None,
                materialized=False,
            ),
        ]

        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = view_records

        call_log: list[str] = []
        connection = _make_poisoned_connection("view_a", call_log)

        autogen_context = MagicMock(spec=AutogenContext)
        autogen_context.connection = connection
        autogen_context.metadata = metadata

        upgrade_ops = alembic_ops.UpgradeOps([])

        # The DB mock reports BOTH views as existing — so without the fix,
        # view_b (un-processed, un-skipped) would be seen as "in DB but not
        # in model" → false DropViewOp.
        db_views_mock = {
            "view_a": "SELECT 1 AS col",
            "view_b": "SELECT 2 AS col",
        }

        with patch(
            "sqlalchemy_utils.alembic.comparator.get_database_views",
            return_value=db_views_mock,
        ), patch(
            "sqlalchemy_utils.alembic.comparator.get_database_materialized_views",
            return_value={},
        ), patch(
            "sqlalchemy_utils.alembic.comparator.get_dependent_views",
            return_value={},
        ):
            compare_views(autogen_context, upgrade_ops, schemas=[None])

        # Collect all DropViewOp (regular + materialized) emitted.
        drop_ops = [
            op for op in upgrade_ops.ops
            if isinstance(op, (DropViewOp, DropMaterializedViewOp))
        ]
        drop_names = {op.name for op in drop_ops}

        # The critical assertion: NO DropViewOp for view_b.
        assert "view_b" not in drop_names, f"drop ops: {drop_names}"


class TestComparatorNonPGDialect:

    def test_warns_on_non_pg_dialect(self, caplog):
        engine = sa.create_engine("sqlite:///:memory:")
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = []

        autogen_context = MagicMock()
        with engine.connect() as conn:
            autogen_context.connection = conn
            autogen_context.metadata = metadata

            upgrade_ops = MagicMock()
            upgrade_ops.ops = []
            schemas = [None]

            with caplog.at_level(
                logging.WARNING, logger="sqlalchemy_utils.alembic.comparator"
            ):
                raised_exc = None
                try:
                    compare_views(autogen_context, upgrade_ops, schemas)
                except Exception as exc:
                    raised_exc = exc

            warnings = [rec for rec in caplog.records if rec.levelno >= logging.WARNING]

        assert raised_exc is None
        assert warnings
        assert any(
            "non" in rec.message.lower() and "postgres" in rec.message.lower()
            for rec in warnings
        )

class TestComparatorNoDoubleFetch:

    def test_does_not_double_fetch(self):

        call_count = {"views": 0, "mvs": 0}

        def mock_get_database_views(connection, schema=None):
            call_count["views"] += 1
            return {}

        def mock_get_database_mvs(connection, schema=None):
            call_count["mvs"] += 1
            return {}

        autogen_context, upgrade_ops = _make_mock_autogen_context(
            model_views=[]
        )

        with patch.object(
            comparator_module, "get_database_views", mock_get_database_views
        ), patch.object(
            comparator_module, "get_database_materialized_views", mock_get_database_mvs
        ), patch.object(
            comparator_module, "_canonicalize_all_views",
            return_value=({}, {}, set()),
        ):
            compare_views(autogen_context, upgrade_ops, [None, "analytics"])

        assert call_count["views"] == 2
        assert call_count["mvs"] == 2


class TestComparatorNeverEmitsRefreshOp:
    """Guard: compare_views must never emit RefreshMaterializedViewOp
    (refresh is a runtime op, not a reversible migration step)."""

    def test_compare_views_never_emits_refresh_materialized_view_op(self):
        # Mock connection + autogen_context for a postgres dialect, with one
        # materialized view record in the model and a matching definition in
        # the DB so _diff_views produces no ops at all — but even if diffing
        # produced ops, none may be RefreshMaterializedViewOp.
        autogen_context, upgrade_ops = _make_mock_autogen_context(
            model_views=[]
        )

        with patch.object(
            comparator_module, "get_database_views", return_value={}
        ), patch.object(
            comparator_module,
            "get_database_materialized_views",
            return_value={},
        ), patch.object(
            comparator_module,
            "_canonicalize_all_views",
            return_value=({}, {}, set()),
        ):
            compare_views(autogen_context, upgrade_ops, [None])

        refresh_ops = [
            op for op in upgrade_ops.ops
            if isinstance(op, RefreshMaterializedViewOp)
        ]
        assert refresh_ops == [], f"found: {refresh_ops}"


# ===========================================================================
# Dependency resolution
# ===========================================================================

class TestDependIndependentViews:

    def test_independent_views_create_order_contains_all(self):
        views = [
            ViewRecord(name="alpha", selectable="SELECT 1"),
            ViewRecord(name="beta", selectable="SELECT 2"),
            ViewRecord(name="gamma", selectable="SELECT 3"),
        ]
        result = resolve_create_order(views, db_views={})
        assert set(v.name for v in result) == {"alpha", "beta", "gamma"}
        assert len(result) == 3

    def test_independent_views_drop_order_contains_all(self):
        views = [
            ViewRecord(name="alpha", selectable="SELECT 1"),
            ViewRecord(name="beta", selectable="SELECT 2"),
        ]
        result = resolve_drop_order(views, db_views={})
        assert set(v.name for v in result) == {"alpha", "beta"}

    def test_empty_list(self):
        assert resolve_create_order([], db_views={}) == []


class TestDependViewOnView:

    def test_dependent_after_dependency_in_create_order(self):
        views = [
            ViewRecord(name="child_view", selectable="SELECT * FROM parent_view"),
            ViewRecord(name="parent_view", selectable="SELECT 1 AS col"),
        ]
        result = resolve_create_order(views, db_views={})
        names = [v.name for v in result]
        assert names.index("parent_view") < names.index("child_view")

    def test_dependent_before_dependency_in_drop_order(self):
        views = [
            ViewRecord(name="child_view", selectable="SELECT * FROM parent_view"),
            ViewRecord(name="parent_view", selectable="SELECT 1 AS col"),
        ]
        result = resolve_drop_order(views, db_views={})
        names = [v.name for v in result]
        assert names.index("child_view") < names.index("parent_view")

    def test_single_view(self):
        vr = ViewRecord(name="solo", selectable="SELECT 1 AS col")
        result = resolve_create_order([vr], db_views={})
        assert len(result) == 1
        assert result[0].name == "solo"

    def test_self_referencing(self):
        """A self-referencing view resolves without an infinite loop."""
        vr = ViewRecord(name="recursive", selectable="SELECT * FROM recursive")
        result = resolve_create_order([vr], db_views={})
        assert len(result) == 1
        assert result[0].name == "recursive"


class TestDependMultipleLevels:

    def test_chain_create_order(self):
        views = [
            ViewRecord(name="a", selectable="SELECT * FROM b"),
            ViewRecord(name="b", selectable="SELECT * FROM c"),
            ViewRecord(name="c", selectable="SELECT 1 AS col"),
        ]
        result = resolve_create_order(views, db_views={})
        names = [v.name for v in result]
        assert names.index("c") < names.index("b") < names.index("a")

    def test_chain_drop_order(self):
        views = [
            ViewRecord(name="a", selectable="SELECT * FROM b"),
            ViewRecord(name="b", selectable="SELECT * FROM c"),
            ViewRecord(name="c", selectable="SELECT 1 AS col"),
        ]
        result = resolve_drop_order(views, db_views={})
        names = [v.name for v in result]
        assert names.index("a") < names.index("b") < names.index("c")


class TestDependDiamondDependency:
    """Diamond graph: A depends on B and C; B and C both depend on D.

    This shape has multiple paths from A to D.  Both ``resolve_create_order``
    (dependencies first) and ``resolve_drop_order`` (dependents first) must
    produce a valid topological ordering of all four views.
    """

    @pytest.fixture
    def diamond_views(self):
        return [
            ViewRecord(name="diamond_a", selectable="SELECT * FROM diamond_b JOIN diamond_c"),
            ViewRecord(name="diamond_b", selectable="SELECT * FROM diamond_d"),
            ViewRecord(name="diamond_c", selectable="SELECT * FROM diamond_d"),
            ViewRecord(name="diamond_d", selectable="SELECT 1 AS col"),
        ]

    def test_create_order_dependencies_before_dependents(self, diamond_views):
        result = resolve_create_order(diamond_views, db_views={})
        names = [v.name for v in result]
        # D has no deps -> must come before B and C; B and C before A.
        assert names.index("diamond_d") < names.index("diamond_b")
        assert names.index("diamond_d") < names.index("diamond_c")
        assert names.index("diamond_b") < names.index("diamond_a")
        assert names.index("diamond_c") < names.index("diamond_a")
        assert names[-1] == "diamond_a"
        assert names[0] == "diamond_d"

    def test_drop_order_dependents_before_dependencies(self, diamond_views):
        result = resolve_drop_order(diamond_views, db_views={})
        names = [v.name for v in result]
        # A is the top dependent -> dropped first; D has no dependents -> dropped last.
        assert names[0] == "diamond_a"
        assert names[-1] == "diamond_d"
        assert names.index("diamond_a") < names.index("diamond_b")
        assert names.index("diamond_a") < names.index("diamond_c")
        assert names.index("diamond_b") < names.index("diamond_d")
        assert names.index("diamond_c") < names.index("diamond_d")


class TestDependCircular:

    def test_simple_cycle_raises_value_error(self):
        views = [
            ViewRecord(name="view_a", selectable="SELECT * FROM view_b"),
            ViewRecord(name="view_b", selectable="SELECT * FROM view_a"),
        ]
        with pytest.raises(ValueError, match="[Cc]ircular"):
            resolve_create_order(views, db_views={})

    def test_cycle_in_drop_order_raises_value_error(self):
        views = [
            ViewRecord(name="x", selectable="SELECT * FROM y"),
            ViewRecord(name="y", selectable="SELECT * FROM x"),
        ]
        with pytest.raises(ValueError, match="[Cc]ircular"):
            resolve_drop_order(views, db_views={})


class TestDependMaterializedViews:

    def test_materialized_view_after_base_table(self):
        views = [
            ViewRecord(
                name="mv_summary",
                selectable="SELECT * FROM v_base",
                materialized=True,
            ),
            ViewRecord(
                name="v_base",
                selectable="SELECT 1 AS col",
                materialized=False,
            ),
        ]
        result = resolve_create_order(views, db_views={})
        names = [v.name for v in result]
        assert names.index("v_base") < names.index("mv_summary")

    def test_materialized_chain(self):
        views = [
            ViewRecord(
                name="mv_level2",
                selectable="SELECT * FROM mv_level1",
                materialized=True,
            ),
            ViewRecord(
                name="mv_level1",
                selectable="SELECT 1 AS col",
                materialized=True,
            ),
        ]
        result = resolve_create_order(views, db_views={})
        names = [v.name for v in result]
        assert names.index("mv_level1") < names.index("mv_level2")


class TestDependDbViews:

    def test_db_view_satisfies_dependency(self):
        views = [
            ViewRecord(
                name="model_view", selectable="SELECT * FROM existing_db_view"
            ),
        ]
        db_views = {"existing_db_view": "SELECT 1 AS col"}
        result = resolve_create_order(views, db_views=db_views)
        assert len(result) == 1
        assert result[0].name == "model_view"

    def test_db_view_not_included_in_output(self):
        views = [
            ViewRecord(name="child", selectable="SELECT * FROM db_parent"),
        ]
        db_views = {"db_parent": "SELECT 1"}
        result = resolve_create_order(views, db_views=db_views)
        names = [v.name for v in result]
        assert "db_parent" not in names
        assert "child" in names


class TestDependNoneDbViews:

    def test_none_db_views(self):
        vr = ViewRecord(name="solo", selectable="SELECT 1 AS col")
        assert resolve_create_order([vr], db_views=None) == [vr]
        assert resolve_drop_order([vr], db_views=None) == [vr]
        assert _build_dependency_graph([vr], {}) == {("solo", None): set()}


class TestDependWordBoundary:

    def test_partial_name_no_false_positive(self):
        views = [
            ViewRecord(name="log", selectable="SELECT 1"),
            ViewRecord(name="report", selectable="SELECT * FROM log_entries"),
        ]
        result = resolve_create_order(views, db_views={})
        assert set(v.name for v in result) == {"log", "report"}

    def test_exact_name_with_word_boundary(self):
        views = [
            ViewRecord(name="log", selectable="SELECT 1"),
            ViewRecord(name="report", selectable="SELECT * FROM log"),
        ]
        result = resolve_create_order(views, db_views={})
        names = [v.name for v in result]
        assert names.index("log") < names.index("report")

    def test_keyword_named_view_has_dependency_tracked(self):
        """A view named after a SQL keyword (e.g. ``user``) still participates
        in dependency matching."""
        summary_view = ViewRecord(
            name="summary", selectable="SELECT * FROM user"
        )
        user_view = ViewRecord(name="user", selectable="SELECT 1 AS col")
        result = resolve_create_order([summary_view, user_view], db_views={})
        names = [v.name for v in result]
        assert names.index("user") < names.index("summary")


class TestSafeResolveHandlesCompileError:
    """``_safe_resolve`` must fall back to model order when a ClauseElement
    raises ``sa.exc.CompileError`` during ``compiled_definition(dialect=...)``.

    ``_safe_resolve`` previously caught only ``ValueError`` (for
    ``CycleError``). A ``ClauseElement`` that fails compilation raises
    ``sa.exc.CompileError`` (a subclass of ``sa.exc.SQLAlchemyError``),
    which propagated uncaught and aborted the entire autogenerate run.
    The except clause must be widened so a single un-compilable view does
    not crash autogenerate — the resolver falls back to model order.
    """

    def test_compile_error_falls_back_to_model_order(self, caplog):
        records = [
            ViewRecord(
                name="broken_view",
                selectable=_BreakingSelectable(exc_type=sa.exc.CompileError),
                schema=None,
            ),
            ViewRecord(
                name="other_view",
                selectable="SELECT 2 AS id",
                schema=None,
            ),
        ]

        with caplog.at_level(
            logging.WARNING, logger="sqlalchemy_utils.alembic.comparator"
        ):
            result = _safe_resolve(
                records,
                {},
                resolve_create_order,
                "creating",
                dialect=sa.dialects.postgresql.dialect(),
            )

        # Must return the records (model order), not raise.
        assert result == records, f"got {result}"
        # A warning must be logged (do not silently swallow).
        warnings = [
            rec for rec in caplog.records if rec.levelno >= logging.WARNING
        ]
        assert warnings, f"got {[(rec.levelname, rec.message) for rec in caplog.records]}"


# ===========================================================================
# Alembic autogenerate test fixtures and helpers
# ===========================================================================

_ENV_PY_TEMPLATE = textwrap.dedent("""\
    from __future__ import annotations

    from alembic import context
    from sqlalchemy import pool

    config = context.config

    target_metadata = config.attributes.get("target_metadata")

    def run_migrations_offline() -> None:
        url = config.get_main_option("sqlalchemy.url")
        context.configure(
            url=url,
            target_metadata=target_metadata,
            literal_binds=True,
            dialect_opts={"paramstyle": "named"},
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()

    def run_migrations_online() -> None:
        connectable = config.attributes.get("connection")
        if connectable is not None:
            from sqlalchemy.engine import Connection as _Connection
            if isinstance(connectable, _Connection):
                context.configure(
                    connection=connectable,
                    target_metadata=target_metadata,
                    compare_type=True,
                )
                with context.begin_transaction():
                    context.run_migrations()
            else:
                with connectable.connect() as connection:
                    context.configure(
                        connection=connection,
                        target_metadata=target_metadata,
                        compare_type=True,
                    )
                    with context.begin_transaction():
                        context.run_migrations()
        else:
            from sqlalchemy import engine_from_config
            configuration = config.get_section(config.config_ini_section, {})
            connectable = engine_from_config(
                configuration,
                prefix="sqlalchemy.",
                poolclass=pool.NullPool,
            )
            with connectable.connect() as connection:
                context.configure(
                    connection=connection,
                    target_metadata=target_metadata,
                    compare_type=True,
                )
                with context.begin_transaction():
                    context.run_migrations()

    if context.is_offline_mode():
        run_migrations_offline()
    else:
        run_migrations_online()
""")

_SCRIPT_MAKO_TEMPLATE = textwrap.dedent('''\
    """${message}

    Revision ID: ${up_revision}
    Revises: ${down_revision | comma,n}
    Create Date: ${create_date}

    """
    from typing import Sequence, Union

    from alembic import op
    import sqlalchemy as sa
    ${imports if imports else ""}

    # revision identifiers, used by Alembic.
    revision: str = ${repr(up_revision)}
    down_revision: Union[str, Sequence[str], None] = ${repr(down_revision)}
    branch_labels: Union[str, Sequence[str], None] = ${repr(branch_labels)}
    depends_on: Union[str, Sequence[str], None] = ${repr(depends_on)}


    def upgrade() -> None:
        """Upgrade schema."""
        ${upgrades if upgrades else "pass"}


    def downgrade() -> None:
        """Downgrade schema."""
        ${downgrades if downgrades else "pass"}
''')

_ALEMBIC_INI_TEMPLATE = textwrap.dedent("""\
    [alembic]
    script_location = {script_location}
    sqlalchemy.url = {sqlalchemy_url}
""")


@pytest.fixture
def alembic_config(tmp_path, connection, postgresql_dsn):
    """Create a temporary Alembic environment configured for autogenerate."""
    migrations_dir = tmp_path / "versions"
    migrations_dir.mkdir()

    env_py_path = tmp_path / "env.py"
    env_py_path.write_text(_ENV_PY_TEMPLATE, encoding="utf-8")

    script_mako_path = tmp_path / "script.py.mako"
    script_mako_path.write_text(_SCRIPT_MAKO_TEMPLATE, encoding="utf-8")

    ini_path = tmp_path / "alembic.ini"
    ini_content = _ALEMBIC_INI_TEMPLATE.format(
        script_location=str(tmp_path),
        sqlalchemy_url=postgresql_dsn,
    )
    ini_path.write_text(ini_content, encoding="utf-8")

    def _make_config(metadata: sa.MetaData) -> config.Config:
        cfg = config.Config(str(ini_path))
        cfg.attributes["connection"] = connection
        cfg.attributes["target_metadata"] = metadata
        return cfg

    return _make_config


def run_autogenerate(metadata: sa.MetaData, connection, alembic_config) -> str:
    """Run ``alembic revision --autogenerate`` and return the migration code."""
    cfg = alembic_config(metadata)
    command.revision(cfg, autogenerate=True, message="test")

    script_location = Path(cfg.get_main_option("script_location"))
    versions_dir = script_location / "versions"
    migration_files = sorted(
        versions_dir.glob("*.py"), key=lambda p: p.stat().st_mtime
    )

    if not migration_files:
        raise AssertionError(
            "No migration file was generated by alembic revision --autogenerate"
        )

    latest = migration_files[-1]
    code = latest.read_text(encoding="utf-8")
    latest.unlink()
    return code


def assert_op(migration_code: str, op_name: str, expected: bool = True) -> None:
    """Assert presence/absence of ``op.<op_name>(`` in *migration_code*.

    *expected*=True asserts the op IS present; *expected*=False asserts it
    is NOT present.
    """
    token = f"op.{op_name}("
    present = token in migration_code
    if present is not expected:
        raise AssertionError(
            f"Expected op.{op_name}( presence={expected} but got {present}.\n"
            f"Migration code:\n{migration_code}"
        )


# ===========================================================================
# Autogenerate integration (full Alembic autogenerate pipeline)
# ===========================================================================

# Module-level Table object mapping to the _cmp_test_base table created
# by _create_base_table(). Allows create_view() to reference columns.
_int_base_table = sa.Table(
    "_cmp_test_base",
    sa.MetaData(),
    sa.Column("id", sa.Integer, primary_key=True),
    sa.Column("name", sa.Text),
    sa.Column("value", sa.Integer),
)

_INT_TEST_VIEW_NAMES = [
    "int_test_new_view", "int_test_new_mv", "int_test_drop_view",
    "int_test_drop_mv", "int_test_change_view", "int_test_change_mv",
    "int_test_same_view", "int_test_view_a", "int_test_view_b",
]


@pytest.fixture
def int_test_base(connection):
    """Create ``_cmp_test_base`` and clean integration test views before/after."""
    _drop_views(connection, _INT_TEST_VIEW_NAMES)
    _drop_base_table(connection)
    _create_base_table(connection)
    yield connection
    _drop_views(connection, _INT_TEST_VIEW_NAMES)
    _drop_base_table(connection)


class TestIntegrationNewView:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_view_detected_and_rendered(self, int_test_base, alembic_config):
        connection = int_test_base
        register_view_comparator()
        metadata = sa.MetaData()
        create_view(
            "int_test_new_view", sa.select(_int_base_table.c.id), metadata
        )
        code = run_autogenerate(metadata, connection, alembic_config)
        assert_op(code, "create_view")
        assert "int_test_new_view" in code


class TestIntegrationNewMV:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_mv_detected_and_rendered(self, int_test_base, alembic_config):
        connection = int_test_base
        register_view_comparator()
        metadata = sa.MetaData()
        create_materialized_view(
            "int_test_new_mv", sa.select(_int_base_table.c.id), metadata
        )
        code = run_autogenerate(metadata, connection, alembic_config)
        assert_op(code, "create_materialized_view")
        assert "int_test_new_mv" in code


class TestIntegrationRemoval:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_view_generates_drop(self, int_test_base, alembic_config):
        connection = int_test_base
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_drop_view AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        register_view_comparator()
        metadata = sa.MetaData()
        code = run_autogenerate(metadata, connection, alembic_config)
        assert_op(code, "drop_view")

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_mv_generates_drop(self, int_test_base, alembic_config):
        connection = int_test_base
        connection.execute(
            sa.text(
                "CREATE MATERIALIZED VIEW int_test_drop_mv "
                "AS SELECT id FROM _cmp_test_base WITH NO DATA"
            )
        )
        connection.commit()
        register_view_comparator()
        metadata = sa.MetaData()
        code = run_autogenerate(metadata, connection, alembic_config)
        assert_op(code, "drop_materialized_view")


class TestIntegrationDefinitionChange:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_view_generates_replace(self, int_test_base, alembic_config):
        connection = int_test_base
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_change_view AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        register_view_comparator()
        metadata = sa.MetaData()
        create_view(
            "int_test_change_view",
            sa.select(_int_base_table.c.id, _int_base_table.c.name),
            metadata,
        )
        code = run_autogenerate(metadata, connection, alembic_config)
        assert_op(code, "replace_view")

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_mv_generates_replace(self, int_test_base, alembic_config):
        connection = int_test_base
        connection.execute(
            sa.text(
                "CREATE MATERIALIZED VIEW int_test_change_mv "
                "AS SELECT id FROM _cmp_test_base WITH NO DATA"
            )
        )
        connection.commit()
        register_view_comparator()
        metadata = sa.MetaData()
        create_materialized_view(
            "int_test_change_mv",
            sa.select(_int_base_table.c.id, _int_base_table.c.name),
            metadata,
        )
        code = run_autogenerate(metadata, connection, alembic_config)
        assert_op(code, "replace_materialized_view")


class TestIntegrationNoOp:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_unchanged_view_no_ops(self, int_test_base, alembic_config):
        connection = int_test_base
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_same_view AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        register_view_comparator()
        metadata = sa.MetaData()
        create_view(
            "int_test_same_view", sa.select(_int_base_table.c.id), metadata
        )
        code = run_autogenerate(metadata, connection, alembic_config)
        assert_op(code, "create_view", expected=False)
        assert_op(code, "drop_view", expected=False)
        assert_op(code, "replace_view", expected=False)


class TestIntegrationDependencyOrdering:

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_dependent_view_created_after_dependency(
        self, int_test_base, alembic_config
    ):
        connection = int_test_base
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_view_a AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        register_view_comparator()
        metadata = sa.MetaData()
        create_view(
            "int_test_view_a", sa.select(_int_base_table.c.id), metadata
        )
        vr_b = ViewRecord(
            name="int_test_view_b",
            selectable="SELECT id FROM int_test_view_a",
            schema=None,
            materialized=False,
        )
        metadata.info.setdefault("sqlalchemy_utils_views", []).append(vr_b)

        code = run_autogenerate(metadata, connection, alembic_config)
        assert_op(code, "create_view")
        assert "int_test_view_b" in code


# ===========================================================================
# Public API
# ===========================================================================

class TestPublicAPIImportable:

    @pytest.mark.parametrize(
        "op_import_path,op_name,args,kwargs,expected_definition",
        [
            ("sqlalchemy_utils.alembic:CreateViewOp", "test_view", ("test_view", "SELECT 1"), {}, "SELECT 1"),
            ("sqlalchemy_utils.alembic:DropViewOp", "test_view", ("test_view",), {}, None),
            ("sqlalchemy_utils.alembic:ReplaceViewOp", "test_view", ("test_view", "SELECT 2"), {}, "SELECT 2"),
            ("sqlalchemy_utils.alembic:CreateMaterializedViewOp", "test_mv", ("test_mv", "SELECT 1"), {}, "SELECT 1"),
            ("sqlalchemy_utils.alembic:DropMaterializedViewOp", "test_mv", ("test_mv",), {"cascade": False}, None),
            ("sqlalchemy_utils.alembic:ReplaceMaterializedViewOp", "test_mv", ("test_mv", "SELECT 2"), {}, "SELECT 2"),
            ("sqlalchemy_utils.alembic:get_database_materialized_views", None, None, {}, None),
            ("sqlalchemy_utils.alembic:get_database_views", None, None, {}, None),
            ("sqlalchemy_utils.alembic:resolve_create_order", None, None, {}, None),
            ("sqlalchemy_utils.alembic:resolve_drop_order", None, None, {}, None),
            ("sqlalchemy_utils.alembic:ViewRecord", None, None, {}, None),
            ("sqlalchemy_utils.alembic:register_view_comparator", None, None, {}, None),
        ],
        ids=[
            "create_view_op", "drop_view_op", "replace_view_op",
            "create_materialized_view_op", "drop_materialized_view_op",
            "replace_materialized_view_op",
            "get_database_materialized_views",
            "get_database_views", "resolve_create_order",
            "resolve_drop_order", "view_record",
            "register_view_comparator",
        ],
    )
    def test_import_op(
        self, op_import_path, op_name, args, kwargs, expected_definition
    ):
        import importlib

        module_path, _, attr = op_import_path.partition(":")
        symbol = getattr(importlib.import_module(module_path), attr)

        # Non-Op public API symbols: just assert importable + callable.
        if args is None:
            assert callable(symbol)
            return

        op = symbol(*args, **kwargs)
        assert op.name == op_name
        if expected_definition is not None:
            assert op.definition == expected_definition


# ===========================================================================
# Import safety
# ===========================================================================

class TestImportSafety:

    def test_importing_op_does_not_register_comparator(self):
        """Importing CreateViewOp does not register compare_views as a side effect."""
        code = (
            "import sys\n"
            "from alembic.autogenerate import comparators\n"
            "from sqlalchemy_utils.alembic.operations import CreateViewOp\n"
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
        assert result.returncode == 0
        last_line = result.stdout.strip().splitlines()[-1]
        assert last_line.startswith("HAS_COMPARE_VIEWS=")
        has_compare_views = last_line.split("=", 1)[1].strip() == "True"
        assert has_compare_views is False


# ===========================================================================
# DDL formatting
# ===========================================================================

class TestDDLFormatting:

    def test_drop_view_no_trailing_space(self):
        """DropView with cascade=False emits no trailing whitespace."""
        drop = DropView("my_view", cascade=False)
        sql = _compile_ddl(drop)
        assert sql.rstrip() == sql

    @pytest.mark.parametrize(
        "schema, concurrently, expected_substring",
        [
            pytest.param("analytics", False, "analytics", id="with_schema"),
            pytest.param(None, True, "CONCURRENTLY", id="concurrently"),
        ],
    )
    def test_refresh_materialized_view(
        self, schema, concurrently, expected_substring
    ):
        """refresh_materialized_view forwards schema/concurrently to the DDL element."""
        session = mock.MagicMock()
        refresh_materialized_view(
            session, "my_mv", concurrently=concurrently, schema=schema
        )

        assert session.execute.call_count == 1
        executed = session.execute.call_args[0][0]
        assert isinstance(executed, RefreshMaterializedView)
        assert executed.name == "my_mv"
        assert executed.schema == schema
        assert executed.concurrently is concurrently

        engine = sa.create_engine("sqlite:///:memory:")
        compiled = str(executed.compile(dialect=engine.dialect))
        assert expected_substring in compiled


# ===========================================================================
# Schema resolution (ViewMixin)
# ===========================================================================

class TestSchemaResolution:

    def test_refresh_uses_resolved_schema_from_table_args(self):
        """refresh() resolves schema from __table_args__ when __view_schema__ unset."""

        class AnalyticsView(ViewMixin):
            __tablename__ = "analytics_mv"
            __view_selectable__ = sa.select(sa.column("id", sa.Integer))
            __view_materialized__ = True
            __table_args__ = {"schema": "analytics"}
            metadata = sa.MetaData()
            id = sa.Column(sa.Integer, primary_key=True)

        session = mock.MagicMock(name="session")
        with mock.patch(
            "sqlalchemy_utils.view_mixin.refresh_materialized_view"
        ) as mock_refresh:
            AnalyticsView.refresh(session)

        mock_refresh.assert_called_once()
        _, kwargs = mock_refresh.call_args
        assert kwargs.get("schema") == "analytics"

    def test_view_mixin_without_table_args(self):
        """ViewMixin without __table_args__ resolves schema to None."""
        Base = sa.orm.declarative_base()

        class SimpleView(ViewMixin, Base):
            __tablename__ = "simple_view"
            __view_selectable__ = sa.select(sa.column("id", sa.Integer))
            id: "Mapped[int]" = sa.Column(sa.Integer, primary_key=True)

        SimpleView.__declare_last__()
        assert SimpleView._resolve_schema() is None
        assert SimpleView.__table__ is not None
        assert SimpleView.__table__.name == "simple_view"

    def test_without_tablename_raises_helpful_error(self):
        """Missing __tablename__ raises a view-specific error."""
        Base = sa.orm.declarative_base()

        with pytest.raises(Exception) as exc_info:

            class NoTablenameThing(ViewMixin, Base):
                __view_selectable__ = sa.select(sa.column("id", sa.Integer))
                id: "Mapped[int]" = sa.Column(sa.Integer, primary_key=True)

            NoTablenameThing.__declare_last__()

        err_msg = str(exc_info.value).lower()
        assert "__view_selectable__" in err_msg or "viewmixin" in err_msg


# ===========================================================================
# ViewMixin integration
# ===========================================================================

class TestViewMixinIntegration:

    def test_replace_attr_passed_to_create_view(self):
        """__view_replace__=True produces CREATE OR REPLACE VIEW DDL."""
        Base = declarative_base()

        class ReplaceView(ViewMixin, Base):
            __tablename__ = "replace_view"
            __view_selectable__ = sa.select(
                sa.table("src", sa.column("id", sa.Integer))
            )
            __view_replace__ = True
            id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

        ReplaceView.__declare_last__()

        create_view_ddl = _find_view_listener(Base.metadata)
        assert create_view_ddl is not None
        assert create_view_ddl.replace is True

        captured = _ddl_sql_for_metadata(Base.metadata)
        assert any(
            "CREATE OR REPLACE VIEW" in stmt.upper() for stmt in captured
        )

    def test_view_replace_is_class_attribute(self):
        """ViewMixin defines __view_replace__ as a class attribute defaulting False."""
        assert hasattr(ViewMixin, "__view_replace__")
        assert ViewMixin.__view_replace__ is False


class TestViewAutoRegistration:

    def test_create_view_registers_view_record_in_metadata(self):
        metadata = sa.MetaData()

        selectable = select(Column("id", Integer), Column("name", sa.String))
        create_view("test_view", selectable, metadata)

        assert "sqlalchemy_utils_views" in metadata.info
        assert len(metadata.info["sqlalchemy_utils_views"]) == 1
        record = metadata.info["sqlalchemy_utils_views"][0]
        assert isinstance(record, ViewRecord)
        assert record.name == "test_view"
        assert record.selectable is selectable
        assert record.schema is None
        assert record.materialized is False
        assert record.replace is False
        assert record.cascade_on_drop is True

    def test_create_materialized_view_registers_view_record_with_materialized_true(self):
        metadata = sa.MetaData()

        selectable = select(Column("id", Integer), Column("name", sa.String))
        create_materialized_view("test_mv", selectable, metadata, indexes=[])

        assert "sqlalchemy_utils_views" in metadata.info
        assert len(metadata.info["sqlalchemy_utils_views"]) == 1
        record = metadata.info["sqlalchemy_utils_views"][0]
        assert isinstance(record, ViewRecord)
        assert record.name == "test_mv"
        assert record.selectable is selectable
        assert record.schema is None
        assert record.materialized is True
        assert record.replace is False
        assert record.cascade_on_drop is True

    def test_multiple_create_view_calls_append_multiple_records(self):
        metadata = sa.MetaData()

        selectable1 = select(Column("id", Integer))
        create_view("view1", selectable1, metadata)

        selectable2 = select(Column("id", Integer), Column("name", sa.String))
        create_view("view2", selectable2, metadata)

        assert len(metadata.info["sqlalchemy_utils_views"]) == 2
        view1 = metadata.info["sqlalchemy_utils_views"][0]
        view2 = metadata.info["sqlalchemy_utils_views"][1]
        assert view1.name == "view1"
        assert view2.name == "view2"
        assert view2.materialized is False

    def test_materialized_and_non_materialized_views_separate(self):
        metadata = sa.MetaData()

        regular_selectable = select(Column("id", Integer))
        mv_selectable = select(Column("id", Integer))

        create_view("regular_view", regular_selectable, metadata)
        create_materialized_view("mv_view", mv_selectable, metadata, indexes=[])

        assert len(metadata.info["sqlalchemy_utils_views"]) == 2

        records = metadata.info["sqlalchemy_utils_views"]
        regular_record = next(r for r in records if r.name == "regular_view")
        mv_record = next(r for r in records if r.name == "mv_view")

        assert regular_record.materialized is False
        assert mv_record.materialized is True

    def test_create_view_with_replace_parameter(self):
        metadata = sa.MetaData()

        selectable = select(Column("id", Integer))
        create_view("replace_view", selectable, metadata, replace=True)

        assert len(metadata.info["sqlalchemy_utils_views"]) == 1
        record = metadata.info["sqlalchemy_utils_views"][0]
        assert record.replace is True

    def test_create_view_with_cascade_on_drop_parameter(self):
        metadata = sa.MetaData()

        selectable = select(Column("id", Integer))
        create_view("no_cascade_view", selectable, metadata, cascade_on_drop=False)

        assert len(metadata.info["sqlalchemy_utils_views"]) == 1
        record = metadata.info["sqlalchemy_utils_views"][0]
        assert record.cascade_on_drop is False

    def test_default_cascade_on_drop_true(self):
        metadata = sa.MetaData()

        selectable = select(Column("id", Integer))
        create_materialized_view("mv_default", selectable, metadata, indexes=[])

        assert len(metadata.info["sqlalchemy_utils_views"]) == 1
        record = metadata.info["sqlalchemy_utils_views"][0]
        assert record.cascade_on_drop is True


# ===========================================================================
# Interface audit fixes
# ===========================================================================

def test_create_view_works_without_alembic_installed():
    """create_view() must not transitively import alembic, so it works
    when alembic is absent."""
    import subprocess, sys
    code = (
        'import sys\n'
        "sys.modules['alembic'] = None\n"
        "sys.modules['alembic.operations'] = None\n"
        "sys.modules['alembic.autogenerate'] = None\n"
        'import sqlalchemy as sa\n'
        'from sqlalchemy_utils.view import create_view\n'
        'metadata = sa.MetaData()\n'
        'create_view("v", sa.select(sa.column("id", sa.Integer)), metadata)\n'
        'print("SUCCESS")\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, text=True,
        env={'PYTHONPATH': 'src', 'PATH': ''},
    )
    assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    assert 'SUCCESS' in result.stdout


def test_op_drop_materialized_view_accepts_with_data():
    """op.drop_materialized_view should accept with_data for round-trip fidelity."""
    operations = MagicMock()
    invoked: list = []
    operations.invoke.side_effect = lambda op: invoked.append(op) or op
    DropMaterializedViewOp.drop_materialized_view(
        operations, "mv", definition="SELECT 1", with_data=False
    )
    operations.invoke.assert_called_once()
    invoked_op = operations.invoke.call_args[0][0]
    assert invoked_op.with_data is False


# ===========================================================================
# Cross-schema dedup and ordering
# ===========================================================================


class TestCrossSchemaSameNameSeparation:
    """Cross-schema same-name views must produce separate graph nodes.

    Previously the graph was keyed by bare ``vr.name``, so two views named
    ``summary`` in schemas ``reporting`` and ``analytics`` collapsed into a
    single graph node — the second view's deps overwrote the first's,
    yielding wrong dependency ordering.
    """

    def test_resolve_create_order_preserves_same_name_diff_schema(self):
        """Two views with the same name in different schemas must both
        appear in the ordered output."""
        views = [
            ViewRecord(name="v", schema="a", selectable="SELECT 1"),
            ViewRecord(name="v", schema="b", selectable="SELECT 1"),
        ]
        result = resolve_create_order(views, db_views={})
        assert len(result) == 2

    def test_build_dependency_graph_merges_deps_same_name_diff_schema(self):
        """When two ViewRecords share a name across schemas, their
        dependencies must be on SEPARATE graph nodes keyed by
        ``(name, schema)`` rather than collapsed into one bare-name node."""
        views = [
            ViewRecord(name="foo", schema="a", selectable="SELECT * FROM bar"),
            ViewRecord(name="foo", schema="b", selectable="SELECT * FROM baz"),
            ViewRecord(name="bar", selectable="SELECT 1"),
            ViewRecord(name="baz", selectable="SELECT 1"),
        ]
        graph = _build_dependency_graph(views, db_views={})
        # Each (name, schema) pair is a distinct graph node.
        assert graph[("foo", "a")] == {("bar", None)}
        assert graph[("foo", "b")] == {("baz", None)}
        # No bare-name keys should exist.
        assert "foo" not in graph

    def test_graph_has_two_distinct_nodes(self):
        views = [
            ViewRecord(
                name="summary",
                schema="reporting",
                selectable="SELECT 1 AS col",
            ),
            ViewRecord(
                name="summary",
                schema="analytics",
                selectable="SELECT * FROM summary",
            ),
        ]
        graph = _build_dependency_graph(views, db_views={})
        assert len(graph) == 2
        assert ("summary", "reporting") in graph
        assert ("summary", "analytics") in graph


# ===========================================================================
# Interface audit: migration refresh, dead listener, mapped_column guard
# ===========================================================================

class TestInterfaceAuditFixes:

    def test_refresh_reverse_raises(self):
        """RefreshMaterializedViewOp.reverse() raises NotImplementedError.

        REFRESH MATERIALIZED VIEW is not meaningfully reversible (you
        cannot "un-refresh" a materialized view), so reverse() must
        raise rather than silently emit another REFRESH in the downgrade.
        """
        op = RefreshMaterializedViewOp("mv")
        with pytest.raises(NotImplementedError, match="not meaningfully reversible"):
            op.reverse()

    def test_viewmixin_init_subclass_catches_mapped_column_without_tablename(self):
        """ViewMixin.__init_subclass__ should catch mapped_column usage without __tablename__ and give a helpful error."""
        Base = declarative_base()
        with pytest.raises(TypeError, match="__tablename__"):
            class BadView(ViewMixin, Base):
                id: Mapped[int] = mapped_column(primary_key=True)
                __view_selectable__ = sa.select(sa.column("id", sa.Integer))


# ===========================================================================
# Cascade-on-drop warning
# ===========================================================================

class TestCascadeOnDropWarning:

    @pytest.fixture
    def cascade_mock_setup(self):

        def _run(db_views: dict, dependent_views: dict):
            autogen_context, upgrade_ops = _make_mock_autogen_context(
                model_views=[]
            )

            with _patch_comparator(
                db_views=db_views,
                canonical_return=({}, {}, set()),
                dependent_views=dependent_views,
            ) as mock_log:
                comparator_module.compare_views(
                    autogen_context, upgrade_ops, [None]
                )
            return upgrade_ops, mock_log

        return _run

    def test_warns_when_dropping_view_with_dependents(self, cascade_mock_setup):
        """compare_views should log a warning when dropping a view that
        other views depend on."""
        upgrade_ops, mock_log = cascade_mock_setup(
            db_views={"base_view": "SELECT 1 AS col"},
            dependent_views={("dependent_view", None): "SELECT * FROM base_view"},
        )

        warning_calls = [
            c for c in mock_log.warning.call_args_list
            if "base_view" in str(c) and "dependent" in str(c).lower()
        ]
        assert len(warning_calls) > 0, f"got {mock_log.warning.call_args_list}"

    def test_does_not_warn_when_dropping_view_without_dependents(
        self, cascade_mock_setup
    ):
        """compare_views should NOT warn when dropping a view with no dependents."""
        upgrade_ops, mock_log = cascade_mock_setup(
            db_views={"lonely_view": "SELECT 1 AS col"},
            dependent_views={},
        )

        warning_calls = [
            c for c in mock_log.warning.call_args_list
            if "dependent" in str(c).lower()
        ]
        assert len(warning_calls) == 0, f"got: {warning_calls}"


# ===========================================================================
# cascade_on_drop propagation
# ===========================================================================

class TestCascadeOnDropPropagation:
    """compare_views should propagate ViewRecord.cascade_on_drop to the
    generated DropViewOp / DropMaterializedViewOp ``cascade`` param."""

    @pytest.fixture
    def run_compare(self):
        """Run compare_views with one model ViewRecord and patched DB state.

        Returns the ``upgrade_ops`` for op-type filtering and assertions.
        """
        def _run(
            view_record,
            db_views=None,
            db_mvs=None,
            canonical_return=None,
        ):
            model_views = [view_record] if view_record else []
            autogen_context, upgrade_ops = _make_mock_autogen_context(
                model_views=model_views
            )

            kwargs = {}
            if db_views is not None:
                kwargs["db_views"] = db_views
            if db_mvs is not None:
                kwargs["db_mvs"] = db_mvs
            if canonical_return is not None:
                kwargs["canonical_return"] = canonical_return

            with _patch_comparator(**kwargs):
                comparator_module.compare_views(
                    autogen_context, upgrade_ops, [None]
                )
            return upgrade_ops

        return _run

    def test_drop_view_propagates_cascade_false(self, run_compare):
        """DropViewOp.cascade=False when ViewRecord.cascade_on_drop=False."""
        vr = ViewRecord(
            name="v_no_cascade", selectable="SELECT 1 AS col",
            schema=None, cascade_on_drop=False,
        )
        upgrade_ops = run_compare(
            vr, db_views={"v_no_cascade": "SELECT 1 AS col"},
        )

        drop_ops = [op for op in upgrade_ops.ops if isinstance(op, DropViewOp)]
        assert len(drop_ops) == 1, f"expected one DropViewOp, got {drop_ops}"
        assert drop_ops[0].name == "v_no_cascade"
        assert drop_ops[0].cascade is False, f"got {drop_ops[0].cascade!r}"

    def test_drop_view_defaults_to_true_when_no_record(self, run_compare):
        """DropViewOp.cascade defaults to True when no model ViewRecord exists."""
        upgrade_ops = run_compare(
            None, db_views={"orphan_view": "SELECT 1 AS col"},
        )

        drop_ops = [op for op in upgrade_ops.ops if isinstance(op, DropViewOp)]
        assert len(drop_ops) == 1
        assert drop_ops[0].cascade is True

    def test_drop_materialized_view_propagates_cascade_false(self, run_compare):
        """DropMaterializedViewOp.cascade=False when cascade_on_drop=False."""
        vr = ViewRecord(
            name="mv_no_cascade", selectable="SELECT 1 AS col",
            schema=None, materialized=True, cascade_on_drop=False,
        )
        upgrade_ops = run_compare(
            vr, db_views={}, db_mvs={"mv_no_cascade": "SELECT 1 AS col"},
        )

        drop_ops = [
            op for op in upgrade_ops.ops
            if isinstance(op, DropMaterializedViewOp)
        ]
        assert len(drop_ops) == 1, f"got {drop_ops}"
        assert drop_ops[0].name == "mv_no_cascade"
        assert drop_ops[0].cascade is False, f"got {drop_ops[0].cascade!r}"

    def test_create_view_propagates_cascade_false(self, run_compare):
        """CreateViewOp.cascade_on_drop=False when ViewRecord.cascade_on_drop
        is False and the view is absent from the DB (Create path, not Drop)."""
        vr = ViewRecord(
            name="v_new_nocascade", selectable="SELECT 1 AS col",
            schema=None, cascade_on_drop=False,
        )
        upgrade_ops = run_compare(
            vr,
            db_views={},
            canonical_return=(
                {"v_new_nocascade": "SELECT 1 AS col"}, {}, set(),
            ),
        )

        create_ops = [
            op for op in upgrade_ops.ops if isinstance(op, CreateViewOp)
        ]
        assert len(create_ops) == 1, f"got {create_ops}"
        assert create_ops[0].name == "v_new_nocascade"
        assert create_ops[0].cascade_on_drop is False, f"got {create_ops[0].cascade_on_drop!r}"

    def test_replace_view_propagates_cascade_false(self, run_compare):
        """ReplaceViewOp.cascade=False when ViewRecord.cascade_on_drop=False
        and the view definition changed (Replace path, not Create/Drop)."""
        vr = ViewRecord(
            name="v_replace_nocascade",
            selectable="SELECT 1 AS col",
            schema=None,
            cascade_on_drop=False,
        )
        upgrade_ops = run_compare(
            vr,
            db_views={"v_replace_nocascade": "SELECT 2 AS col"},
            canonical_return=(
                {"v_replace_nocascade": "SELECT 1 AS col"}, {}, set(),
            ),
        )

        replace_ops = [
            op for op in upgrade_ops.ops if isinstance(op, ReplaceViewOp)
        ]
        assert len(replace_ops) == 1, f"got {replace_ops}"
        assert replace_ops[0].name == "v_replace_nocascade"
        assert replace_ops[0].cascade is False, f"got {replace_ops[0].cascade!r}"

    def test_replace_view_defaults_to_cascade_true(self, run_compare):
        """ReplaceViewOp.cascade defaults to True when ViewRecord has
        no cascade_on_drop preference set (model view present, DB differs)."""
        vr = ViewRecord(
            name="v_replace_default",
            selectable="SELECT 1 AS col",
            schema=None,
        )
        upgrade_ops = run_compare(
            vr,
            db_views={"v_replace_default": "SELECT 2 AS col"},
            canonical_return=(
                {"v_replace_default": "SELECT 1 AS col"}, {}, set(),
            ),
        )

        replace_ops = [
            op for op in upgrade_ops.ops if isinstance(op, ReplaceViewOp)
        ]
        assert len(replace_ops) == 1
        assert replace_ops[0].cascade is True


# ===========================================================================
# Cross-schema same-name view handling
# ===========================================================================

class TestCrossSchemaSameNameBothOps:
    """When two schemas each have a model view with the same name, both
    create ops must survive — the second must not overwrite the first
    in the ``create_by_name`` / ``drop_by_name`` dicts.
    """

    def test_cross_schema_same_name_both_ops(self):

        # Two model views, same name, different schemas
        model_views = [
            ViewRecord(
                name="foo",
                selectable="SELECT 1 AS id",
                schema="public",
            ),
            ViewRecord(
                name="foo",
                selectable="SELECT 2 AS id",
                schema="analytics",
            ),
        ]

        autogen_context, upgrade_ops = _make_mock_autogen_context(
            model_views=model_views
        )

        # DB has no views in either schema → both model views become creates.
        # _canonicalize_all_views is called once per schema; return a
        # distinct definition per call so the created ops are distinguishable.
        canonical_returns = iter(
            [
                ({"foo": "SELECT 1 AS id"}, {}, set()),
                ({"foo": "SELECT 2 AS id"}, {}, set()),
            ]
        )

        def mock_canonicalize_all(connection, view_records, db_views):
            return next(canonical_returns)

        with patch.object(
            comparator_module, "get_database_views", return_value={}
        ), patch.object(
            comparator_module, "get_database_materialized_views", return_value={}
        ), patch.object(
            comparator_module, "_canonicalize_all_views",
            side_effect=mock_canonicalize_all,
        ), patch.object(
            comparator_module, "get_dependent_views", return_value={}
        ), patch.object(comparator_module, "log"):
            comparator_module.compare_views(
                autogen_context, upgrade_ops, ["public", "analytics"]
            )

        create_ops = [
            op for op in upgrade_ops.ops if isinstance(op, CreateViewOp)
        ]
        # Both ops must survive — one per schema
        assert len(create_ops) == 2, f"got {len(create_ops)}: {[(op.name, op.schema) for op in create_ops]}"

        schemas_seen = {(op.name, op.schema) for op in create_ops}
        assert ("foo", "public") in schemas_seen, f"got {schemas_seen}"
        assert ("foo", "analytics") in schemas_seen, f"got {schemas_seen}"


# ===========================================================================
# Regression: schema=None asymmetric comparison
# ===========================================================================

class TestSchemaNoneNoFalseDrop:

    def test_schema_none_no_false_drop(self):
        from types import SimpleNamespace

        model_views = [
            ViewRecord(
                name="model_view",
                selectable="SELECT 1 AS id",
                schema="public",
            ),
        ]

        public_row = SimpleNamespace(
            viewname="model_view", definition="SELECT 1 AS id"
        )
        analytics_row = SimpleNamespace(
            viewname="analytics_view", definition="SELECT 2 AS id"
        )

        def mock_execute(sql, params=None):
            sql_str = str(sql)
            if "current_schema()" in sql_str:
                return [public_row]
            if "NOT IN" in sql_str:
                return [public_row, analytics_row]
            return []

        connection = MagicMock()
        connection.dialect.name = "postgresql"
        connection.execute = mock_execute

        metadata = MagicMock()
        metadata.info = {"sqlalchemy_utils_views": model_views}

        autogen_context = MagicMock()
        autogen_context.connection = connection
        autogen_context.metadata = metadata

        upgrade_ops = MagicMock()
        upgrade_ops.ops = []

        with patch.object(
            comparator_module, "get_database_materialized_views",
            return_value={},
        ), patch.object(
            comparator_module, "_canonicalize_all_views",
            return_value=({"model_view": "SELECT 1 AS id"}, {}, set()),
        ), patch.object(
            comparator_module, "get_dependent_views",
            return_value={},
        ), patch.object(comparator_module, "log"):
            comparator_module.compare_views(
                autogen_context, upgrade_ops, [None]
            )

        drop_ops = [
            op for op in upgrade_ops.ops if isinstance(op, DropViewOp)
        ]
        analytics_drops = [
            op for op in drop_ops if op.name == "analytics_view"
        ]
        assert analytics_drops == [], f"got drop ops: {[(op.name, op.schema) for op in drop_ops]}"


# ===========================================================================
# Regression: MV canonicalization DROP must use CASCADE for dependent views
# ===========================================================================

_MV_CASCADE_TEST_VIEW_NAMES = ["mv_cascade_test_mv", "mv_cascade_test_dep_view"]


class TestMvCanonicalizationCascade:
    """Regression: MV canonicalization DROP must use CASCADE.

    When a materialized view has dependent views in the database, PG refuses
    a plain ``DROP MATERIALIZED VIEW``. Without CASCADE the DROP fails inside
    the canonicalization savepoint, the MV is added to ``skipped``, and its
    definition change is silently missed (no ``ReplaceMaterializedViewOp``
    emitted). Using ``DROP MATERIALIZED VIEW IF EXISTS ... CASCADE`` lets the
    DROP succeed inside the savepoint (dependents are cascade-dropped and
    restored on savepoint rollback), so the MV is canonicalized correctly.
    """

    @pytest.mark.parametrize(
        "materialized, view_name, drop_clause, create_clause, create_extra",
        [
            (
                True,
                "mv_cascade_test_mv",
                "DROP MATERIALIZED VIEW IF EXISTS",
                "CREATE MATERIALIZED VIEW",
                "WITH NO DATA",
            ),
            (
                False,
                "v_cascade_test",
                "DROP VIEW IF EXISTS",
                "CREATE VIEW",
                "",
            ),
        ],
        ids=["materialized_view", "regular_view"],
    )
    def test_build_create_sql_returns_drop_then_create_list_with_cascade(
        self, materialized, view_name, drop_clause, create_clause, create_extra,
    ):
        """``_build_create_sql`` must return a list of two SQL strings
        (DROP+CREATE) with CASCADE on the DROP so dependent views do not
        block canonicalization. Applies to both materialized and regular
        views (regular views use DROP+CREATE because ``CREATE OR REPLACE
        VIEW`` fails on column-structure changes)."""
        connection = MagicMock()
        connection.dialect = sa.dialects.postgresql.dialect()
        with patch(
            "sqlalchemy_utils.alembic.comparator.ViewRecord.compiled_definition",
            return_value="SELECT 1 AS id",
        ):
            vr = ViewRecord(
                name=view_name,
                selectable="SELECT 1 AS id",
                schema=None,
                materialized=materialized,
            )
            stmts = _build_create_sql(connection, vr)

        assert isinstance(stmts, list), f"got {type(stmts)}"
        assert len(stmts) == 2, f"got {stmts!r}"
        drop_sql = stmts[0].upper()
        create_sql = stmts[1].upper()
        assert drop_clause.upper() in drop_sql, f"got {stmts[0]!r}"
        assert "CASCADE" in drop_sql, f"got {stmts[0]!r}"
        assert create_clause.upper() in create_sql, f"got {stmts[1]!r}"
        if create_extra:
            assert create_extra.upper() in create_sql, f"got {stmts[1]!r}"
        assert "OR REPLACE" not in create_sql, f"got {stmts[1]!r}"

    @pytest.mark.infrastructure
    @pytest.mark.usefixtures("postgresql_dsn")
    def test_mv_with_dependent_view_definition_change_detected(self, view_cleanup_factory):
        """MV with a dependent view: definition change is detected, not skipped.

        Pre-create an MV with an old definition and a regular view that
        selects from it. The model defines the MV with a NEW definition.
        Without CASCADE the DROP fails (dependent view blocks it), the MV is
        skipped, and the change is silently missed. With CASCADE the DROP
        succeeds inside the savepoint, the MV is recreated, and the change is
        detected as a ``ReplaceMaterializedViewOp``.
        """
        connection = view_cleanup_factory(_MV_CASCADE_TEST_VIEW_NAMES, create_base=True)
        # Pre-create MV with OLD definition.
        connection.execute(
            sa.text(
                "CREATE MATERIALIZED VIEW mv_cascade_test_mv AS "
                "SELECT id, name FROM _cmp_test_base WITH DATA"
            )
        )
        # Pre-create a dependent regular view on the MV.
        connection.execute(
            sa.text(
                "CREATE VIEW mv_cascade_test_dep_view AS "
                "SELECT * FROM mv_cascade_test_mv"
            )
        )
        connection.commit()

        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="mv_cascade_test_mv",
                selectable="SELECT id, value FROM _cmp_test_base",
                schema=None,
                materialized=True,
            ),
            ViewRecord(
                name="mv_cascade_test_dep_view",
                selectable=sa.select(
                    sa.text("* FROM mv_cascade_test_mv")
                ),
                schema=None,
            ),
        ]

        upgrade_ops = _run_comparator(connection, metadata, schemas=[None])

        replace_ops = [
            op
            for op in upgrade_ops.ops
            if isinstance(op, ReplaceMaterializedViewOp)
            and op.name == "mv_cascade_test_mv"
        ]
        assert len(replace_ops) == 1, f"got {replace_ops}; all ops: {[(type(o).__name__, o.name) for o in upgrade_ops.ops]}"

    @pytest.mark.infrastructure
    @pytest.mark.usefixtures("postgresql_dsn")
    def test_mv_with_dependent_view_not_in_skipped(self, view_cleanup_factory):
        """Directly verify ``_canonicalize_all_views`` does not skip an MV
        that has a dependent view in the database."""
        connection = view_cleanup_factory(_MV_CASCADE_TEST_VIEW_NAMES, create_base=True)
        connection.execute(
            sa.text(
                "CREATE MATERIALIZED VIEW mv_cascade_test_mv AS "
                "SELECT id, name FROM _cmp_test_base WITH DATA"
            )
        )
        connection.execute(
            sa.text(
                "CREATE VIEW mv_cascade_test_dep_view AS "
                "SELECT * FROM mv_cascade_test_mv"
            )
        )
        connection.commit()

        view_records = [
            ViewRecord(
                name="mv_cascade_test_mv",
                selectable="SELECT id, value FROM _cmp_test_base",
                schema=None,
                materialized=True,
            ),
            ViewRecord(
                name="mv_cascade_test_dep_view",
                selectable=sa.select(
                    sa.text("* FROM mv_cascade_test_mv")
                ),
                schema=None,
            ),
        ]
        db_views_for_deps = {
            "mv_cascade_test_mv": "SELECT id, name FROM _cmp_test_base",
            "mv_cascade_test_dep_view": "SELECT * FROM mv_cascade_test_mv",
        }

        view_defs, mv_defs, skipped = _canonicalize_all_views(
            connection, view_records, db_views_for_deps
        )

        assert "mv_cascade_test_mv" not in skipped, f"skipped={skipped}"
        assert "mv_cascade_test_mv" in mv_defs, f"mv_defs={mv_defs}, skipped={skipped}"


# ===========================================================================
# Dedup guard: non-view ops must pass through untouched
# ===========================================================================

class TestDedupPreservesNonViewOps:
    """The dedup loop at the end of ``compare_views`` must only process
    view ops. Built-in Alembic ops (CreateTableOp, DropTableOp, etc.) lack
    a ``name`` attribute, so ``getattr(op, "name", None)`` returns ``None``
    for all of them. Without a guard, every non-view op produces an
    identical dedup key ``("create_or_replace"|"drop", None, None)`` and all
    but the first are silently discarded — losing table migrations.
    """

    @staticmethod
    def _make_table_ops(spec):
        from alembic.operations.ops import CreateTableOp, DropTableOp
        from sqlalchemy import Column, Integer, MetaData, Table

        md = MetaData()
        ops = []
        for kind, name in spec:
            t = Table(name, md, Column("id", Integer))
            if kind == "create":
                ops.append(CreateTableOp.from_table(t))
            else:
                ops.append(DropTableOp.from_table(t))
        return ops

    @pytest.mark.parametrize(
        "existing_ops_spec, db_views, expected_op_types",
        [
            (
                [("create", "table_a"), ("drop", "table_b")],
                {},
                ["CreateTableOp", "DropTableOp"],
            ),
            (
                [("create", f"t{i}") for i in range(5)],
                {},
                ["CreateTableOp"] * 5,
            ),
            (
                [("create", "real_table")],
                {"old_view": "SELECT 1 AS col"},
                ["CreateTableOp", "DropViewOp"],
            ),
        ],
        ids=["create_plus_drop", "five_creates", "mixed_with_view_drop"],
    )
    def test_non_view_ops_survive_dedup(
        self, existing_ops_spec, db_views, expected_op_types
    ):
        existing_ops = self._make_table_ops(existing_ops_spec)
        autogen_context, upgrade_ops = _make_mock_autogen_context(
            existing_ops=existing_ops
        )

        patches = _patch_comparator(db_views=db_views)
        with patches:
            compare_views(autogen_context, upgrade_ops, [None])

        actual_types = [type(op).__name__ for op in upgrade_ops.ops]
        for expected_type in expected_op_types:
            assert actual_types.count(expected_type) >= expected_op_types.count(
                expected_type
            ), f"got {actual_types}"


# ===========================================================================
# View type change (regular <-> materialized) drop-before-create ordering
# ===========================================================================

class TestViewTypeChangeOrdering:
    """When a view changes from regular to materialized (or vice versa),
    ``compare_views`` emits a Drop for the old type and a Create for the
    new type. The Drop MUST come before the Create so the migration does
    not fail trying to CREATE while the old-type view still exists.
    """

    def test_regular_to_materialized_drops_before_creates(self):

        # DB has the view as a regular view; model defines it as materialized.
        # _canonicalize_all_views is called per schema. For schema=None the
        # model materialized view is canonicalized → mv_defs={"v": "..."}.
        autogen_context, upgrade_ops = _make_mock_autogen_context(
            model_views=[
                ViewRecord(
                    name="v",
                    selectable="SELECT 1 AS id",
                    schema=None,
                    materialized=True,
                ),
            ]
        )

        with patch.object(
            comparator_module, "get_database_views",
            return_value={"v": "SELECT 1 AS id"},
        ), patch.object(
            comparator_module, "get_database_materialized_views",
            return_value={},
        ), patch.object(
            comparator_module, "_canonicalize_all_views",
            return_value=({}, {"v": "SELECT 1 AS id"}, set()),
        ), patch.object(
            comparator_module, "get_dependent_views", return_value={}
        ), patch.object(comparator_module, "log"):
            comparator_module.compare_views(
                autogen_context, upgrade_ops, [None]
            )

        # Find indices of the drop (DropViewOp) and create
        # (CreateMaterializedViewOp) for the same view name.
        drop_idx = next(
            (i for i, op in enumerate(upgrade_ops.ops)
             if isinstance(op, DropViewOp) and op.name == "v"),
            None,
        )
        create_idx = next(
            (i for i, op in enumerate(upgrade_ops.ops)
             if isinstance(op, CreateMaterializedViewOp) and op.name == "v"),
            None,
        )
        assert drop_idx is not None, f"ops={[(type(o).__name__, o.name) for o in upgrade_ops.ops]}"
        assert create_idx is not None, f"ops={[(type(o).__name__, o.name) for o in upgrade_ops.ops]}"
        assert drop_idx < create_idx, f"got drop_idx={drop_idx}, create_idx={create_idx}; ops={[(type(o).__name__, o.name) for o in upgrade_ops.ops]}"


# ===========================================================================
# Cross-type reordering: reverse-dependency drop order for dependent views
# ===========================================================================

class TestCrossTypeReorderDependentViewsDropOrder:
    """When two dependent views (A depends on B) both change type
    simultaneously, drops must be emitted in reverse dependency order
    (A before B), not in create-order (B before A).

    Creates are ordered by ``resolve_create_order`` (dependencies first):
    B's create comes before A's create. The old per-key reorder emitted
    each drop at its create's position, producing B-before-A drops —
    which fails with ``cascade=False`` because dropping B while A still
    references it is refused by PG.

    The fix emits ALL cross-type drops (in their existing drop-order from
    ``resolve_drop_order``, which is dependents-first) before ANY
    cross-type creates, so drops come out A-before-B (correct) and
    creates come out B-before-A (unchanged).
    """

    def test_dependent_views_drops_in_reverse_dependency_order(self):
        """Given two views A (depends on B) both changing regular→materialized,
        DropViewOp(A) must appear before DropViewOp(B) in the ops list,
        and CreateMaterializedViewOp(B) before CreateMaterializedViewOp(A)."""
        # A depends on B (A references B in its definition).
        # Creates in dep-order (B first, A second):
        create_mv_b = CreateMaterializedViewOp("b", "SELECT 1 AS id")
        create_mv_a = CreateMaterializedViewOp(
            "a", "SELECT * FROM b"
        )
        # Drops in drop-order (A first, B second — dependents before deps):
        drop_view_a = DropViewOp("a", definition="SELECT * FROM b")
        drop_view_b = DropViewOp("b", definition="SELECT 1 AS id")

        # Comparator layout: create_ops (dep-order) extended before drop_ops
        # (drop-order).
        ops = [create_mv_b, create_mv_a, drop_view_a, drop_view_b]

        result = _reorder_cross_type_drops_before_creates(list(ops))

        names_and_types = [(type(o).__name__, o.name) for o in result]

        # Drops must be in reverse dependency order: A before B.
        drop_a_idx = names_and_types.index(("DropViewOp", "a"))
        drop_b_idx = names_and_types.index(("DropViewOp", "b"))
        assert drop_a_idx < drop_b_idx, f"got indices {drop_a_idx}, {drop_b_idx}; ops={names_and_types!r}"

        # Creates must stay in dependency order: B before A.
        create_b_idx = names_and_types.index(
            ("CreateMaterializedViewOp", "b")
        )
        create_a_idx = names_and_types.index(
            ("CreateMaterializedViewOp", "a")
        )
        assert create_b_idx < create_a_idx, f"got indices {create_b_idx}, {create_a_idx}; ops={names_and_types!r}"

        # All drops must come before all creates.
        assert drop_b_idx < create_b_idx, f"ops={names_and_types!r}"


# ===========================================================================
# Cross-type reordering: deterministic order + dependency preservation
# ===========================================================================

class TestCrossTypeReorderDeterministic:
    """When multiple views change type simultaneously, the drop+create
    pairs emitted by ``_reorder_cross_type_drops_before_creates`` MUST
    preserve the original relative order from ``ops`` so dependent views
    are dropped before their dependencies and created after them.

    Iterating over a set (``cross_keys = create_keys & drop_keys``) yields
    non-deterministic order under Python hash randomization; this tests
    that the function instead walks ``ops`` in order.
    """

    def test_order_deterministic_across_runs(self):
        """The order of cross-type ops must be identical on every call.

        Because the bug is non-deterministic set iteration under hash
        randomization, this test asserts the result matches the
        expected order (i.e. the order from `ops`). When the set is
        iterated, the order would (probabilistically) vary; this test
        locks the contract that the function MUST walk `ops` in order.
        """
        names = [f"view_{i}" for i in range(2)]
        creates = [CreateMaterializedViewOp(n, "SELECT 1") for n in names]
        drops = [DropViewOp(n, definition="SELECT 1 AS id") for n in names]
        ops = creates + drops

        result = _reorder_cross_type_drops_before_creates(list(ops))

        result_names_in_order = [
            o.name for o in result
            if isinstance(o, (CreateMaterializedViewOp, DropViewOp))
        ]
        # All drops (in ops appearance order = names order) then all creates
        # (in ops appearance order = names order).
        expected_order = names + names

        assert result_names_in_order == expected_order, f"got {result_names_in_order!r}, expected {expected_order!r}"


class TestCrossTypeReorderPreservesDependencyOrder:
    """Cross-type reorder must emit all cross-type drops before all
    cross-type creates, at the position of the first cross-type op,
    so non-cross-type ops keep their relative order.

    Buffering to the end breaks dependency order: if view W depends on view V
    and V changes type, W's create (a non-cross-type op positioned after V's
    cross-type create) would end up BEFORE V's drop+create pair. When V's old
    type is dropped with CASCADE, W is cascade-dropped and lost.
    """

    def test_non_cross_type_op_stays_after_cross_type_pair(self):
        """A non-cross-type CreateViewOp for a dependent view W must NOT
        be reordered before the cross-type drop+create pair for view V.

        Setup: view V changes type (regular→materialized, cross-type).
        View W is a regular view that depends on V.
        Ordered ops input (as the comparator lays them out — creates
        extended before drops):
            [CreateMV(V), CreateView(W), DropView(V)]

        Expected after reorder: V's drop+create pair is inserted at the
        position where V FIRST appeared (index 0), and W's create stays
        in its original relative position after V's pair:
            [DropView(V), CreateMV(V), CreateView(W)]

        Bug: the old code buffers all cross-type ops to the END, producing
        [CreateView(W), DropView(V), CreateMV(V)] — W ends up before V's
        new type exists, so when V's old type is dropped (CASCADE), W is
        cascade-dropped and lost.
        """
        # V changes type (regular -> materialized): cross-type key.
        create_mv_v = CreateMaterializedViewOp("v", "SELECT 1 AS id")
        drop_view_v = DropViewOp("v", definition="SELECT 1 AS id")
        # W is a regular view that depends on V (not cross-type).
        create_view_w = CreateViewOp("w", "SELECT * FROM v")

        # Comparator layout: create_ops extended before drop_ops.
        ops = [create_mv_v, create_view_w, drop_view_v]

        result = _reorder_cross_type_drops_before_creates(list(ops))

        names_and_types = [(type(o).__name__, o.name) for o in result]
        # V's pair must be inserted at V's original first position (index 0),
        # and W's create must come AFTER V's pair (preserving dependency order).
        expected = [
            ("DropViewOp", "v"),
            ("CreateMaterializedViewOp", "v"),
            ("CreateViewOp", "w"),
        ]
        assert names_and_types == expected, f"got {names_and_types!r}, expected {expected!r}"

    def test_non_cross_type_op_between_two_cross_type_pairs_preserves_order(self):
        """A non-cross-type op sitting between two cross-type pairs must
        stay in its relative position (after all cross-type ops)."""
        create_mv_v = CreateMaterializedViewOp("v", "SELECT 1 AS id")
        create_mv_x = CreateMaterializedViewOp("x", "SELECT 3 AS id")
        create_view_w = CreateViewOp("w", "SELECT 2 AS id")
        drop_view_v = DropViewOp("v", definition="SELECT 1 AS id")
        drop_view_x = DropViewOp("x", definition="SELECT 3 AS id")

        ops = [create_mv_v, create_mv_x, create_view_w, drop_view_v, drop_view_x]

        result = _reorder_cross_type_drops_before_creates(list(ops))

        names_and_types = [(type(o).__name__, o.name) for o in result]
        # All cross-type drops (in ops appearance order) emitted at the first
        # cross-type op's position, then all cross-type creates, then the
        # non-cross-type op W (which appeared after the creates and before the
        # drops in the original ops list).
        expected = [
            ("DropViewOp", "v"),
            ("DropViewOp", "x"),
            ("CreateMaterializedViewOp", "v"),
            ("CreateMaterializedViewOp", "x"),
            ("CreateViewOp", "w"),
        ]
        assert names_and_types == expected, f"got {names_and_types!r}, expected {expected!r}"


class TestCrossTypeReorderPreservesNonCrossDropDependency:
    """When a view changes type AND another (non-cross-type) view being
    dropped depends on it, the non-cross-type drop must be emitted before
    the cross-type creates.

    The comparator emits creates before drops. ``_reorder_cross_type_drops_before_creates``
    currently extracts ONLY cross-type drops (drops whose (name, schema) is
    in ``cross_keys``) and moves them before the cross-type creates. A
    non-cross-type drop (a view being dropped that is NOT changing type)
    stays in its original position — after the cross-type creates. If that
    view depends on a cross-type view, it ends up dropped AFTER the
    cross-type view's new type is created, while the old type (which the
    dependent references) is already gone.

    The fix: when cross_keys is non-empty, extract ALL drops (not just
    cross-type drops) and emit them before ALL cross-type creates. Non-cross
    creates and other ops stay in their original positions.
    """

    def test_non_cross_drop_depends_on_cross_type_view_dropped_before_create(self):
        """View A changes type (regular→materialized). View B is a regular
        view being dropped that depends on A. B's DropViewOp must appear
        before A's CreateMaterializedViewOp, and before A's DropViewOp
        (B is a dependent, so it must be dropped first).
        """
        # A changes type regular→materialized (cross-type).
        create_mv_a = CreateMaterializedViewOp("a", "SELECT 1")
        drop_view_a = DropViewOp("a", definition="SELECT 1 AS id")
        # B is a regular view being dropped (non-cross-type), depends on A.
        drop_view_b = DropViewOp("b", definition="SELECT * FROM a")

        # Comparator layout: create_ops extended before drop_ops.
        ops = [create_mv_a, drop_view_b, drop_view_a]

        result = _reorder_cross_type_drops_before_creates(list(ops))

        names_and_types = [(type(o).__name__, o.name) for o in result]

        # B depends on A, so B must be dropped before A.
        drop_b_idx = names_and_types.index(("DropViewOp", "b"))
        drop_a_idx = names_and_types.index(("DropViewOp", "a"))
        assert drop_b_idx < drop_a_idx, (
            f"B must be dropped before A; got indices {drop_b_idx}, {drop_a_idx}; "
            f"ops={names_and_types!r}"
        )

        # All drops must come before all cross-type creates.
        create_a_idx = names_and_types.index(("CreateMaterializedViewOp", "a"))
        assert drop_a_idx < create_a_idx, (
            f"all drops must precede creates; got drop_a={drop_a_idx}, "
            f"create_a={create_a_idx}; ops={names_and_types!r}"
        )


# ===========================================================================
# CreateMaterializedViewOp cascade_on_drop
# ===========================================================================

class TestCreateMaterializedViewOpCascade:
    """CreateMaterializedViewOp must carry and honor a cascade_on_drop
    field, mirroring CreateViewOp. The reverse() must propagate it to
    the emitted DropMaterializedViewOp instead of hardcoding cascade=True.
    """

    def test_init_accepts_cascade_on_drop_false(self):
        op = CreateMaterializedViewOp(
            "mv", "SELECT 1", cascade_on_drop=False
        )
        assert op.cascade_on_drop is False

    def test_reverse_propagates_cascade_on_drop_false(self):
        """reverse() must produce a DropMaterializedViewOp with
        cascade=False when cascade_on_drop=False — not the hardcoded
        cascade=True."""
        op = CreateMaterializedViewOp(
            "mv", "SELECT 1", cascade_on_drop=False
        )
        rev = op.reverse()
        assert isinstance(rev, DropMaterializedViewOp)
        assert rev.cascade is False, f"got {rev.cascade!r}"

    def test_op_create_materialized_view_passes_cascade_on_drop(self):
        """op.create_materialized_view(..., cascade_on_drop=False) must
        produce a CreateMaterializedViewOp with cascade_on_drop=False."""
        operations = MagicMock()
        operations.invoke.return_value = None
        CreateMaterializedViewOp.create_materialized_view(
            operations, "mv", "SELECT 1", cascade_on_drop=False
        )
        operations.invoke.assert_called_once()
        invoked_op = operations.invoke.call_args[0][0]
        assert isinstance(invoked_op, CreateMaterializedViewOp)
        assert invoked_op.cascade_on_drop is False


# ===========================================================================
# op.create_view cascade_on_drop passthrough + kwarg ordering
# ===========================================================================

class TestCreateViewOpCascadePassthrough:
    """op.create_view must expose and forward cascade_on_drop, and the
    keyword-only ordering must place schema before cascade_on_drop (matching
    the other op.* helpers).
    """

    def test_op_create_view_accepts_cascade_on_drop_false(self):
        operations = MagicMock()
        operations.invoke.return_value = None
        CreateViewOp.create_view(
            operations, "v", "SELECT 1", cascade_on_drop=False
        )
        operations.invoke.assert_called_once()
        invoked_op = operations.invoke.call_args[0][0]
        assert isinstance(invoked_op, CreateViewOp)
        assert invoked_op.cascade_on_drop is False

    def test_op_create_view_signature_schema_before_cascade_on_drop(self):
        """The signature must be `*, schema=None,
        cascade_on_drop=True` — schema first, matching other op.* helpers."""
        sig = inspect.signature(CreateViewOp.create_view)
        params = list(sig.parameters.values())
        # classmethod: cls is stripped, so params are
        # [operations, name, definition, <keyword-only...>]. The keyword-only
        # params start after `definition` (index 3).
        kw_names = [p.name for p in params[3:]]
        assert kw_names == ["schema", "cascade_on_drop"], f"got {kw_names!r}"


# ===========================================================================
# Bug 1: CREATE OR REPLACE VIEW fails on column structure changes
# ===========================================================================

_COLUMN_CHANGE_VIEW_NAMES = ["col_change_view"]


class TestCanonicalizationColumnStructureChange:
    """Regression: canonicalization must detect column-structure changes.

    When a view exists in the DB with columns (id, name) and the model
    changes to columns (id, name, email), PG refuses ``CREATE OR REPLACE
    VIEW`` because the new column structure differs from the existing view.
    The old code used ``CREATE OR REPLACE VIEW`` for regular views during
    canonicalization; when it failed, the view was skipped and the
    definition change was silently missed (false negative — no
    ReplaceViewOp emitted).

    The fix uses DROP+CREATE (same pattern as materialized views), which
    succeeds inside the rolled-back savepoint and lets the definition
    change be detected.
    """

    @pytest.mark.infrastructure
    @pytest.mark.usefixtures("postgresql_dsn")
    def test_column_structure_change_detected_not_skipped(self, view_cleanup_factory):
        """End-to-end: a view with columns (id, name, value) in the DB changed
        to (id, name) in the model must emit a ReplaceViewOp.

        PG refuses ``CREATE OR REPLACE VIEW`` when the new view has FEWER
        columns than the existing view (``cannot drop columns from view``).
        With the old code the canonicalization failed, the view was skipped,
        and the change was silently missed. With DROP+CREATE the
        canonicalization succeeds and the change is detected.
        """
        connection = view_cleanup_factory(_COLUMN_CHANGE_VIEW_NAMES, create_base=True)
        # Pre-create view with OLD column structure (id, name, value) —
        # three columns. PG refuses CREATE OR REPLACE VIEW when the new
        # view has FEWER columns, so removing 'value' triggers the bug.
        connection.execute(
            sa.text(
                "CREATE VIEW col_change_view AS "
                "SELECT id, name, value FROM _cmp_test_base"
            )
        )
        connection.commit()

        # Model defines the view with NEW columns (id, name) — a column
        # REMOVAL that CREATE OR REPLACE VIEW refuses.
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="col_change_view",
                selectable="SELECT id, name FROM _cmp_test_base",
                schema=None,
            ),
        ]

        upgrade_ops = _run_comparator(connection, metadata, schemas=[None])

        replace_ops = [
            op
            for op in upgrade_ops.ops
            if isinstance(op, ReplaceViewOp) and op.name == "col_change_view"
        ]
        assert len(replace_ops) == 1, f"got {replace_ops}; all ops: {[(type(o).__name__, o.name) for o in upgrade_ops.ops]}"


_REPLACE_EXEC_VIEW_NAMES = ["replace_exec_view"]


class TestReplaceViewOpExecutionColumnStructureChange:
    """Regression: ``op.replace_view()`` execution must use DROP+CREATE.

    PG refuses ``CREATE OR REPLACE VIEW`` when the new view has FEWER
    columns than the existing view (``cannot drop columns from view``).
    The comparator already emits a ``ReplaceViewOp`` for column-structure
    changes; the execution path must also use DROP+CREATE so the
    generated migration is runnable. ``_replace_materialized_view_impl``
    already uses DROP+CREATE; this makes ``_replace_view_impl``
    consistent.
    """

    @pytest.mark.infrastructure
    @pytest.mark.usefixtures("postgresql_dsn")
    def test_replace_view_removing_column_succeeds(self, view_cleanup_factory):
        """End-to-end: replace a view removing a column must not raise.

        Pre-create a view with (id, name); then invoke
        ``ReplaceViewOp`` with a definition that selects only (id).
        With ``CREATE OR REPLACE VIEW`` PG raises
        ``cannot drop columns from view``. With DROP+CREATE the replace
        succeeds.
        """
        connection = view_cleanup_factory(_REPLACE_EXEC_VIEW_NAMES, create_base=True)
        connection.execute(
            sa.text(
                "CREATE VIEW replace_exec_view AS "
                "SELECT id, name FROM _cmp_test_base"
            )
        )
        connection.commit()

        op = ReplaceViewOp(
            "replace_exec_view",
            "SELECT id FROM _cmp_test_base",
        )
        ctx = MigrationContext.configure(connection)
        operations = Operations(ctx)
        operations.invoke(op)
        connection.commit()

        result = connection.execute(
            sa.text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'replace_exec_view' "
                "ORDER BY ordinal_position"
            )
        )
        columns = [row[0] for row in result]
        assert columns == ["id"], f"got {columns}"


# ===========================================================================
# Interface 2: DropViewOp/DropMaterializedViewOp validate definition
# ===========================================================================

class TestDropOpValidatesDefinition:
    """Drop ops must validate ``definition`` at construction time.

    ``_validate_definition`` is called by Create/Replace ops but NOT by
    Drop ops. If ``definition=""`` is passed to ``DropViewOp``, it's
    silently stored but will fail later in ``reverse()`` with a confusing
    ``TypeError`` from ``CreateViewOp``'s validation. Validate at
    construction so the error points at the actual misuse site.
    """

    @pytest.mark.parametrize(
        "op_class",
        [DropViewOp, DropMaterializedViewOp],
        ids=["drop_view", "drop_mv"],
    )
    def test_drop_op_rejects_empty_definition(self, op_class):
        with pytest.raises(TypeError, match="(?i)definition"):
            op_class("v", definition="")

    @pytest.mark.parametrize(
        "op_class",
        [DropViewOp, DropMaterializedViewOp],
        ids=["drop_view", "drop_mv"],
    )
    def test_drop_op_rejects_non_string_definition(self, op_class):
        with pytest.raises(TypeError, match="(?i)definition"):
            op_class("v", definition=123)

    @pytest.mark.parametrize(
        "op_class",
        [DropViewOp, DropMaterializedViewOp],
        ids=["drop_view", "drop_mv"],
    )
    def test_drop_op_accepts_none_definition(self, op_class):
        """``definition=None`` (the default) is valid — means no reverse."""
        op = op_class("v")
        assert op.definition is None

    def test_drop_view_op_accepts_valid_definition(self):
        op = DropViewOp("v", definition="SELECT 1")
        assert op.definition == "SELECT 1"


# ===========================================================================
# Interface 3: schema="" normalization to None in all 7 op classes
# ===========================================================================

class TestOpSchemaNormalization:
    """All 7 view op classes must normalize ``schema=""`` to ``None``.

    ``ViewRecord`` normalizes ``schema=""`` to ``None`` so the
    schema-match check in the comparator works (``"" != None`` would
    cause a false DropViewOp). The op classes must do the same so ops
    constructed with ``schema=""`` (e.g. from a renderer round-trip)
    dedup correctly against ``schema=None`` ops.
    """

    @pytest.mark.parametrize(
        "op_class,kwargs",
        [
            (CreateViewOp, {"definition": "SELECT 1"}),
            (DropViewOp, {}),
            (ReplaceViewOp, {"definition": "SELECT 1"}),
            (CreateMaterializedViewOp, {"definition": "SELECT 1"}),
            (DropMaterializedViewOp, {}),
            (ReplaceMaterializedViewOp, {"definition": "SELECT 1"}),
            (RefreshMaterializedViewOp, {}),
        ],
        ids=[
            "create_view", "drop_view", "replace_view",
            "create_materialized_view", "drop_materialized_view",
            "replace_materialized_view", "refresh_materialized_view",
        ],
    )
    def test_empty_string_schema_normalized_to_none(self, op_class, kwargs):
        op = op_class("v", schema="", **kwargs)
        assert op.schema is None, f"got {op.schema!r}"


# ===========================================================================
# Runtime DDL functions: positional params for create_view / create_materialized_view
# ===========================================================================

class TestRuntimePositionalParams:
    """The runtime ``create_view`` / ``create_materialized_view`` /
    ``refresh_materialized_view`` functions accept the params that were
    positional in upstream sqlalchemy-utils 0.42.0 positionally.

    ``schema`` remains keyword-only (it was always keyword-only in our
    changes). The keyword form also continues to work.
    """

    def test_create_view_cascade_on_drop_positional_works(self):
        """Passing ``cascade_on_drop`` positionally must succeed."""
        md = sa.MetaData()
        sel = sa.select(sa.text("1"))
        table = create_view("v", sel, md, False)
        assert isinstance(table, sa.Table)

    def test_create_view_replace_positional_works(self):
        """Passing ``replace`` positionally must succeed."""
        md = sa.MetaData()
        sel = sa.select(sa.text("1"))
        table = create_view("v", sel, md, True, True)
        assert isinstance(table, sa.Table)

    def test_create_materialized_view_cascade_on_drop_positional_works(self):
        """Passing ``cascade_on_drop`` positionally must succeed."""
        md = sa.MetaData()
        sel = sa.select(sa.text("1"))
        table = create_materialized_view("mv", sel, md, None, None, True)
        assert isinstance(table, sa.Table)

    def test_refresh_materialized_view_concurrently_positional_works(self):
        """Passing ``concurrently`` positionally to refresh_materialized_view
        must succeed (does not touch DB; just constructs the DDL element)."""
        md = sa.MetaData()
        sel = sa.select(sa.text("1"))
        create_materialized_view("mv", sel, md)
        session = MagicMock()
        refresh_materialized_view(session, "mv", True)
        session.execute.assert_called_once()
        executed = session.execute.call_args[0][0]
        assert isinstance(executed, RefreshMaterializedView)
        assert executed.concurrently is True

    def test_refresh_materialized_view_ddl_concurrently_positional_works(self):
        """Passing ``concurrently`` positionally to RefreshMaterializedView
        constructor must succeed."""
        element = RefreshMaterializedView("mv", True)
        assert element.concurrently is True

class TestStringSelectableGuard:
    """create_view / create_materialized_view reject string selectables with
    a helpful TypeError directing the caller to wrap in sa.text().
    """

    def test_create_view_rejects_string_selectable(self):
        md = sa.MetaData()
        with pytest.raises(TypeError, match="sa.text"):
            create_view("v", "SELECT 1", md)

    def test_create_materialized_view_rejects_string_selectable(self):
        md = sa.MetaData()
        with pytest.raises(TypeError, match="sa.text"):
            create_materialized_view("mv", "SELECT 1", md)


# ===========================================================================
# Cross-schema create ordering
# ===========================================================================

class TestCrossSchemaCreateOrdering:
    """When a view in schema A depends on a view in schema B, the generated
    migration ops must create the dependency (schema B's view) before the
    dependent (schema A's view), regardless of schema iteration order.
    """

    def test_cross_schema_dependency_create_order(self):
        model_views = [
            ViewRecord(
                name="dependent_view",
                schema="schema_a",
                selectable="SELECT * FROM schema_b.base_view",
            ),
            ViewRecord(
                name="base_view",
                schema="schema_b",
                selectable="SELECT 1 AS id",
            ),
        ]

        canonical_by_schema = {
            "schema_a": ({"dependent_view": "SELECT * FROM schema_b.base_view"}, {}, set()),
            "schema_b": ({"base_view": "SELECT 1 AS id"}, {}, set()),
        }

        def mock_canonicalize_all(connection, view_records, db_views):
            schema = view_records[0].schema if view_records else None
            return canonical_by_schema[schema]

        # Test both schema iteration orders — the bug only manifests when
        # the dependent's schema is processed before the dependency's.
        for schema_order in (["schema_b", "schema_a"], ["schema_a", "schema_b"]):
            autogen_context, upgrade_ops = _make_mock_autogen_context(
                model_views=model_views
            )

            with patch.object(
                comparator_module, "get_database_views", return_value={}
            ), patch.object(
                comparator_module, "get_database_materialized_views", return_value={}
            ), patch.object(
                comparator_module, "_canonicalize_all_views",
                side_effect=mock_canonicalize_all,
            ), patch.object(
                comparator_module, "get_dependent_views", return_value={}
            ), patch.object(comparator_module, "log"):
                comparator_module.compare_views(
                    autogen_context, upgrade_ops, schema_order
                )

            create_ops = [
                op for op in upgrade_ops.ops if isinstance(op, CreateViewOp)
            ]
            assert len(create_ops) == 2, f"got {len(create_ops)}: {[(op.name, op.schema) for op in create_ops]}"

            base_idx = next(
                i for i, op in enumerate(create_ops)
                if op.name == "base_view" and op.schema == "schema_b"
            )
            dep_idx = next(
                i for i, op in enumerate(create_ops)
                if op.name == "dependent_view" and op.schema == "schema_a"
            )
            assert base_idx < dep_idx, f"schemas={schema_order}; ops: {[(op.name, op.schema) for op in create_ops]}"


# ===========================================================================
# RefreshMaterializedViewOp: SQL execution + renderer
# ===========================================================================

class TestRefreshMaterializedViewOpSql:

    def test_sql_refresh(self):
        op = RefreshMaterializedViewOp("mv")
        sqls = _capture_sql(op)
        assert len(sqls) == 1
        assert "REFRESH MATERIALIZED VIEW" in sqls[0]
        assert "mv" in sqls[0]
        assert "CONCURRENTLY" not in sqls[0]

    def test_sql_refresh_concurrently(self):
        op = RefreshMaterializedViewOp("mv", concurrently=True)
        sqls = _capture_sql(op)
        assert len(sqls) == 1
        assert "REFRESH MATERIALIZED VIEW CONCURRENTLY" in sqls[0]


class TestRendererRefreshMaterializedView:

    def test_renders_basic(self):
        op = RefreshMaterializedViewOp("mv")
        result = render_refresh_materialized_view(_make_autogen_context(), op)
        assert result == "op.refresh_materialized_view('mv')"

    def test_renders_with_schema(self):
        op = RefreshMaterializedViewOp("mv", schema="public")
        result = render_refresh_materialized_view(_make_autogen_context(), op)
        assert "schema='public'" in result

    def test_renders_concurrently(self):
        op = RefreshMaterializedViewOp("mv", concurrently=True)
        result = render_refresh_materialized_view(_make_autogen_context(), op)
        assert "concurrently=True" in result


# ===========================================================================
# Comparator offline mode and dependent warning edge cases
# ===========================================================================

class TestCompareViewsOfflineMode:

    def test_compare_views_offline_mode_skips(self, caplog):
        """When ``autogen_context.connection`` is None (offline mode),
        compare_views must skip view diffing and log a warning containing
        'offline'."""
        autogen_context = MagicMock(spec=AutogenContext)
        autogen_context.connection = None
        autogen_context.metadata = sa.MetaData()

        upgrade_ops = MagicMock()
        upgrade_ops.ops = []

        with caplog.at_level(
            logging.WARNING, logger="sqlalchemy_utils.alembic.comparator"
        ):
            compare_views(autogen_context, upgrade_ops)

        assert upgrade_ops.ops == []
        warnings = [
            rec for rec in caplog.records if rec.levelno >= logging.WARNING
        ]
        assert warnings
        assert any("offline" in rec.message.lower() for rec in warnings)


class TestWarnIfDependentsHandlesFailure:

    def test_warn_if_dependents_handles_failure(self, caplog):
        """When ``get_dependent_views`` raises SQLAlchemyError, the failure
        is logged and the drop proceeds without the dependent warning."""
        connection = MagicMock()

        with patch.object(
            comparator_module, "get_dependent_views",
            side_effect=sa.exc.SQLAlchemyError("test"),
        ):
            with caplog.at_level(
                logging.WARNING, logger="sqlalchemy_utils.alembic.comparator"
            ):
                comparator_module._warn_if_dependents(
                    connection, "my_view", None, "view"
                )

        messages = [rec.message for rec in caplog.records]
        assert any("Failed to query dependent views" in m for m in messages)
        assert not any("dependent view(s)" in m for m in messages)


# ===========================================================================
# DDL validation: materialized+replace, non-list indexes
# ===========================================================================

class TestCreateViewRejectsMaterializedAndReplace:

    def test_create_view_rejects_materialized_and_replace(self):
        with pytest.raises(ValueError):
            CreateView("v", sa.select(1), materialized=True, replace=True)


class TestCreateMaterializedViewRejectsNonListIndexes:

    def test_create_materialized_view_rejects_non_list_indexes(self):
        with pytest.raises(TypeError, match="list"):
            create_materialized_view(
                "mv", sa.select(1), sa.MetaData(), indexes=("idx",)
            )


# ===========================================================================
# Materialized view op reverse() preserves with_data
# ===========================================================================

class TestReversePreservesWithData:

    def test_reverse_create_mv_preserves_with_data(self):
        op = CreateMaterializedViewOp("mv", "SELECT 1", with_data=True)
        rev = op.reverse()
        assert rev.with_data is True

    def test_reverse_drop_mv_preserves_with_data(self):
        op = DropMaterializedViewOp(
            "mv", definition="SELECT 1", with_data=True
        )
        rev = op.reverse()
        assert rev.with_data is True

    def test_reverse_replace_mv_preserves_with_data(self):
        op = ReplaceMaterializedViewOp(
            "mv", "SELECT 2", old_definition="SELECT 1", with_data=True
        )
        rev = op.reverse()
        assert rev.with_data is True


# ===========================================================================
# DropView DDL compilation
# ===========================================================================

class TestCompileDropViewMaterialized:

    def test_compile_drop_view_materialized(self):
        sql = _compile_ddl(DropView("mv", materialized=True))
        assert "DROP MATERIALIZED VIEW" in sql
        assert "CASCADE" in sql


# ===========================================================================
# Cascade propagation for materialized view create/replace paths
# ===========================================================================

class TestMaterializedViewCascadePropagation:
    """CreateMaterializedViewOp.cascade_on_drop and
    ReplaceMaterializedViewOp.cascade must honor ViewRecord.cascade_on_drop
    on the create and replace paths (mirroring the regular-view behavior
    covered by TestCascadeOnDropPropagation).
    """

    def test_create_mv_propagates_cascade_false(self):
        """CreateMaterializedViewOp.cascade_on_drop=False when
        ViewRecord.cascade_on_drop is False and the MV is absent from the
        DB (Create path, not Drop)."""
        vr = ViewRecord(
            name="mv_new_nocascade", selectable="SELECT 1 AS col",
            schema=None, materialized=True, cascade_on_drop=False,
        )
        autogen_context, upgrade_ops = _make_mock_autogen_context(
            model_views=[vr]
        )
        with _patch_comparator(
            db_views={}, db_mvs={},
            canonical_return=({}, {"mv_new_nocascade": "SELECT 1 AS col"}, set()),
        ):
            comparator_module.compare_views(
                autogen_context, upgrade_ops, [None]
            )

        create_ops = [
            op for op in upgrade_ops.ops
            if isinstance(op, CreateMaterializedViewOp)
        ]
        assert len(create_ops) == 1, f"got {create_ops}"
        assert create_ops[0].name == "mv_new_nocascade"
        assert create_ops[0].cascade_on_drop is False, (
            f"got {create_ops[0].cascade_on_drop!r}"
        )

    def test_replace_mv_propagates_cascade_false(self):
        """ReplaceMaterializedViewOp.cascade=False when
        ViewRecord.cascade_on_drop is False and the MV definition changed
        (Replace path, not Create/Drop)."""
        vr = ViewRecord(
            name="mv_replace_nocascade",
            selectable="SELECT 1 AS col",
            schema=None,
            materialized=True,
            cascade_on_drop=False,
        )
        autogen_context, upgrade_ops = _make_mock_autogen_context(
            model_views=[vr]
        )
        with _patch_comparator(
            db_views={}, db_mvs={"mv_replace_nocascade": "SELECT 2 AS col"},
            canonical_return=({}, {"mv_replace_nocascade": "SELECT 1 AS col"}, set()),
        ):
            comparator_module.compare_views(
                autogen_context, upgrade_ops, [None]
            )

        replace_ops = [
            op for op in upgrade_ops.ops
            if isinstance(op, ReplaceMaterializedViewOp)
        ]
        assert len(replace_ops) == 1, f"got {replace_ops}"
        assert replace_ops[0].name == "mv_replace_nocascade"
        assert replace_ops[0].cascade is False, (
            f"got {replace_ops[0].cascade!r}"
        )


# ===========================================================================
# pg_catalog: explicit schema filter
# ===========================================================================

class TestGetDatabaseViewsWithExplicitSchema:
    """get_database_views(connection, schema="public") must return only the
    views in the given schema, keyed by view name."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_get_database_views_with_explicit_schema(self, connection):
        _drop_views(connection, ["schema_filter_view"])
        _create_base_table(connection)
        try:
            connection.execute(
                sa.text(
                    "CREATE VIEW schema_filter_view AS "
                    "SELECT id FROM _cmp_test_base"
                )
            )
            connection.commit()

            views = get_database_views(connection, schema="public")
            assert "schema_filter_view" in views
            assert "SELECT" in views["schema_filter_view"].upper()
        finally:
            _drop_views(connection, ["schema_filter_view"])
            _drop_base_table(connection)


# ===========================================================================
# register_view_comparator idempotency
# ===========================================================================

class TestRegisterViewComparatorIdempotency:
    """register_view_comparator() must be safe to call more than once."""

    def test_register_view_comparator_is_idempotent(self):
        # The autouse ``_reset_registered`` fixture clears the flag before
        # this test runs, so the first call performs the registration.
        assert comparator_module._registered is False
        register_view_comparator()
        assert comparator_module._registered is True
        # Second call must be a no-op and must not raise.
        register_view_comparator()
        assert comparator_module._registered is True


# ===========================================================================
# create_materialized_view: index creation on SQLite
# ===========================================================================

class TestCreateMaterializedViewWithIndexes:
    """create_materialized_view must create the backing table's indexes
    when metadata.create_all() is invoked (SQLite runtime path)."""

    def test_create_materialized_view_with_indexes(self):
        metadata = sa.MetaData()
        source = sa.Table(
            "mv_src", metadata,
            sa.Column("col", sa.Integer),
        )
        selectable = sa.select(source.c.col)
        mv_table = create_materialized_view(
            "mv_name",
            selectable,
            metadata,
            indexes=[sa.Index("idx_col", sa.Column("col", sa.Integer))],
        )

        engine = sa.create_engine("sqlite:///:memory:")
        # Create the source table and the MV backing table directly. SQLite
        # does not support CREATE MATERIALIZED VIEW, so metadata.create_all
        # would abort on the MV DDL listener before the backing table exists.
        source.create(engine)
        mv_table.create(engine)
        indexes = sa.inspect(engine).get_indexes("mv_name")
        index_names = [idx["name"] for idx in indexes]
        assert "idx_col" in index_names, f"got {index_names!r}"

