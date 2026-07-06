"""Tests for the SQLAlchemy-Utils Alembic view integration.

Covers ViewRecord, the 6 view Operations, renderers, comparators,
dependency resolution, pg_catalog helpers, autogenerate integration,
public API, import safety, DDL formatting, schema resolution and the
ViewMixin integration.
"""
from __future__ import annotations

import inspect
import logging
import subprocess
import sys
import textwrap
from dataclasses import FrozenInstanceError as _FrozenInstanceError
from pathlib import Path
from typing import Optional
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
    _OUTER_SAVEPOINT,
    _safe_resolve,
    _schema_matches,
    compare_views,
    register_view_comparator,
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
)
from sqlalchemy_utils.alembic.pg_catalog import (
    get_database_materialized_views,
    get_database_views,
)
from sqlalchemy_utils.alembic.renderer import (
    render_create_materialized_view,
    render_create_view,
    render_drop_materialized_view,
    render_drop_view,
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


# ===========================================================================
# Shared helpers
# ===========================================================================

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


def _find_view_listener(metadata: sa.MetaData, materialized: Optional[bool] = None):
    """Find a CreateView DDL element registered on *metadata*.

    *materialized*=None returns any CreateView listener; True/False
    filters on the listener's ``materialized`` attribute.
    """
    listeners = list(metadata.dispatch.after_create)

    def _listener_materialized(listener) -> Optional[bool]:
        cv = getattr(listener, "__self__", None)
        if isinstance(cv, CreateView):
            return cv.materialized
        return getattr(listener, "materialized", None)

    found = [
        listener
        for listener in listeners
        if isinstance(getattr(listener, "__self__", None), CreateView)
        and (materialized is None or _listener_materialized(listener) is materialized)
    ]
    if found:
        return found[0]
    found = [
        listener
        for listener in listeners
        if hasattr(listener, "name")
        and hasattr(listener, "replace")
        and hasattr(listener, "selectable")
        and (materialized is None or _listener_materialized(listener) is materialized)
    ]
    return found[0] if found else None


def _make_autogen_context() -> AutogenContext:
    """Create a minimal mock AutogenContext for renderer tests."""
    ctx = MagicMock(spec=AutogenContext)
    ctx.imports = set()
    return ctx


def _make_real_autogen_context(connection, metadata):
    """Create a real AutogenContext backed by a PG connection."""
    migration_ctx = MigrationContext.configure(connection)
    return AutogenContext(migration_ctx, metadata=metadata)


def _run_comparator(connection, metadata, schemas=None):
    """Run compare_views and return the generated UpgradeOps."""
    autogen_context = _make_real_autogen_context(connection, metadata)
    upgrade_ops = alembic_ops.UpgradeOps([])
    if schemas is None:
        schemas = [None]
    compare_views(autogen_context, upgrade_ops, schemas)
    return upgrade_ops


# ===========================================================================
# ViewRecord
# ===========================================================================

class TestViewRecordCreation:
    """ViewRecord creation with required and optional fields."""

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

    def test_create_with_none_schema(self):
        record = ViewRecord(name="test_view", selectable="SELECT 1", schema=None)
        assert record.schema is None

    def test_create_with_empty_string_schema_normalizes_to_none(self):
        """Empty-string schema is falsy and must be normalized to None.

        Without normalization, ``""`` is treated as "no schema" by
        ``_quote_qualified_name`` (view created in ``current_schema()``),
        but ``_schema_matches("", None)`` returns ``False``, causing the
        view to be dropped as a false ``DropViewOp``.
        """
        record = ViewRecord(name="test_view", selectable="SELECT 1", schema="")
        assert record.schema is None
        assert record == ViewRecord(name="test_view", selectable="SELECT 1", schema=None)
        assert hash(record) == hash(
            ViewRecord(name="test_view", selectable="SELECT 1", schema=None)
        )

    def test_create_default_cascade_on_drop(self):
        record = ViewRecord(name="test_view", selectable="SELECT 1")
        assert record.cascade_on_drop is True

    def test_rejects_none_selectable(self):
        """ViewRecord rejects selectable=None at construction time."""
        with pytest.raises((TypeError, ValueError), match="(?i)selectable"):
            ViewRecord(name="v", selectable=None)


class TestViewRecordFreezing:
    """ViewRecord is frozen; mutation raises FrozenInstanceError."""

    def test_is_frozen(self):
        record = ViewRecord(name="test_view", selectable="SELECT 1")
        with pytest.raises(_FrozenInstanceError):
            record.name = "different_view"

    def test_raises_frozen_error_on_attribute_deletion(self):
        record = ViewRecord(name="test_view", selectable="SELECT 1")
        with pytest.raises(_FrozenInstanceError):
            del record.name


class TestViewRecordEquality:
    """ViewRecord equality semantics."""

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
    """ViewRecord hashing for set/dict use."""

    def test_hash_consistent_with_equality(self):
        record1 = ViewRecord(name="test_view", selectable="SELECT 1")
        record2 = ViewRecord(name="test_view", selectable="SELECT 1")
        assert hash(record1) == hash(record2)
        assert record1 == record2

    def test_different_records_have_different_hashes(self):
        record1 = ViewRecord(name="view1", selectable="SELECT 1")
        record2 = ViewRecord(name="view2", selectable="SELECT 1")
        assert hash(record1) != hash(record2)

    def test_storable_in_set(self):
        record1 = ViewRecord(name="test_view", selectable="SELECT 1")
        record2 = ViewRecord(name="test_view", selectable="SELECT 1")
        record3 = ViewRecord(name="other_view", selectable="SELECT 1")
        view_set = {record1, record3}
        assert len(view_set) == 2
        assert record1 in view_set
        assert record3 in view_set
        assert record2 in view_set

    def test_storable_in_dict_multiple_keys(self):
        record1 = ViewRecord(name="view1", selectable="SELECT 1")
        record2 = ViewRecord(name="view2", selectable="SELECT 2")
        record3 = ViewRecord(name="view3", selectable="SELECT 3")
        value_map = {
            record1: "data1",
            record2: "data2",
            record3: "data3",
        }
        assert len(value_map) == 3
        assert value_map[record1] == "data1"
        assert value_map[record2] == "data2"
        assert value_map[record3] == "data3"


class TestViewRecordRepr:
    """ViewRecord string representations."""

    def test_repr_with_schema(self):
        record = ViewRecord(
            name="test_view", selectable="SELECT 1", schema="public"
        )
        repr_str = repr(record)
        assert "ViewRecord" in repr_str
        assert "name='test_view'" in repr_str
        assert "schema=" in repr_str

    def test_repr_without_schema(self):
        record = ViewRecord(name="test_view", selectable="SELECT 1")
        repr_str = repr(record)
        assert "ViewRecord" in repr_str
        assert "schema=" in repr_str

    def test_str(self):
        record = ViewRecord(
            name="test_view", selectable="SELECT 1", materialized=True
        )
        str_repr = str(record)
        assert "test_view" in str_repr
        assert "materialized=True" in str_repr


# ===========================================================================
# pg_catalog
# ===========================================================================

@pytest.mark.parametrize(
    "fetch_fn",
    [get_database_views, get_database_materialized_views],
    ids=["views", "materialized_views"],
)
class TestGetDatabaseViews:
    """get_database_views / get_database_materialized_views query pg_catalog."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_query_empty_database(self, connection, fetch_fn):
        assert fetch_fn(connection) == {}

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_query_with_schema_filter(self, connection, fetch_fn):
        result = fetch_fn(connection, schema="public")
        assert isinstance(result, dict)
        for name, definition in result.items():
            assert isinstance(name, str)
            assert isinstance(definition, str)

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_query_all_schemas_when_schema_none(self, connection, fetch_fn):
        result = fetch_fn(connection, schema=None)
        assert isinstance(result, dict)
        for name, definition in result.items():
            assert isinstance(name, str)
            assert isinstance(definition, str)


# ===========================================================================
# Operations
# ===========================================================================

class TestCreateViewOp:
    """CreateViewOp instantiation, reverse, SQL, and classmethod."""

    def test_instantiation(self):
        op = CreateViewOp("v1", "SELECT 1")
        assert op.name == "v1"
        assert op.definition == "SELECT 1"
        assert op.schema is None
        assert op.replace is False

    def test_reverse_returns_drop_view(self):
        op = CreateViewOp("v1", "SELECT 1")
        rev = op.reverse()
        assert isinstance(rev, DropViewOp)
        assert rev.name == "v1"
        assert rev.definition == "SELECT 1"

    def test_sql_without_replace(self):
        op = CreateViewOp("v1", "SELECT 1")
        sqls = _capture_sql(op)
        assert sqls == ["CREATE VIEW v1 AS SELECT 1"]

    def test_sql_with_schema(self):
        op = CreateViewOp("v1", "SELECT 1", schema="public")
        sqls = _capture_sql(op)
        assert sqls == ["CREATE VIEW public.v1 AS SELECT 1"]

    def test_create_view_accepts_replace_kwarg(self):
        operations = MagicMock()
        operations.invoke.return_value = None
        with pytest.warns(DeprecationWarning):
            CreateViewOp.create_view(operations, "test_view", "SELECT 1", replace=True)
        operations.invoke.assert_called_once()
        invoked_op = operations.invoke.call_args[0][0]
        assert isinstance(invoked_op, CreateViewOp)
        assert invoked_op.replace is True

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

    def test_deprecate_replace_on_create(self):
        """CreateViewOp(replace=True) emits a DeprecationWarning."""
        with pytest.warns(DeprecationWarning) as record:
            CreateViewOp("v", "SELECT 1", replace=True)
        assert len(record) == 1
        message = str(record[0].message)
        assert "op.replace_view" in message


class TestDropViewOp:
    """DropViewOp instantiation, reverse, SQL, and classmethod."""

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

    def test_sql_materialized(self):
        op = DropMaterializedViewOp("v1")
        sqls = _capture_sql(op)
        assert sqls == ["DROP MATERIALIZED VIEW IF EXISTS v1 CASCADE"]

    def test_sql_with_schema(self):
        op = DropViewOp("v1", schema="myschema")
        sqls = _capture_sql(op)
        assert sqls == ["DROP VIEW IF EXISTS myschema.v1 CASCADE"]


class TestReplaceViewOp:
    """ReplaceViewOp instantiation, reverse, and SQL."""

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
        assert sqls == ["CREATE OR REPLACE VIEW v1 AS SELECT 2"]


class TestCreateMaterializedViewOp:
    """CreateMaterializedViewOp instantiation, reverse, and SQL."""

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
    """DropMaterializedViewOp instantiation, reverse, and SQL."""

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
    """ReplaceMaterializedViewOp instantiation, reverse, and SQL."""

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

    def test_replace_mv_cascade_field(self):
        """ReplaceMaterializedViewOp stores cascade kwarg (default True)."""
        op_default = ReplaceMaterializedViewOp("mv", "SELECT 1")
        assert op_default.cascade is True

        op_false = ReplaceMaterializedViewOp("mv", "SELECT 1", cascade=False)
        assert op_false.cascade is False

    def test_replace_mv_impl_respects_cascade_false(self):
        """``_replace_materialized_view_impl`` omits CASCADE when op.cascade=False."""
        from sqlalchemy_utils.alembic.operations import (
            _replace_materialized_view_impl,
        )

        op = ReplaceMaterializedViewOp("mv", "SELECT 1", cascade=False)
        sqls = _capture_sql(op)
        assert len(sqls) == 2
        assert "CASCADE" not in sqls[0].upper(), (
            f"DROP must not contain CASCADE when op.cascade=False. "
            f"DROP SQL: {sqls[0]!r}"
        )
        assert sqls[0] == "DROP MATERIALIZED VIEW IF EXISTS mv"
        assert sqls[1] == "CREATE MATERIALIZED VIEW mv AS SELECT 1 WITH NO DATA"

    def test_replace_mv_impl_cascade_true_default(self):
        """``_replace_materialized_view_impl`` emits CASCADE when op.cascade=True."""
        op = ReplaceMaterializedViewOp("mv", "SELECT 1")
        sqls = _capture_sql(op)
        assert sqls[0] == "DROP MATERIALIZED VIEW IF EXISTS mv CASCADE"


# ---------------------------------------------------------------------------
# Operations: keyword-only params (parametrized)
# ---------------------------------------------------------------------------

class TestOpKeywordOnlyParams:
    """Op classmethods enforce schema= as keyword-only."""

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

    def test_init_keyword_only_params_dropview(self):
        """DropViewOp.__init__ params after name are keyword-only."""
        sig = inspect.signature(DropViewOp.__init__)
        params = list(sig.parameters.values())
        for p in params[2:]:
            assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"DropViewOp.__init__ param '{p.name}' should be keyword-only"
            )

    def test_init_keyword_only_params_create_mv(self):
        """CreateMaterializedViewOp.__init__ params after name/definition are keyword-only."""
        sig = inspect.signature(CreateMaterializedViewOp.__init__)
        params = list(sig.parameters.values())
        for p in params[3:]:
            assert p.kind == inspect.Parameter.KEYWORD_ONLY, (
                f"CreateMaterializedViewOp.__init__ param '{p.name}' should be keyword-only"
            )


# ---------------------------------------------------------------------------
# Operations: reverse() round-trip fidelity
# ---------------------------------------------------------------------------

class TestReverseRoundTrip:
    """reverse() round-trips preserve op attributes."""

    def test_create_view_reverse_round_trip_drops_replace(self):
        with pytest.warns(DeprecationWarning):
            op = CreateViewOp("v", "SELECT 1", replace=True)
        double_reversed = op.reverse().reverse()
        assert isinstance(double_reversed, CreateViewOp)
        assert double_reversed.replace is False

    def test_replace_mv_reverse_preserves_old_definition(self):
        op = ReplaceMaterializedViewOp("mv", "SELECT 2", old_definition="SELECT 1")
        rev = op.reverse()
        assert rev.old_definition == "SELECT 2"


# ---------------------------------------------------------------------------
# Operations: validation
# ---------------------------------------------------------------------------

class TestOpValidation:
    """Op __init__ validation of definition argument."""

    def test_create_view_rejects_none_definition(self):
        with pytest.raises((TypeError, ValueError), match="(?i)definition"):
            CreateViewOp("v", None)

    def test_listener_accumulation(self):
        """Calling create_view twice with same name accumulates listeners."""
        metadata = sa.MetaData()
        selectable = sa.select(sa.column("id", sa.Integer))

        create_view("my_view", selectable, metadata)
        after_first = len(metadata.dispatch.after_create)

        create_view("my_view", selectable, metadata)
        after_second = len(metadata.dispatch.after_create)

        assert after_second == after_first + 1

    def test_create_mv_runtime_vs_op_consistency(self):
        """Runtime and op paths agree on WITH [NO] DATA when with_data=True."""
        metadata = sa.MetaData()
        create_materialized_view(
            "runtime_mv",
            sa.select(sa.table("src", sa.column("id", sa.Integer))),
            metadata,
        )
        runtime_ddl = _find_view_listener(metadata, materialized=True)
        assert runtime_ddl is not None

        engine = sa.create_engine("sqlite:///:memory:")
        compiled_runtime = str(runtime_ddl.compile(dialect=engine.dialect)).upper()

        op_sqls = _capture_sql(
            CreateMaterializedViewOp("op_mv", "SELECT 1", with_data=True)
        )
        op_sql = op_sqls[0].upper() if op_sqls else ""

        runtime_emits_with_no_data = "WITH NO DATA" in compiled_runtime
        op_emits_with_no_data = "WITH NO DATA" in op_sql
        assert runtime_emits_with_no_data == op_emits_with_no_data


# ===========================================================================
# Renderer
# ===========================================================================

class TestRendererCreateView:
    """render_create_view behavior."""

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

    def test_preserves_empty_string_schema(self):
        op = CreateViewOp("v", "SELECT 1", schema="")
        rendered = render_create_view(_make_autogen_context(), op)
        assert "schema=" in rendered

    @pytest.mark.parametrize(
        "replace,expected_in_output",
        [(False, ""), (True, "replace=True")],
    )
    def test_renders_replace(self, replace, expected_in_output):
        if replace:
            with pytest.warns(DeprecationWarning):
                op = CreateViewOp("v", "SELECT 1", replace=replace)
        else:
            op = CreateViewOp("v", "SELECT 1", replace=replace)
        rendered = render_create_view(_make_autogen_context(), op)
        if expected_in_output:
            assert expected_in_output in rendered
        else:
            assert "replace=True" not in rendered


class TestRendererDropView:
    """render_drop_view behavior."""

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
    """render_replace_view behavior."""

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

    def test_preserves_empty_string_old_definition(self):
        op = ReplaceViewOp("v", "SELECT 2", old_definition="")
        rendered = render_replace_view(_make_autogen_context(), op)
        assert "old_definition=" in rendered


class TestRendererCreateMaterializedView:
    """render_create_materialized_view behavior."""

    def test_produces_valid_python(self):
        op = CreateMaterializedViewOp("mv_stats", "SELECT count(*) FROM events")
        result = render_create_materialized_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_omits_with_data_when_default(self):
        """Renderer omits with_data by default and when False (the default)."""
        op = CreateMaterializedViewOp("mv_stats", "SELECT 1")
        result = render_create_materialized_view(_make_autogen_context(), op)
        assert "with_data=" not in result

        op_false = CreateMaterializedViewOp("mv_stats", "SELECT 1", with_data=False)
        result_false = render_create_materialized_view(_make_autogen_context(), op_false)
        assert "with_data=" not in result_false

        op_true = CreateMaterializedViewOp("mv_stats", "SELECT 1", with_data=True)
        result_true = render_create_materialized_view(_make_autogen_context(), op_true)
        assert "with_data=True" in result_true

    def test_schema_included_when_provided(self):
        op = CreateMaterializedViewOp("mv_stats", "SELECT 1", schema="analytics")
        result = render_create_materialized_view(_make_autogen_context(), op)
        assert "schema='analytics'" in result


class TestRendererDropMaterializedView:
    """render_drop_materialized_view behavior."""

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
        assert "with_data=True" in rendered, (
            f"with_data=True must be rendered for drop_materialized_view so "
            f"the downgrade re-creates the MV WITH DATA; got: {rendered!r}"
        )

    def test_omits_with_data_when_default(self):
        """``with_data`` is omitted when False (the default)."""
        op = DropMaterializedViewOp("mv", definition="SELECT 1", with_data=False)
        rendered = render_drop_materialized_view(_make_autogen_context(), op)
        assert "with_data=" not in rendered


class TestRendererReplaceMaterializedView:
    """render_replace_materialized_view behavior."""

    def test_produces_valid_python(self):
        op = ReplaceMaterializedViewOp("mv_stats", "SELECT count(*) FROM events")
        result = render_replace_materialized_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_omits_with_data_when_default(self):
        """Renderer omits with_data by default and when False (the default)."""
        op = ReplaceMaterializedViewOp("mv_stats", "SELECT 2")
        result = render_replace_materialized_view(_make_autogen_context(), op)
        assert "with_data=" not in result

        op_false = ReplaceMaterializedViewOp("mv_stats", "SELECT 2", with_data=False)
        result_false = render_replace_materialized_view(_make_autogen_context(), op_false)
        assert "with_data=" not in result_false

        op_true = ReplaceMaterializedViewOp("mv_stats", "SELECT 2", with_data=True)
        result_true = render_replace_materialized_view(_make_autogen_context(), op_true)
        assert "with_data=True" in result_true

    def test_schema_included_when_provided(self):
        op = ReplaceMaterializedViewOp("mv_stats", "SELECT 2", schema="analytics")
        result = render_replace_materialized_view(_make_autogen_context(), op)
        assert "schema='analytics'" in result

    def test_renders_old_definition(self):
        op = ReplaceMaterializedViewOp("mv_repl", "SELECT 2", old_definition="SELECT 1")
        rendered = render_replace_materialized_view(_make_autogen_context(), op)
        assert "old_definition=" in rendered

    def test_preserves_empty_string_old_definition(self):
        op = ReplaceMaterializedViewOp("mv", "SELECT 2", old_definition="")
        rendered = render_replace_materialized_view(_make_autogen_context(), op)
        assert "old_definition=" in rendered


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


class TestComparatorCreateView:
    """New view detected → CreateViewOp generated."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_view_generates_create_view_op(self, connection):
        _create_base_table(connection)
        _drop_views(connection, _CMP_TEST_VIEW_NAMES)

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

        _drop_views(connection, _CMP_TEST_VIEW_NAMES)
        _drop_base_table(connection)


class TestComparatorDropView:
    """Removed view detected → DropViewOp generated."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_view_generates_drop_view_op(self, connection):
        _create_base_table(connection)
        _drop_views(connection, _CMP_TEST_VIEW_NAMES)

        connection.execute(
            sa.text(
                "CREATE VIEW cmp_test_view AS SELECT id, name FROM _cmp_test_base"
            )
        )
        connection.commit()

        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = []

        upgrade_ops = _run_comparator(connection, metadata)

        drop_ops = [
            op
            for op in upgrade_ops.ops
            if isinstance(op, DropViewOp)
            and not getattr(op, "materialized", False)
        ]
        assert len(drop_ops) == 1
        assert drop_ops[0].name == "cmp_test_view"

        _drop_views(connection, _CMP_TEST_VIEW_NAMES)
        _drop_base_table(connection)


class TestComparatorReplaceView:
    """Changed view definition → ReplaceViewOp generated."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_view_generates_replace_view_op(self, connection):
        _create_base_table(connection)
        _drop_views(connection, _CMP_TEST_VIEW_NAMES)

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

        _drop_views(connection, _CMP_TEST_VIEW_NAMES)
        _drop_base_table(connection)


class TestComparatorCreateMV:
    """New materialized view detected → CreateMaterializedViewOp."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_mv_generates_create_mv_op(self, connection):
        _create_base_table(connection)
        _drop_views(connection, _CMP_TEST_VIEW_NAMES)

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

        _drop_views(connection, _CMP_TEST_VIEW_NAMES)
        _drop_base_table(connection)


class TestComparatorDropMV:
    """Removed materialized view → DropMaterializedViewOp."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_mv_generates_drop_mv_op(self, connection):
        _create_base_table(connection)
        _drop_views(connection, _CMP_TEST_VIEW_NAMES)

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

        _drop_views(connection, _CMP_TEST_VIEW_NAMES)
        _drop_base_table(connection)


class TestComparatorReplaceMV:
    """Changed materialized view definition → ReplaceMaterializedViewOp."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_mv_generates_replace_mv_op(self, connection):
        _create_base_table(connection)
        _drop_views(connection, _CMP_TEST_VIEW_NAMES)

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

        _drop_views(connection, _CMP_TEST_VIEW_NAMES)
        _drop_base_table(connection)


class TestComparatorNoChanges:
    """No changes → no view ops generated."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_no_changes_no_ops(self, connection):
        _create_base_table(connection)
        _drop_views(connection, _CMP_TEST_VIEW_NAMES)

        connection.execute(
            sa.text(
                "CREATE VIEW cmp_test_view2 AS SELECT id, name FROM _cmp_test_base"
            )
        )
        connection.commit()

        db_views = get_database_views(connection)
        _ = db_views.get("cmp_test_view2")

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
        assert len(matching_view_ops) == 0, (
            f"Expected no ops for matching view, got: {matching_view_ops}"
        )

        _drop_views(connection, _CMP_TEST_VIEW_NAMES)
        _drop_base_table(connection)

    def test_empty_metadata(self):
        """compare_views with no model views produces no create ops."""
        metadata = sa.MetaData()
        autogen_context = mock.MagicMock()
        autogen_context.connection = mock.MagicMock()
        autogen_context.metadata = metadata

        metadata.info.pop("sqlalchemy_utils_views", None)
        autogen_context.connection.execute.return_value.fetchall.return_value = []

        upgrade_ops = mock.MagicMock()
        upgrade_ops.ops = []

        compare_views(autogen_context, upgrade_ops, [None])

        create_op_count = sum(
            1
            for op in upgrade_ops.ops
            if type(op).__name__ in ("CreateViewOp", "CreateMaterializedViewOp")
        )
        assert create_op_count == 0


class TestComparatorSavepointRollback:
    """Savepoint rollback works (view doesn't persist after compare)."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_canonicalized_view_does_not_persist(self, connection):
        _create_base_table(connection)
        _drop_views(connection, _CMP_TEST_VIEW_NAMES)

        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="cmp_test_view",
                selectable="SELECT id, name FROM _cmp_test_base",
            )
        ]

        _run_comparator(connection, metadata)

        db_views = get_database_views(connection)
        assert "cmp_test_view" not in db_views, (
            "View should not persist after canonicalization savepoint rollback"
        )

        _drop_base_table(connection)


class TestComparatorDDLError:
    """DDL error in savepoint doesn't crash comparator."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_invalid_view_skipped_with_warning(self, connection):
        _drop_views(connection, _CMP_TEST_VIEW_NAMES)
        _drop_base_table(connection)

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
        assert len(bad_view_ops) == 0, (
            f"Invalid view should be skipped, got ops: {bad_view_ops}"
        )


# ===========================================================================
# Regression: programming errors must propagate, not be swallowed
# ===========================================================================

class _SelectableBreakingOnDialectCompile:
    """Compiles for dependency resolution but raises TypeError when
    ``compile(dialect=...)`` is called inside ``_canonicalize_all_views``."""

    def compile(self, **kw):
        if "dialect" in kw:
            raise TypeError(
                "programming error: selectable cannot be compiled against a dialect"
            )
        return "SELECT 1 AS id"


class TestProgrammingErrorPropagates:
    """Programming errors during canonicalization must propagate, not be
    swallowed by the broad except in ``_canonicalize_all_views``."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_programming_error_propagates(self, connection):
        _drop_views(connection, _CMP_TEST_VIEW_NAMES)
        _drop_base_table(connection)

        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="broken_view_for_dialect_test",
                selectable=_SelectableBreakingOnDialectCompile(),
                schema=None,
            ),
        ]

        with pytest.raises(TypeError):
            _run_comparator(connection, metadata, schemas=[None])


# ===========================================================================
# Regression: canonicalization savepoint refactor
# ===========================================================================

# Distinct names so tests don't collide with other view fixtures.
_FAILED_CANON_VIEW_NAMES = ["failed_canon_view"]
_VIEW_ON_VIEW_VIEW_NAMES = ["dep_chain_a", "dep_chain_b"]


class TestCanonicalizeViewOnViewDeps:
    """Regression: view-on-view dependencies must survive the savepoint.

    Two new model views that reference each other must both produce
    CreateViewOp (single outer savepoint keeps both alive during canonicalization).
    """

    @pytest.mark.infrastructure
    @pytest.mark.usefixtures("postgresql_dsn")
    def test_dependent_view_chain_both_created(self, connection):
        _drop_views(connection, _FAILED_CANON_VIEW_NAMES + _VIEW_ON_VIEW_VIEW_NAMES)
        _drop_base_table(connection)
        try:
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
            assert "dep_chain_a" in created_names, (
                f"Regression: dep_chain_a missing from create ops; "
                f"got {sorted(created_names)}"
            )
            assert "dep_chain_b" in created_names, (
                f"Regression: dep_chain_b missing from create ops "
                f"(likely rolled back A before canonicalizing B); "
                f"got {sorted(created_names)}"
            )
        finally:
            _drop_views(connection, _FAILED_CANON_VIEW_NAMES + _VIEW_ON_VIEW_VIEW_NAMES)
            _drop_base_table(connection)


class TestCanonicalizeSkipDoesNotDrop:
    """Regression: a view whose canonicalization fails must be SKIPPED, not dropped."""

    @pytest.mark.infrastructure
    @pytest.mark.usefixtures("postgresql_dsn")
    def test_failing_canonicalization_does_not_emit_drop(self, connection):
        _drop_views(connection, _FAILED_CANON_VIEW_NAMES + _VIEW_ON_VIEW_VIEW_NAMES)
        _drop_base_table(connection)
        try:
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
            assert failed_canon_drops == [], (
                f"Regression: false DropViewOp emitted for "
                f"failed_canon_view (canonicalization failed → should be SKIPPED, "
                f"not dropped). Got drop ops: "
                f"{[(op.name, op.schema) for op in drop_ops]}"
            )
        finally:
            _drop_views(connection, _FAILED_CANON_VIEW_NAMES + _VIEW_ON_VIEW_VIEW_NAMES)
            _drop_base_table(connection)


# ===========================================================================
# Regression: savepoint name reuse after ROLLBACK TO skips later views
# ===========================================================================

# Distinct names so this regression test does not collide with other view fixtures.
_SAVEPOINT_TEST_VIEW_NAMES = ["savepoint_a", "savepoint_b", "savepoint_c"]


def _drop_savepoint_test_views(connection):
    """Drop any leftover views created by this regression test."""
    for view_name in _SAVEPOINT_TEST_VIEW_NAMES:
        try:
            connection.execute(
                sa.text(f"DROP VIEW IF EXISTS {view_name} CASCADE")
            )
        except sa.exc.SQLAlchemyError:
            connection.rollback()
    connection.commit()


class TestCanonicalizeFailureDoesNotSkipSubsequentViews:
    """Regression: a failed view must not cascade-skip later views.

    After a view CREATE fails, ROLLBACK TO + RELEASE of the inner savepoint
    lets subsequent views (B, C) still produce CreateViewOp.
    """

    @pytest.mark.infrastructure
    @pytest.mark.usefixtures("postgresql_dsn")
    def test_failed_canonicalization_does_not_skip_subsequent_views(
        self, connection
    ):
        _drop_savepoint_test_views(connection)
        try:

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
            assert "savepoint_a" not in created_names, (
                f"Regression: savepoint_a should have been skipped (its CREATE "
                f"fails), but got {sorted(created_names)}"
            )
            # savepoint_b and savepoint_c come after the failure; they MUST still be
            # canonicalized. Before the fix the reused savepoint name caused
            # both to be silently dropped.
            assert "savepoint_b" in created_names, (
                f"Regression: savepoint_b missing from create ops "
                f"(savepoint name reuse after ROLLBACK TO likely skipped "
                f"it); got {sorted(created_names)}"
            )
            assert "savepoint_c" in created_names, (
                f"Regression: savepoint_c missing from create ops "
                f"(savepoint name reuse after ROLLBACK TO likely skipped "
                f"it); got {sorted(created_names)}"
            )
        finally:
            _drop_savepoint_test_views(connection)


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
        #   3. CREATE OR REPLACE VIEW abort_a AS ...      (FAILS — view_a)
        #   4. ROLLBACK TO SAVEPOINT su_view_cmp_v        (view_a cleanup)
        #   5. RELEASE SAVEPOINT su_view_cmp_v            (view_a cleanup)
        #   6. SELECT 1                                   (PROBE — FAILS)
        #   --- loop breaks; view_b is never touched ---
        #   7. ROLLBACK TO SAVEPOINT su_view_cmp          (finally)
        # After the loop, get_database_views / get_database_materialized_views
        # are called — stubbed via patch to return empty dicts.
        call_log: list[str] = []

        class _PoisonedTransaction(sa.exc.SQLAlchemyError):
            """Raised by the probe (SELECT 1) to simulate a poisoned tx."""

            def __init__(self):
                super().__init__(
                    "current transaction is aborted, "
                    "commands ignored until end of transaction block"
                )

        def _execute(stmt):
            text = getattr(stmt, "text", str(stmt))
            call_log.append(text)
            stripped = text.strip()

            # view_a CREATE fails — this is the trigger for the inner except.
            if stripped.startswith("CREATE OR REPLACE VIEW abort_a"):
                raise sa.exc.ProgrammingError(
                    statement=text,
                    params=None,
                    orig=Exception("relation does not exist"),
                )

            # The probe: SELECT 1 after the inner except. This simulates a
            # poisoned outer savepoint.
            if stripped == "SELECT 1":
                raise _PoisonedTransaction()

            # All other statements (SAVEPOINT, ROLLBACK TO, RELEASE, the
            # outer ROLLBACK TO) succeed.
            return MagicMock()

        connection = MagicMock()
        connection.dialect = sa.dialects.postgresql.dialect()
        connection.execute.side_effect = _execute

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
            and ("aborted state" in rec.message
                 or "aborting canonicalization" in rec.message)
        ]
        assert abort_warnings, (
            f"Expected a warning about aborted state / aborting "
            f"canonicalization after the probe failed. Got log records: "
            f"{[(rec.levelname, rec.message) for rec in caplog.records]}"
        )

        # 2. The loop broke early — view_b (abort_b) was never canonicalized.
        #    Because the loop broke before reaching it, abort_b is added to
        #    `skipped` so drop detection does not emit a false DropViewOp for
        #    a view that is still modeled but merely un-processed.
        assert "abort_b" in skipped, (
            f"abort_b should be in skipped (loop broke before reaching it; "
            f"unreached views are added to skipped to avoid false drops); "
            f"skipped={skipped}"
        )
        assert "abort_b" not in view_defs, (
            f"abort_b should not be in view_defs (loop broke before "
            f"canonicalizing it); view_defs={view_defs}"
        )
        # abort_a WAS attempted and failed — it should be in skipped.
        assert "abort_a" in skipped, (
            f"abort_a failed to canonicalize and must be in skipped; "
            f"skipped={skipped}"
        )

        # 3. No CREATE statement for abort_b was ever issued (loop broke).
        abort_b_creates = [
            sql for sql in call_log if "abort_b" in sql and "CREATE" in sql
        ]
        assert not abort_b_creates, (
            f"Loop should have broken before attempting abort_b, but "
            f"executed: {abort_b_creates}"
        )

    def test_aborted_transaction_no_false_drop_in_compare_views(self, caplog):
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

        class _PoisonedTransaction(sa.exc.SQLAlchemyError):
            """Raised by the probe (SELECT 1) to simulate a poisoned tx."""

            def __init__(self):
                super().__init__(
                    "current transaction is aborted, "
                    "commands ignored until end of transaction block"
                )

        call_log: list[str] = []

        def _execute(stmt):
            text = getattr(stmt, "text", str(stmt))
            call_log.append(text)
            stripped = text.strip()

            # view_a CREATE fails — triggers the inner except block.
            if stripped.startswith("CREATE OR REPLACE VIEW view_a"):
                raise sa.exc.ProgrammingError(
                    statement=text,
                    params=None,
                    orig=Exception("relation does not exist"),
                )

            # The probe after view_a's failure: poisoned transaction.
            if stripped == "SELECT 1":
                raise _PoisonedTransaction()

            # All other statements (SAVEPOINT, ROLLBACK TO, RELEASE) succeed.
            return MagicMock()

        connection = MagicMock()
        connection.dialect = sa.dialects.postgresql.dialect()
        connection.execute.side_effect = _execute

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

        with caplog.at_level(
            logging.WARNING, logger="sqlalchemy_utils.alembic.comparator"
        ), patch(
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
        assert "view_b" not in drop_names, (
            f"view_b must NOT be dropped — the loop broke before processing "
            f"it, so it must be in `skipped` (not a false drop). "
            f"Drop ops emitted: {drop_names}"
        )


class TestPoisonedOuterSavepointRollback:
    """A poisoned outer savepoint makes the ``finally`` ROLLBACK TO fail.

    If the outer savepoint becomes poisoned (a DB-level abort that is
    not contained by the inner savepoint), the ``ROLLBACK TO SAVEPOINT``
    in the ``finally`` block fails; the transaction stays aborted and
    crashes subsequent schema processing. After the ``finally`` block
    the connection must be probed with ``SELECT 1``; if the probe fails
    the function must return early (logging a warning) so the caller
    skips subsequent schema processing instead of crashing.
    """

    def test_poisoned_outer_savepoint_returns_empty_and_warns(self, caplog):
        """When the outer savepoint is poisoned, ``_canonicalize_all_views``
        must NOT crash. The catalog readback (``get_database_views``) fails
        because the transaction is aborted; the function catches it, marks
        all views as ``skipped`` (so drop detection does not emit false
        drops), and returns empty results. The ``finally`` ROLLBACK TO also
        fails (caught), and a post-finally ``SELECT 1`` probe logs a warning
        so the caller knows subsequent schema processing may be affected."""
        view_records = [
            ViewRecord(
                name="poison_a",
                selectable="SELECT 1 AS col",
                schema=None,
                materialized=False,
            ),
        ]

        call_log: list[str] = []

        class _PoisonedTransaction(sa.exc.SQLAlchemyError):
            def __init__(self):
                super().__init__(
                    "current transaction is aborted, "
                    "commands ignored until end of transaction block"
                )

        def _execute(stmt, *args, **kwargs):
            text = getattr(stmt, "text", str(stmt))
            call_log.append(text)
            stripped = text.strip()

            if stripped.startswith("CREATE OR REPLACE VIEW poison_a"):
                return MagicMock()

            if stripped == f"ROLLBACK TO SAVEPOINT {_OUTER_SAVEPOINT}":
                raise _PoisonedTransaction()

            if stripped == "SELECT 1":
                raise _PoisonedTransaction()

            return MagicMock()

        connection = MagicMock()
        connection.dialect = sa.dialects.postgresql.dialect()
        connection.execute.side_effect = _execute

        with caplog.at_level(
            logging.WARNING, logger="sqlalchemy_utils.alembic.comparator"
        ), patch(
            "sqlalchemy_utils.alembic.comparator.get_database_views",
            side_effect=_PoisonedTransaction(),
        ), patch(
            "sqlalchemy_utils.alembic.comparator.get_database_materialized_views",
            side_effect=_PoisonedTransaction(),
        ):
            view_defs, mv_defs, skipped = _canonicalize_all_views(
                connection, view_records, db_views_for_deps=None
            )

        assert view_defs == {}, (
            f"view_defs must be empty when the outer savepoint is poisoned; "
            f"got {view_defs}"
        )
        assert mv_defs == {}, (
            f"mv_defs must be empty when the outer savepoint is poisoned; "
            f"got {mv_defs}"
        )
        assert "poison_a" in skipped, (
            f"poison_a must be in skipped when the outer savepoint is "
            f"poisoned; got {skipped}"
        )

        poison_warnings = [
            rec
            for rec in caplog.records
            if rec.levelno >= logging.WARNING
            and ("poison" in rec.message.lower()
                 or "aborted" in rec.message.lower()
                 or "skip" in rec.message.lower()
                 or "failed" in rec.message.lower())
        ]
        assert poison_warnings, (
            f"Expected a warning about the poisoned savepoint; got "
            f"{[(rec.levelname, rec.message) for rec in caplog.records]}"
        )

        probe_calls = [s for s in call_log if s.strip() == "SELECT 1"]
        assert probe_calls, (
            "Expected a SELECT 1 probe after the finally ROLLBACK TO to "
            "detect the poisoned transaction; no probe was issued. "
            f"Call log: {call_log}"
        )

    def test_poisoned_outer_savepoint_does_not_crash_compare_views(self, caplog):
        """A poisoned outer savepoint must not crash compare_views.

        ``compare_views`` calls ``_canonicalize_all_views`` once per
        schema. If the savepoint is poisoned for one schema, the
        comparator must not crash with an aborted transaction —
        ``_canonicalize_all_views`` catches the catalog-readback failure,
        marks all views as skipped, and returns empty results. The outer
        DB-state collection (``get_database_views``) runs BEFORE
        canonicalization, so it is allowed to be called.
        """
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="poison_v",
                selectable="SELECT 1 AS col",
                schema=None,
                materialized=False,
            ),
        ]

        call_log: list[str] = []

        class _PoisonedTransaction(sa.exc.SQLAlchemyError):
            def __init__(self):
                super().__init__(
                    "current transaction is aborted, "
                    "commands ignored until end of transaction block"
                )

        def _execute(stmt, *args, **kwargs):
            text = getattr(stmt, "text", str(stmt))
            call_log.append(text)
            stripped = text.strip()

            if stripped.startswith("CREATE OR REPLACE VIEW poison_v"):
                return MagicMock()

            if stripped == f"ROLLBACK TO SAVEPOINT {_OUTER_SAVEPOINT}":
                raise _PoisonedTransaction()

            if stripped == "SELECT 1":
                raise _PoisonedTransaction()

            return MagicMock()

        connection = MagicMock()
        connection.dialect = sa.dialects.postgresql.dialect()
        connection.execute.side_effect = _execute

        autogen_context = MagicMock(spec=AutogenContext)
        autogen_context.connection = connection
        autogen_context.metadata = metadata

        upgrade_ops = alembic_ops.UpgradeOps([])

        # The outer DB-state collection (before canonicalization) returns
        # empty dicts. The critical assertion is that compare_views does
        # NOT raise and does NOT emit a false DropViewOp for poison_v.
        with caplog.at_level(
            logging.WARNING, logger="sqlalchemy_utils.alembic.comparator"
        ), patch(
            "sqlalchemy_utils.alembic.comparator.get_database_views",
            return_value={},
        ), patch(
            "sqlalchemy_utils.alembic.comparator.get_database_materialized_views",
            return_value={},
        ), patch(
            "sqlalchemy_utils.alembic.comparator.get_dependent_views",
            return_value={},
        ):
            # Must NOT raise.
            compare_views(autogen_context, upgrade_ops, schemas=[None])

        # No false DropViewOp for poison_v (it is in skipped, not dropped).
        drop_ops = [
            op for op in upgrade_ops.ops
            if isinstance(op, (DropViewOp, DropMaterializedViewOp))
        ]
        drop_names = {op.name for op in drop_ops}
        assert "poison_v" not in drop_names, (
            f"poison_v must not be dropped (poisoned savepoint → skipped); "
            f"got drops: {drop_names}"
        )


class TestComparatorNonPGDialect:
    """compare_views warns on non-PostgreSQL dialects."""

    def test_warns_on_non_pg_dialect(self, caplog):
        engine = sa.create_engine("sqlite:///:memory:")
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = []

        autogen_context = MagicMock()
        autogen_context.connection = engine.connect()
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

    def test_handles_none_schemas(self):
        """compare_views handles schemas=None without raising TypeError."""
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = []

        autogen_context = MagicMock()
        autogen_context.connection = MagicMock()
        autogen_context.connection.dialect.name = "postgresql"
        autogen_context.metadata = metadata

        upgrade_ops = MagicMock()
        upgrade_ops.ops = []

        raised = None
        try:
            compare_views(autogen_context, upgrade_ops, None)
        except Exception as exc:
            raised = exc

        assert raised is None or not isinstance(raised, TypeError)


class TestComparatorNoDoubleFetch:
    """compare_views fetches each schema's DB views only once."""

    def test_does_not_double_fetch(self, monkeypatch):
        import sqlalchemy_utils.alembic.comparator as comparator_module

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
        autogen_context.connection.dialect.name = "postgresql"
        autogen_context.metadata = metadata

        upgrade_ops = MagicMock()
        upgrade_ops.ops = []

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
        import sqlalchemy_utils.alembic.comparator as comparator_module
        from sqlalchemy_utils.alembic.operations import (
            RefreshMaterializedViewOp,
        )

        # Mock connection + autogen_context for a postgres dialect, with one
        # materialized view record in the model and a matching definition in
        # the DB so _diff_views produces no ops at all — but even if diffing
        # produced ops, none may be RefreshMaterializedViewOp.
        metadata = MagicMock()
        metadata.info = {"sqlalchemy_utils_views": []}

        autogen_context = MagicMock()
        autogen_context.connection = MagicMock()
        autogen_context.connection.dialect.name = "postgresql"
        autogen_context.metadata = metadata

        upgrade_ops = alembic_ops.UpgradeOps([])

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
        assert refresh_ops == [], (
            f"compare_views must NEVER emit RefreshMaterializedViewOp "
            f"(refresh is a runtime op, not a migration step). Found: "
            f"{refresh_ops}"
        )


# ===========================================================================
# Schema matching
# ===========================================================================

class TestSchemaMatching:
    """_schema_matches behavior."""

    @pytest.mark.parametrize(
        "view_schema,loop_schema,expected",
        [
            (None, None, True),
            (None, "public", False),
            ("public", "public", True),
            ("analytics", "public", False),
        ],
    )
    def test_schema_matches(self, view_schema, loop_schema, expected):
        assert _schema_matches(view_schema, loop_schema) is expected

    def test_no_duplicate_ops_for_none_public_schemas(self):
        """A None-schema view is processed only in the None loop, not 'public'."""
        both_match = (
            _schema_matches(None, None) and _schema_matches(None, "public")
        )
        assert not both_match


# ===========================================================================
# Dependency resolution
# ===========================================================================

class TestDependIndependentViews:
    """Independent views (no deps) maintain any order."""

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
    """View-on-view dependency ordering."""

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
    """Multi-level dependency chains (A → B → C)."""

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


class TestDependCircular:
    """Circular dependencies raise ValueError."""

    def test_simple_cycle_raises_value_error(self):
        views = [
            ViewRecord(name="view_a", selectable="SELECT * FROM view_b"),
            ViewRecord(name="view_b", selectable="SELECT * FROM view_a"),
        ]
        with pytest.raises(ValueError, match="[Cc]ircular"):
            resolve_create_order(views, db_views={})

    def test_three_way_cycle_raises_value_error(self):
        views = [
            ViewRecord(name="v_a", selectable="SELECT * FROM v_b"),
            ViewRecord(name="v_b", selectable="SELECT * FROM v_c"),
            ViewRecord(name="v_c", selectable="SELECT * FROM v_a"),
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


class TestDependDropOrder:
    """Drop order is the exact reverse of create order."""

    def test_drop_is_reverse_of_create(self):
        views = [
            ViewRecord(name="top", selectable="SELECT * FROM mid"),
            ViewRecord(name="mid", selectable="SELECT * FROM base"),
            ViewRecord(name="base", selectable="SELECT 1"),
        ]
        create = resolve_create_order(views, db_views={})
        drop = resolve_drop_order(views, db_views={})
        assert [v.name for v in drop] == list(
            reversed([v.name for v in create])
        )


class TestDependMaterializedViews:
    """Materialized views with dependencies."""

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
    """db_views are used as pre-existing dependencies."""

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
    """db_views=None is treated as {}."""

    def test_none_db_views(self):
        vr = ViewRecord(name="solo", selectable="SELECT 1 AS col")
        assert resolve_create_order([vr], db_views=None) == [vr]
        assert resolve_drop_order([vr], db_views=None) == [vr]
        assert _build_dependency_graph([vr], {}) == {"solo": set()}


class TestDependWordBoundary:
    """Dependency detection uses word-boundary matching."""

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
        class _BrokenClauseElement:
            """Raises CompileError when compiled against a dialect."""

            def compile(self, **kw):
                if "dialect" in kw:
                    raise sa.exc.CompileError(
                        "Cannot compile clause element for this dialect"
                    )
                return "SELECT 1 AS id"

        records = [
            ViewRecord(
                name="broken_view",
                selectable=_BrokenClauseElement(),
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
        assert result == records, (
            f"_safe_resolve must fall back to model order on CompileError; "
            f"got {result}"
        )
        # A warning must be logged (do not silently swallow).
        warnings = [
            rec for rec in caplog.records if rec.levelno >= logging.WARNING
        ]
        assert warnings, (
            f"Expected a warning when falling back to model order; got "
            f"{[(rec.levelname, rec.message) for rec in caplog.records]}"
        )


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


# ---------------------------------------------------------------------------
# Infrastructure tests — verify fixtures/helpers work in isolation
# ---------------------------------------------------------------------------

@pytest.mark.infrastructure
class TestAutogenerateFixture:
    """Tests for the alembic_config fixture itself."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_creates_valid_alembic_environment(self, alembic_config, connection):
        metadata = sa.MetaData()
        cfg = alembic_config(metadata)

        assert isinstance(cfg, config.Config)
        assert cfg.attributes["connection"] is connection
        assert cfg.attributes["target_metadata"] is metadata
        script_location = Path(cfg.get_main_option("script_location"))
        assert script_location.exists()
        assert (script_location / "env.py").exists()


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


class TestIntegrationNewView:
    """Integration: autogenerate detects new view definition."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_view_detected_and_rendered(self, connection, alembic_config):
        _drop_views(connection, _INT_TEST_VIEW_NAMES)
        _drop_base_table(connection)
        _create_base_table(connection)
        try:
            register_view_comparator()
            metadata = sa.MetaData()
            create_view(
                "int_test_new_view", sa.select(_int_base_table.c.id), metadata
            )
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_op(code, "create_view")
            assert "int_test_new_view" in code
        finally:
            _drop_views(connection, _INT_TEST_VIEW_NAMES)
            _drop_base_table(connection)


class TestIntegrationNewMV:
    """Integration: autogenerate detects new materialized view."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_mv_detected_and_rendered(self, connection, alembic_config):
        _drop_views(connection, _INT_TEST_VIEW_NAMES)
        _drop_base_table(connection)
        _create_base_table(connection)
        try:
            register_view_comparator()
            metadata = sa.MetaData()
            create_materialized_view(
                "int_test_new_mv", sa.select(_int_base_table.c.id), metadata
            )
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_op(code, "create_materialized_view")
            assert "int_test_new_mv" in code
        finally:
            _drop_views(connection, _INT_TEST_VIEW_NAMES)
            _drop_base_table(connection)


class TestIntegrationRemoval:
    """Integration: autogenerate detects view/MV removed from model."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_view_generates_drop(self, connection, alembic_config):
        _drop_views(connection, _INT_TEST_VIEW_NAMES)
        _drop_base_table(connection)
        _create_base_table(connection)
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_drop_view AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        try:
            register_view_comparator()
            metadata = sa.MetaData()
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_op(code, "drop_view")
        finally:
            _drop_views(connection, _INT_TEST_VIEW_NAMES)
            _drop_base_table(connection)

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_mv_generates_drop(self, connection, alembic_config):
        _drop_views(connection, _INT_TEST_VIEW_NAMES)
        _drop_base_table(connection)
        _create_base_table(connection)
        connection.execute(
            sa.text(
                "CREATE MATERIALIZED VIEW int_test_drop_mv "
                "AS SELECT id FROM _cmp_test_base WITH NO DATA"
            )
        )
        connection.commit()
        try:
            register_view_comparator()
            metadata = sa.MetaData()
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_op(code, "drop_materialized_view")
        finally:
            _drop_views(connection, _INT_TEST_VIEW_NAMES)
            _drop_base_table(connection)


class TestIntegrationDefinitionChange:
    """Integration: autogenerate detects view/MV definition change."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_view_generates_replace(self, connection, alembic_config):
        _drop_views(connection, _INT_TEST_VIEW_NAMES)
        _drop_base_table(connection)
        _create_base_table(connection)
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_change_view AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        try:
            register_view_comparator()
            metadata = sa.MetaData()
            create_view(
                "int_test_change_view",
                sa.select(_int_base_table.c.id, _int_base_table.c.name),
                metadata,
            )
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_op(code, "replace_view")
        finally:
            _drop_views(connection, _INT_TEST_VIEW_NAMES)
            _drop_base_table(connection)

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_mv_generates_replace(self, connection, alembic_config):
        _drop_views(connection, _INT_TEST_VIEW_NAMES)
        _drop_base_table(connection)
        _create_base_table(connection)
        connection.execute(
            sa.text(
                "CREATE MATERIALIZED VIEW int_test_change_mv "
                "AS SELECT id FROM _cmp_test_base WITH NO DATA"
            )
        )
        connection.commit()
        try:
            register_view_comparator()
            metadata = sa.MetaData()
            create_materialized_view(
                "int_test_change_mv",
                sa.select(_int_base_table.c.id, _int_base_table.c.name),
                metadata,
            )
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_op(code, "replace_materialized_view")
        finally:
            _drop_views(connection, _INT_TEST_VIEW_NAMES)
            _drop_base_table(connection)


class TestIntegrationNoOp:
    """Integration: no view ops generated when view definitions match."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_unchanged_view_no_ops(self, connection, alembic_config):
        _drop_views(connection, _INT_TEST_VIEW_NAMES)
        _drop_base_table(connection)
        _create_base_table(connection)
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_same_view AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        try:
            register_view_comparator()
            metadata = sa.MetaData()
            create_view(
                "int_test_same_view", sa.select(_int_base_table.c.id), metadata
            )
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_op(code, "create_view", expected=False)
            assert_op(code, "drop_view", expected=False)
            assert_op(code, "replace_view", expected=False)
        finally:
            _drop_views(connection, _INT_TEST_VIEW_NAMES)
            _drop_base_table(connection)


class TestIntegrationDependencyOrdering:
    """Integration: views are created in dependency order."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_dependent_view_created_after_dependency(
        self, connection, alembic_config
    ):
        _drop_views(connection, _INT_TEST_VIEW_NAMES)
        _drop_base_table(connection)
        _create_base_table(connection)
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_view_a AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        try:
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
        finally:
            _drop_views(connection, _INT_TEST_VIEW_NAMES)
            _drop_base_table(connection)


# ===========================================================================
# Public API
# ===========================================================================

class TestPublicAPIImportable:
    """Public API symbols are importable from sqlalchemy_utils.alembic."""

    @pytest.mark.parametrize(
        "op_import_path,op_name,args,kwargs,expected_definition",
        [
            ("sqlalchemy_utils.alembic:CreateViewOp", "test_view", ("test_view", "SELECT 1"), {}, "SELECT 1"),
            ("sqlalchemy_utils.alembic:DropViewOp", "test_view", ("test_view",), {}, None),
            ("sqlalchemy_utils.alembic:ReplaceViewOp", "test_view", ("test_view", "SELECT 2"), {}, "SELECT 2"),
            ("sqlalchemy_utils.alembic:CreateMaterializedViewOp", "test_mv", ("test_mv", "SELECT 1"), {}, "SELECT 1"),
            ("sqlalchemy_utils.alembic:DropMaterializedViewOp", "test_mv", ("test_mv",), {"cascade": False}, None),
            ("sqlalchemy_utils.alembic:ReplaceMaterializedViewOp", "test_mv", ("test_mv", "SELECT 2"), {}, "SELECT 2"),
            ("sqlalchemy_utils.alembic:compare_views", None, None, {}, None),
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
            "compare_views", "get_database_materialized_views",
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

    def test_internal_import_view_record(self):
        from sqlalchemy_utils.view_record import ViewRecord as VR

        assert VR is not None


# ===========================================================================
# Import safety
# ===========================================================================

class TestImportSafety:
    """Import behavior under edge conditions."""

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
    """DDL element SQL formatting."""

    def test_drop_view_no_trailing_space(self):
        """DropView with cascade=False emits no trailing whitespace."""
        drop = DropView("my_view", cascade=False)
        sql = _compile_ddl(drop)
        assert sql.rstrip() == sql

    def test_refresh_materialized_view_with_schema(self):
        """refresh_materialized_view forwards schema to the DDL element."""
        session = mock.MagicMock()
        refresh_materialized_view(
            session, "my_mv", concurrently=False, schema="analytics"
        )

        assert session.execute.call_count == 1
        executed = session.execute.call_args[0][0]
        assert isinstance(executed, RefreshMaterializedView)
        assert executed.name == "my_mv"
        assert executed.schema == "analytics"
        assert executed.concurrently is False

        engine = sa.create_engine("sqlite:///:memory:")
        compiled = str(executed.compile(dialect=engine.dialect))
        assert "analytics" in compiled

    def test_refresh_materialized_view_concurrently(self):
        """refresh_materialized_view with concurrently=True emits CONCURRENTLY."""
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
        assert "CONCURRENTLY" in compiled

    def test_refresh_param_order_consistency(self):
        """RefreshMaterializedView and refresh_materialized_view agree on param order."""
        cls_sig = inspect.signature(RefreshMaterializedView.__init__)
        fn_sig = inspect.signature(refresh_materialized_view)

        cls_params = [p for p in cls_sig.parameters.values() if p.name not in {"self"}]
        fn_params = [
            p for p in fn_sig.parameters.values() if p.name not in {"session"}
        ]

        cls_order = [
            p.name for p in cls_params if p.name in {"schema", "concurrently"}
        ]
        fn_order = [
            p.name for p in fn_params if p.name in {"schema", "concurrently"}
        ]

        assert cls_order == fn_order


# ===========================================================================
# Schema resolution (ViewMixin)
# ===========================================================================

class TestSchemaResolution:
    """ViewMixin schema resolution behavior."""

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
        assert SimpleView._resolved_view_schema is None
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
    """ViewMixin DDL/listener behavior."""

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
    """create_view/create_materialized_view auto-register ViewRecords."""

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
# Documentation
# ===========================================================================

class TestDocumentation:
    """Docstring contracts on public API."""

    def test_create_view_documents_no_indexes(self):
        """create_view docstring documents the indexes/aliases asymmetry."""
        from sqlalchemy_utils.view import create_view as cv

        assert cv.__doc__ is not None
        doc = cv.__doc__.lower()
        assert "index" in doc or "materialized" in doc

    def test_materialized_view_ops_document_pg_only(self):
        """Materialized view Op classes document PostgreSQL-only semantics."""
        for cls in [
            CreateMaterializedViewOp,
            DropMaterializedViewOp,
            ReplaceMaterializedViewOp,
        ]:
            doc = (cls.__doc__ or "").lower()
            assert "postgresql" in doc or "postgres" in doc


# ===========================================================================
# Interface audit fixes
# ===========================================================================

def test_create_view_works_without_alembic_installed():
    """create_view() must work when alembic is not installed.

    ViewRecord has no alembic dependency, so importing it should not
    fail when alembic is absent. The try/except guard in view.py sets
    ViewRecord=None, which then crashes when create_view() calls it.
    """
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
    from unittest.mock import MagicMock
    from sqlalchemy_utils.alembic.operations import DropMaterializedViewOp
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


class TestCrossSchemaDedup:
    """Dedup and ordering across schemas with same-name views."""

    def test_keyword_filter_preserves_dependency_order(self):
        """A view named 'data' (a SQL keyword) must still be ordered
        before views that reference it."""
        views = [
            ViewRecord(name="data", selectable="SELECT 1 AS col"),
            ViewRecord(name="report", selectable="SELECT * FROM data"),
        ]
        order = resolve_create_order(views, db_views={})
        names = [v.name for v in order]
        assert names.index("data") < names.index("report")

    def test_resolve_create_order_preserves_same_name_diff_schema(self):
        """Two views with the same name in different schemas must both
        appear in the ordered output."""
        views = [
            ViewRecord(name="v", schema="a", selectable="SELECT 1"),
            ViewRecord(name="v", schema="b", selectable="SELECT 1"),
        ]
        result = resolve_create_order(views, db_views={})
        assert len(result) == 2


# ===========================================================================
# Interface audit: migration refresh, dead listener, mapped_column guard
# ===========================================================================

class TestInterfaceAuditFixes:
    """Interface audit fixes for migration refresh, dead listener, SA2 guard."""

    def test_op_refresh_materialized_view_exists(self):
        """op.refresh_materialized_view should exist for use inside migrations."""
        from sqlalchemy_utils.alembic.operations import (
            RefreshMaterializedViewOp,
        )
        assert hasattr(RefreshMaterializedViewOp, "refresh_materialized_view")
        assert callable(RefreshMaterializedViewOp.refresh_materialized_view)

    def test_refresh_reverse_raises(self):
        """RefreshMaterializedViewOp.reverse() raises NotImplementedError.

        REFRESH MATERIALIZED VIEW is not meaningfully reversible (you
        cannot "un-refresh" a materialized view), so reverse() must
        raise rather than silently emit another REFRESH in the downgrade.
        """
        from sqlalchemy_utils.alembic.operations import (
            RefreshMaterializedViewOp,
        )
        op = RefreshMaterializedViewOp("mv")
        with pytest.raises(NotImplementedError, match="not meaningfully reversible"):
            op.reverse()

    def test_viewmixin_init_subclass_catches_mapped_column_without_tablename(self):
        """ViewMixin.__init_subclass__ should catch mapped_column usage without __tablename__ and give a helpful error."""
        from sqlalchemy_utils.view_mixin import ViewMixin
        from sqlalchemy.orm import declarative_base, Mapped, mapped_column
        Base = declarative_base()
        with pytest.raises(TypeError, match="__tablename__"):
            class BadView(ViewMixin, Base):
                id: Mapped[int] = mapped_column(primary_key=True)
                __view_selectable__ = sa.select(sa.column("id", sa.Integer))


# ===========================================================================
# Cascade-on-drop warning
# ===========================================================================

class TestCascadeOnDropWarning:
    """When autogenerate drops a view that has dependents, warn the user."""

    @pytest.fixture
    def cascade_mock_setup(self):
        import sqlalchemy_utils.alembic.comparator as comparator_module

        def _run(db_views: dict, dependent_views: dict):
            metadata = MagicMock()
            metadata.info = {"sqlalchemy_utils_views": []}

            autogen_context = MagicMock()
            autogen_context.connection = MagicMock()
            autogen_context.connection.dialect.name = "postgresql"
            autogen_context.metadata = metadata

            upgrade_ops = MagicMock()
            upgrade_ops.ops = []

            with patch.object(
                comparator_module, "get_database_views",
                return_value=db_views,
            ), patch.object(
                comparator_module, "get_database_materialized_views",
                return_value={},
            ), patch.object(
                comparator_module, "_canonicalize_all_views",
                return_value=({}, {}, set()),
            ), patch.object(
                comparator_module, "get_dependent_views",
                return_value=dependent_views,
            ), patch.object(
                comparator_module, "log",
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
        assert len(warning_calls) > 0, (
            f"Expected warning about dependents when dropping base_view, "
            f"got warnings: {mock_log.warning.call_args_list}"
        )

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
        assert len(warning_calls) == 0, (
            f"Should not warn about dependents for lonely_view, got: {warning_calls}"
        )

    def test_drop_op_still_generated_even_with_dependents(
        self, cascade_mock_setup
    ):
        """The DropViewOp should still be generated even if the view has dependents
        (warn, don't block)."""
        from sqlalchemy_utils.alembic.operations import DropViewOp

        upgrade_ops, _mock_log = cascade_mock_setup(
            db_views={"base_view": "SELECT 1 AS col"},
            dependent_views={("dependent_view", None): "SELECT * FROM base_view"},
        )

        drop_ops = [op for op in upgrade_ops.ops if isinstance(op, DropViewOp)]
        assert any(op.name == "base_view" for op in drop_ops), (
            f"DropViewOp for base_view should still be generated, got ops: {upgrade_ops.ops}"
        )


# ===========================================================================
# cascade_on_drop propagation
# ===========================================================================

class TestCascadeOnDropPropagation:
    """compare_views should propagate ViewRecord.cascade_on_drop to the
    generated DropViewOp / DropMaterializedViewOp ``cascade`` param."""

    @staticmethod
    def _make_context(model_views):
        metadata = MagicMock()
        metadata.info = {"sqlalchemy_utils_views": model_views}

        autogen_context = MagicMock()
        autogen_context.connection = MagicMock()
        autogen_context.connection.dialect.name = 'postgresql'
        autogen_context.metadata = metadata

        upgrade_ops = MagicMock()
        upgrade_ops.ops = []
        return autogen_context, upgrade_ops

    @staticmethod
    def _patch_comparator(db_views, db_mvs=None, canonical_return=({}, {}, set())):
        """Patch all comparator module dependencies for one compare_views call."""
        import sqlalchemy_utils.alembic.comparator as comparator_module

        if db_mvs is None:
            db_mvs = {}
        return (
            patch.object(comparator_module, 'get_database_views',
                         return_value=db_views),
            patch.object(comparator_module, 'get_database_materialized_views',
                         return_value=db_mvs),
            patch.object(comparator_module, '_canonicalize_all_views',
                        return_value=canonical_return),
            patch.object(comparator_module, 'get_dependent_views',
                        return_value={}),
            patch.object(comparator_module, 'log'),
        )

    def test_drop_view_propagates_cascade_false(self):
        """DropViewOp.cascade=False when ViewRecord.cascade_on_drop=False."""
        vr = ViewRecord(
            name="v_no_cascade", selectable="SELECT 1 AS col",
            schema=None, cascade_on_drop=False,
        )
        autogen_context, upgrade_ops = self._make_context([vr])

        patches = self._patch_comparator(
            db_views={"v_no_cascade": "SELECT 1 AS col"},
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            import sqlalchemy_utils.alembic.comparator as comparator_module
            comparator_module.compare_views(autogen_context, upgrade_ops, [None])

        drop_ops = [op for op in upgrade_ops.ops if isinstance(op, DropViewOp)]
        assert len(drop_ops) == 1, f"expected one DropViewOp, got {drop_ops}"
        assert drop_ops[0].name == "v_no_cascade"
        assert drop_ops[0].cascade is False, (
            f"DropViewOp.cascade should be False when "
            f"ViewRecord.cascade_on_drop=False, got {drop_ops[0].cascade!r}"
        )

    def test_drop_view_defaults_to_true_when_no_record(self):
        """DropViewOp.cascade defaults to True when no model ViewRecord exists."""
        autogen_context, upgrade_ops = self._make_context([])

        patches = self._patch_comparator(
            db_views={"orphan_view": "SELECT 1 AS col"},
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            import sqlalchemy_utils.alembic.comparator as comparator_module
            comparator_module.compare_views(autogen_context, upgrade_ops, [None])

        drop_ops = [op for op in upgrade_ops.ops if isinstance(op, DropViewOp)]
        assert len(drop_ops) == 1
        assert drop_ops[0].cascade is True

    def test_drop_materialized_view_propagates_cascade_false(self):
        """DropMaterializedViewOp.cascade=False when cascade_on_drop=False."""
        vr = ViewRecord(
            name="mv_no_cascade", selectable="SELECT 1 AS col",
            schema=None, materialized=True, cascade_on_drop=False,
        )
        autogen_context, upgrade_ops = self._make_context([vr])

        patches = self._patch_comparator(
            db_views={},
            db_mvs={"mv_no_cascade": "SELECT 1 AS col"},
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            import sqlalchemy_utils.alembic.comparator as comparator_module
            comparator_module.compare_views(autogen_context, upgrade_ops, [None])

        drop_ops = [
            op for op in upgrade_ops.ops
            if isinstance(op, DropMaterializedViewOp)
        ]
        assert len(drop_ops) == 1, (
            f"expected one DropMaterializedViewOp, got {drop_ops}"
        )
        assert drop_ops[0].name == "mv_no_cascade"
        assert drop_ops[0].cascade is False, (
            f"DropMaterializedViewOp.cascade should be False when "
            f"ViewRecord.cascade_on_drop=False, got {drop_ops[0].cascade!r}"
        )


# ===========================================================================
# Cross-schema same-name view handling
# ===========================================================================

class TestCrossSchemaSameNameBothOps:
    """When two schemas each have a model view with the same name, both
    create ops must survive — the second must not overwrite the first
    in the ``create_by_name`` / ``drop_by_name`` dicts.
    """

    def test_cross_schema_same_name_both_ops(self):
        import sqlalchemy_utils.alembic.comparator as comparator_module

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

        metadata = MagicMock()
        metadata.info = {"sqlalchemy_utils_views": model_views}

        autogen_context = MagicMock()
        autogen_context.connection = MagicMock()
        autogen_context.connection.dialect.name = "postgresql"
        autogen_context.metadata = metadata

        upgrade_ops = MagicMock()
        upgrade_ops.ops = []

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
        assert len(create_ops) == 2, (
            f"Expected 2 CreateViewOps (one per schema), got {len(create_ops)}: "
            f"{[(op.name, op.schema) for op in create_ops]}"
        )

        schemas_seen = {(op.name, op.schema) for op in create_ops}
        assert ("foo", "public") in schemas_seen, (
            f"Missing CreateViewOp for (foo, public); got {schemas_seen}"
        )
        assert ("foo", "analytics") in schemas_seen, (
            f"Missing CreateViewOp for (foo, analytics); got {schemas_seen}"
        )


# ===========================================================================
# Regression: schema=None asymmetric comparison
# ===========================================================================

class TestSchemaNoneNoFalseDrop:
    """schemas=[None] must not produce false DropViewOps for non-default schemas."""

    def test_schema_none_no_false_drop(self):
        import sqlalchemy_utils.alembic.comparator as comparator_module
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
        assert analytics_drops == [], (
            f"Regression: false DropViewOp emitted for "
            f"analytics_view (schema=analytics) when schemas=[None]. "
            f"Got drop ops: {[(op.name, op.schema) for op in drop_ops]}"
        )


# ===========================================================================
# Regression: MV canonicalization DROP must use CASCADE for dependent views
# ===========================================================================

_MV_CASCADE_TEST_VIEW_NAMES = ["mv_cascade_test_mv", "mv_cascade_test_dep_view"]


def _drop_mv_cascade_test_views(connection):
    """Drop any leftover views/MVs created by this regression test."""
    # Drop dependent view first (depends on the MV).
    try:
        connection.execute(
            sa.text("DROP VIEW IF EXISTS mv_cascade_test_dep_view CASCADE")
        )
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    try:
        connection.execute(
            sa.text("DROP MATERIALIZED VIEW IF EXISTS mv_cascade_test_mv CASCADE")
        )
    except sa.exc.SQLAlchemyError:
        connection.rollback()
    connection.commit()


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

    def test_build_create_sql_uses_cascade_for_materialized_view(self):
        """``_build_create_sql`` must emit CASCADE for the MV DROP so that
        dependent views do not block canonicalization."""
        connection = MagicMock()
        connection.dialect = sa.dialects.postgresql.dialect()
        with patch(
            "sqlalchemy_utils.alembic.comparator.ViewRecord.compiled_definition",
            return_value="SELECT 1 AS id",
        ):
            vr = ViewRecord(
                name="mv_cascade_test_mv",
                selectable="SELECT 1 AS id",
                schema=None,
                materialized=True,
            )
            stmts = _build_create_sql(connection, vr)

        # _build_create_sql now returns a list; join for substring checks.
        sql = " ".join(stmts)

        assert "DROP MATERIALIZED VIEW IF EXISTS" in sql.upper(), (
            f"Expected DROP MATERIALIZED VIEW IF EXISTS in SQL: {sql!r}"
        )
        assert "CASCADE" in sql.upper(), (
            f"Regression: _build_create_sql must emit CASCADE for MV "
            f"canonicalization DROP so dependent views do not block it. "
            f"SQL: {sql!r}"
        )
        # CASCADE must be on the DROP, not just somewhere in CREATE.
        drop_part = sql.upper().split("CREATE MATERIALIZED VIEW")[0]
        assert "CASCADE" in drop_part, (
            f"CASCADE must be on the DROP clause, not the CREATE. SQL: {sql!r}"
        )

    @pytest.mark.infrastructure
    @pytest.mark.usefixtures("postgresql_dsn")
    def test_mv_with_dependent_view_definition_change_detected(self, connection):
        """MV with a dependent view: definition change is detected, not skipped.

        Pre-create an MV with an old definition and a regular view that
        selects from it. The model defines the MV with a NEW definition.
        Without CASCADE the DROP fails (dependent view blocks it), the MV is
        skipped, and the change is silently missed. With CASCADE the DROP
        succeeds inside the savepoint, the MV is recreated, and the change is
        detected as a ``ReplaceMaterializedViewOp``.
        """
        _create_base_table(connection)
        _drop_mv_cascade_test_views(connection)
        try:
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
            assert len(replace_ops) == 1, (
                f"Regression: MV 'mv_cascade_test_mv' with a dependent view "
                f"was silently skipped during canonicalization (DROP without "
                f"CASCADE fails when a dependent view exists). Expected a "
                f"ReplaceMaterializedViewOp; got {replace_ops}. "
                f"All ops: {[(type(o).__name__, o.name) for o in upgrade_ops.ops]}"
            )
        finally:
            _drop_mv_cascade_test_views(connection)
            _drop_base_table(connection)

    @pytest.mark.infrastructure
    @pytest.mark.usefixtures("postgresql_dsn")
    def test_mv_with_dependent_view_not_in_skipped(self, connection):
        """Directly verify ``_canonicalize_all_views`` does not skip an MV
        that has a dependent view in the database."""
        _create_base_table(connection)
        _drop_mv_cascade_test_views(connection)
        try:
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

            assert "mv_cascade_test_mv" not in skipped, (
                f"Regression: MV was skipped during canonicalization because "
                f"the DROP failed without CASCADE (dependent view blocks it). "
                f"skipped={skipped}"
            )
            assert "mv_cascade_test_mv" in mv_defs, (
                f"Regression: MV definition missing from mv_defs. "
                f"mv_defs={mv_defs}, skipped={skipped}"
            )
        finally:
            _drop_mv_cascade_test_views(connection)
            _drop_base_table(connection)


# ===========================================================================
# Regression: multi-statement sa.text() is not portable across drivers
# ===========================================================================

class TestBuildCreateSqlReturnsListForMv:
    """``_build_create_sql`` must return a list of SQL strings for MVs.

    A materialized view requires a ``DROP`` then a ``CREATE``. Previously
    these were concatenated into one string passed to a single
    ``connection.execute(sa.text(sql))`` call. Multi-statement ``text()``
    relies on the simple-query protocol, supported by psycopg2 but not
    asyncpg / some drivers. Returning a list lets the caller execute each
    statement separately, keeping the canonicalization portable.

    For regular (non-materialized) views a single-statement string list is
    returned — the public contract is now always a list of strings.
    """

    def test_mv_returns_two_statements_drop_then_create(self):
        connection = MagicMock()
        connection.dialect = sa.dialects.postgresql.dialect()
        with patch(
            "sqlalchemy_utils.alembic.comparator.ViewRecord.compiled_definition",
            return_value="SELECT 1 AS id",
        ):
            vr = ViewRecord(
                name="mv_portable",
                selectable="SELECT 1 AS id",
                schema=None,
                materialized=True,
            )
            result = _build_create_sql(connection, vr)

        assert isinstance(result, list), (
            f"_build_create_sql must return a list for MVs; got {type(result)}"
        )
        assert len(result) == 2, (
            f"MV must produce two statements (DROP + CREATE); got {result!r}"
        )
        drop_sql = result[0].upper()
        create_sql = result[1].upper()
        assert "DROP MATERIALIZED VIEW IF EXISTS" in drop_sql, (
            f"First statement must be the DROP; got {result[0]!r}"
        )
        assert "CASCADE" in drop_sql, (
            f"DROP must use CASCADE so dependent views do not block it; "
            f"got {result[0]!r}"
        )
        assert "CREATE MATERIALIZED VIEW" in create_sql, (
            f"Second statement must be the CREATE; got {result[1]!r}"
        )
        assert "WITH NO DATA" in create_sql, (
            f"CREATE must include WITH NO DATA; got {result[1]!r}"
        )

    def test_regular_view_returns_single_statement_list(self):
        connection = MagicMock()
        connection.dialect = sa.dialects.postgresql.dialect()
        with patch(
            "sqlalchemy_utils.alembic.comparator.ViewRecord.compiled_definition",
            return_value="SELECT 1 AS id",
        ):
            vr = ViewRecord(
                name="v_portable",
                selectable="SELECT 1 AS id",
                schema=None,
                materialized=False,
            )
            result = _build_create_sql(connection, vr)

        assert isinstance(result, list), (
            f"_build_create_sql must always return a list; got {type(result)}"
        )
        assert len(result) == 1, (
            f"Regular view must produce a single statement; got {result!r}"
        )
        assert "CREATE OR REPLACE VIEW" in result[0].upper(), (
            f"Statement must be CREATE OR REPLACE VIEW; got {result[0]!r}"
        )


class TestMvCanonicalizationUsesSeparateExecuteCalls:
    """MV canonicalization must issue separate ``connection.execute`` calls.

    The caller of ``_build_create_sql`` (the canonicalization loop) must
    execute each returned statement separately rather than passing a
    multi-statement string to a single ``connection.execute(sa.text(...))``.
    Multi-statement ``text()`` relies on the simple-query protocol, which
    is driver-specific (psycopg2 supports it; asyncpg does not). Executing
    statements individually is portable across drivers.
    """

    def test_mv_canonicalization_makes_two_execute_calls(self):
        """For an MV, the canonicalization loop issues one execute call for
        the DROP and one for the CREATE — never a single multi-statement
        call. A mock connection records every ``sa.text`` executed."""
        execute_calls: list[str] = []

        def _execute(stmt, *args, **kwargs):
            text = getattr(stmt, "text", str(stmt))
            execute_calls.append(text)
            # ROLLBACK TO / RELEASE / SAVEPOINT succeed; DROP+CREATE succeed.
            return MagicMock()

        connection = MagicMock()
        connection.dialect = sa.dialects.postgresql.dialect()
        connection.execute.side_effect = _execute

        vr = ViewRecord(
            name="mv_separate_exec",
            selectable="SELECT 1 AS id",
            schema=None,
            materialized=True,
        )

        with patch(
            "sqlalchemy_utils.alembic.comparator.get_database_views",
            return_value={},
        ), patch(
            "sqlalchemy_utils.alembic.comparator.get_database_materialized_views",
            return_value={},
        ):
            _canonicalize_all_views(connection, [vr], db_views_for_deps=None)

        # Find the DROP and CREATE statements issued for the MV.
        drop_calls = [
            s for s in execute_calls
            if "DROP MATERIALIZED VIEW" in s.upper()
        ]
        create_calls = [
            s for s in execute_calls
            if "CREATE MATERIALIZED VIEW" in s.upper()
        ]
        assert len(drop_calls) == 1, (
            f"Expected exactly one DROP statement executed; got {drop_calls}"
        )
        assert len(create_calls) == 1, (
            f"Expected exactly one CREATE statement executed; got "
            f"{create_calls}"
        )
        # No single execute call may contain BOTH DROP and CREATE — that
        # would be a multi-statement call.
        multi_statement_calls = [
            s for s in execute_calls
            if "DROP MATERIALIZED VIEW" in s.upper()
            and "CREATE MATERIALIZED VIEW" in s.upper()
        ]
        assert multi_statement_calls == [], (
            f"DROP and CREATE must be in separate execute calls (not a "
            f"multi-statement text()). Found combined call(s): "
            f"{multi_statement_calls}"
        )


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
    def _make_context_with_existing_ops(existing_ops):
        """Build a mock autogen_context + upgrade_ops pre-populated with
        ``existing_ops`` (non-view ops that Alembic's own comparators may
        have already appended)."""
        metadata = MagicMock()
        metadata.info = {"sqlalchemy_utils_views": []}

        autogen_context = MagicMock()
        autogen_context.connection = MagicMock()
        autogen_context.connection.dialect.name = "postgresql"
        autogen_context.metadata = metadata

        upgrade_ops = MagicMock()
        upgrade_ops.ops = list(existing_ops)
        return autogen_context, upgrade_ops

    @staticmethod
    def _patch_comparator(db_views=None, db_mvs=None):
        import sqlalchemy_utils.alembic.comparator as comparator_module

        if db_views is None:
            db_views = {}
        if db_mvs is None:
            db_mvs = {}
        return (
            patch.object(comparator_module, "get_database_views",
                         return_value=db_views),
            patch.object(comparator_module, "get_database_materialized_views",
                         return_value=db_mvs),
            patch.object(comparator_module, "_canonicalize_all_views",
                        return_value=({}, {}, set())),
            patch.object(comparator_module, "get_dependent_views",
                        return_value={}),
            patch.object(comparator_module, "log"),
        )

    def test_create_table_op_preserved_alongside_create_view_op(self):
        """A CreateTableOp already in upgrade_ops.ops must survive the
        dedup loop when a CreateViewOp is also emitted.

        Without the guard, both ops have no ``name``-based key collision
        risk individually, but multiple non-view ops with ``name=None``
        collide with each other AND with any view op that happens to share
        the same (family, None, None) key. The critical regression: two
        distinct non-view ops (CreateTableOp + DropTableOp) get deduped
        down to one because both yield ``name=None``.
        """
        from alembic.operations.ops import (
            CreateTableOp, DropTableOp,
        )
        from sqlalchemy import Column, Integer, MetaData, Table

        md = MetaData()
        table_a = Table("table_a", md, Column("id", Integer, primary_key=True))
        table_b = Table("table_b", md, Column("id", Integer, primary_key=True))
        create_table = CreateTableOp.from_table(table_a)
        drop_table = DropTableOp.from_table(table_b)

        autogen_context, upgrade_ops = self._make_context_with_existing_ops(
            [create_table, drop_table]
        )

        patches = self._patch_comparator()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            compare_views(autogen_context, upgrade_ops, [None])

        op_types = [type(op).__name__ for op in upgrade_ops.ops]
        assert "CreateTableOp" in op_types, (
            f"CreateTableOp was silently dropped by the dedup loop. "
            f"Got ops: {op_types}"
        )
        assert "DropTableOp" in op_types, (
            f"DropTableOp was silently dropped by the dedup loop. "
            f"Got ops: {op_types}"
        )

    def test_multiple_non_view_ops_all_preserved(self):
        """Multiple CreateTableOps (all with name=None) must ALL survive.

        This is the core regression: without the guard, N CreateTableOps
        all dedup to the same key ("create_or_replace", None, None) and
        only the first survives. Silent data loss.
        """
        from alembic.operations.ops import CreateTableOp
        from sqlalchemy import Column, Integer, MetaData, Table

        md = MetaData()
        tables = [
            CreateTableOp.from_table(Table(f"t{i}", md, Column("id", Integer)))
            for i in range(5)
        ]

        autogen_context, upgrade_ops = self._make_context_with_existing_ops(
            tables
        )

        patches = self._patch_comparator()
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            compare_views(autogen_context, upgrade_ops, [None])

        create_table_count = sum(
            1 for op in upgrade_ops.ops
            if isinstance(op, CreateTableOp)
        )
        assert create_table_count == 5, (
            f"Expected all 5 CreateTableOps to survive dedup, got "
            f"{create_table_count}. Silent data loss in dedup loop. "
            f"Ops: {[type(op).__name__ for op in upgrade_ops.ops]}"
        )

    def test_non_view_op_preserved_alongside_view_drop(self):
        """A CreateTableOp must survive when compare_views also emits a
        DropViewOp for a view present in the DB but not the model.

        Both ops have ``schema=None``; without the guard the CreateTableOp
        (family "create_or_replace", name None) and the DropViewOp (family
        "drop", name "v") would not collide, but two DropViewOps for
        different views would both get name from the op — the real risk is
        multiple non-view ops colliding with each other, covered above. This
        test confirms the mixed scenario (non-view + view op) is stable.
        """
        from alembic.operations.ops import CreateTableOp
        from sqlalchemy import Column, Integer, MetaData, Table

        md = MetaData()
        create_table = CreateTableOp.from_table(
            Table("real_table", md, Column("id", Integer))
        )

        autogen_context, upgrade_ops = self._make_context_with_existing_ops(
            [create_table]
        )

        # "old_view" is in DB but not in model -> compare_views emits DropViewOp
        patches = self._patch_comparator(
            db_views={"old_view": "SELECT 1 AS col"}
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            compare_views(autogen_context, upgrade_ops, [None])

        has_create_table = any(
            isinstance(op, CreateTableOp) for op in upgrade_ops.ops
        )
        has_drop_view = any(
            isinstance(op, DropViewOp) for op in upgrade_ops.ops
        )
        assert has_create_table, (
            f"CreateTableOp lost in dedup. "
            f"Ops: {[type(op).__name__ for op in upgrade_ops.ops]}"
        )
        assert has_drop_view, (
            f"DropViewOp lost in dedup. "
            f"Ops: {[type(op).__name__ for op in upgrade_ops.ops]}"
        )

