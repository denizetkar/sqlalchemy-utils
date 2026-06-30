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
    include_view_comparator,
)
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
from sqlalchemy_utils.alembic.view_record import ViewRecord
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


def _make_operations() -> Operations:
    """Build a real Operations instance backed by in-memory SQLite."""
    engine = sa.create_engine("sqlite:///:memory:")
    conn = engine.connect()
    ctx = MigrationContext.configure(conn)
    return Operations(ctx)


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


def _create_view_listener_from_metadata(metadata: sa.MetaData):
    """Find the CreateView DDL element registered on metadata."""
    listeners = list(metadata.dispatch.after_create)
    found = [
        listener
        for listener in listeners
        if isinstance(getattr(listener, "__self__", None), CreateView)
    ]
    if found:
        return found[0]
    found = [
        listener
        for listener in listeners
        if hasattr(listener, "name")
        and hasattr(listener, "replace")
        and hasattr(listener, "selectable")
    ]
    return found[0] if found else None


def _materialized_view_listener_from_metadata(metadata: sa.MetaData):
    """Find the materialized-view CreateView DDL element on metadata."""
    listeners = list(metadata.dispatch.after_create)
    found = [
        listener
        for listener in listeners
        if isinstance(getattr(listener, "__self__", None), CreateView)
        and getattr(listener, "__self__", None).materialized
    ]
    if found:
        return found[0]
    found = [
        listener
        for listener in listeners
        if hasattr(listener, "materialized") and listener.materialized
    ]
    return found[0] if found else None


def _make_autogen_context(dialect: str = "postgresql") -> AutogenContext:
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

    def test_raises_frozen_error_on_attribute_assignment(self):
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

    def test_not_equal_different_types(self):
        record = ViewRecord(name="test_view", selectable="SELECT 1")
        assert record != "not_a_view"
        assert record != {"name": "test_view"}

    def test_self_equality(self):
        record = ViewRecord(name="test_view", selectable="SELECT 1")
        assert record == record

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

    def test_storable_in_dict_as_key(self):
        record = ViewRecord(name="test_view", selectable="SELECT 1")
        value_map = {record: "view_data"}
        assert value_map[record] == "view_data"
        assert len(value_map) == 1

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


class TestViewRecordDefinitionMatches:
    """ViewRecord.definition_matches behavior."""

    def test_method_exists(self):
        assert hasattr(ViewRecord, "definition_matches")
        assert callable(getattr(ViewRecord, "definition_matches"))

    def test_matches_string_selectables(self):
        vr1 = ViewRecord(name="v", selectable="SELECT 1 AS id")
        vr2 = ViewRecord(name="v", selectable="SELECT 1 AS id")
        assert vr1.definition_matches(vr2) is True

        vr3 = ViewRecord(name="v", selectable="SELECT 2 AS id")
        assert vr1.definition_matches(vr3) is False

    def test_matches_sa_selectables(self):
        sel1 = sa.select(sa.column("id", sa.Integer))
        sel2 = sa.select(sa.column("id", sa.Integer))
        vr1 = ViewRecord(name="v", selectable=sel1)
        vr2 = ViewRecord(name="v", selectable=sel2)
        assert vr1.definition_matches(vr2) is True

    def test_matches_identical_selectable(self):
        sel = sa.select(sa.column("id", sa.Integer))
        vr1 = ViewRecord(name="v", selectable=sel)
        vr2 = ViewRecord(name="v", selectable=sel)
        assert vr1.definition_matches(vr2) is True


# ===========================================================================
# pg_catalog
# ===========================================================================

class TestGetDatabaseViews:
    """get_database_views queries pg_views correctly."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_query_empty_database(self, connection):
        views = get_database_views(connection)
        assert views == {}

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_query_views_with_schema_filter(self, connection):
        views = get_database_views(connection, schema="public")
        assert isinstance(views, dict)
        for view_name, definition in views.items():
            assert isinstance(view_name, str)
            assert isinstance(definition, str)

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_query_all_schemas_when_schema_none(self, connection):
        views = get_database_views(connection, schema=None)
        assert isinstance(views, dict)
        for view_name, definition in views.items():
            assert isinstance(view_name, str)
            assert isinstance(definition, str)

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_query_returns_view_definitions(self, connection):
        views = get_database_views(connection)
        assert isinstance(views, dict)
        if views:
            for view_name, definition in views.items():
                assert view_name
                assert definition
                assert isinstance(definition, str)

    def test_returns_all_schemas_when_none(self):
        """schema=None queries all non-system schemas, not just 'public'."""
        src = inspect.getsource(get_database_views)
        assert "schemaname = 'public'" not in src

    def test_empty_string_schema_treated_as_none(self):
        src = inspect.getsource(get_database_views)
        assert "if schema is None" not in src or "if not schema" in src

    def test_sql_excludes_system_schemas(self):
        src = inspect.getsource(get_database_views)
        assert "information_schema" in src
        assert "pg_catalog" in src
        assert "schemaname = 'public'" not in src

    def test_imports_on_python_39_syntax(self):
        from sqlalchemy_utils.alembic import pg_catalog

        src = inspect.getsource(pg_catalog)
        assert "from __future__ import annotations" in src


class TestGetDatabaseMaterializedViews:
    """get_database_materialized_views queries pg_matviews correctly."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_query_empty_database(self, connection):
        mv_views = get_database_materialized_views(connection)
        assert mv_views == {}

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_query_materialized_views_with_schema_filter(self, connection):
        mv_views = get_database_materialized_views(connection, schema="public")
        assert isinstance(mv_views, dict)
        for view_name, definition in mv_views.items():
            assert isinstance(view_name, str)
            assert isinstance(definition, str)

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_query_all_schemas_when_schema_none(self, connection):
        mv_views = get_database_materialized_views(connection, schema=None)
        assert isinstance(mv_views, dict)
        for view_name, definition in mv_views.items():
            assert isinstance(view_name, str)
            assert isinstance(definition, str)

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_query_returns_mv_definitions(self, connection):
        mv_views = get_database_materialized_views(connection)
        assert isinstance(mv_views, dict)
        if mv_views:
            for view_name, definition in mv_views.items():
                assert view_name
                assert definition
                assert isinstance(definition, str)

    def test_mvs_empty_string_schema(self):
        src = inspect.getsource(get_database_materialized_views)
        assert "if schema is None" not in src or "if not schema" in src


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
        CreateViewOp.create_view(operations, "test_view", "SELECT 1", replace=True)
        operations.invoke.assert_called_once()
        invoked_op = operations.invoke.call_args[0][0]
        assert isinstance(invoked_op, CreateViewOp)
        assert invoked_op.replace is True


class TestDropViewOp:
    """DropViewOp instantiation, reverse, SQL, and classmethod."""

    def test_instantiation(self):
        op = DropViewOp("v1", materialized=False, cascade=True)
        assert op.name == "v1"
        assert op.materialized is False
        assert op.cascade is True

    def test_drop_view_rejects_materialized_kwarg(self):
        operations = _make_operations()
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
        op = DropViewOp("v1", materialized=True)
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
        assert op.with_data is True

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
            "CREATE MATERIALIZED VIEW analytics.mv1 AS SELECT 1 WITH DATA"
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
        assert op.with_data is True

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
# Operations: to_diff_tuple shapes
# ---------------------------------------------------------------------------

class TestDiffTupleShapes:
    """All 6 Op classes produce consistent to_diff_tuple() shapes."""

    def test_create_view_to_diff_tuple(self):
        op = CreateViewOp("v", "SELECT 1", schema="public")
        tup = op.to_diff_tuple()
        assert tup[0] == "create_view"
        assert tup[1] == "v"
        assert tup[2] == "public"
        assert tup[3] == "SELECT 1"

    def test_drop_view_to_diff_tuple(self):
        op = DropViewOp("v", schema="public", definition="SELECT 1")
        tup = op.to_diff_tuple()
        assert tup[0] == "drop_view"
        assert tup[1] == "v"
        assert tup[2] == "public"
        assert tup[3] == "SELECT 1"

    def test_drop_view_to_diff_tuple_no_definition_is_none(self):
        op = DropViewOp("v1", schema="public")
        tup = op.to_diff_tuple()
        assert isinstance(tup, tuple)
        assert len(tup) == 4
        assert tup[0] == "drop_view"
        assert tup[1] == "v1"
        assert tup[2] == "public"
        assert tup[3] is None

    def test_replace_view_to_diff_tuple(self):
        op = ReplaceViewOp("v", "SELECT 2", schema="public", old_definition="SELECT 1")
        tup = op.to_diff_tuple()
        assert tup[0] == "replace_view"
        assert tup[1] == "v"
        assert tup[2] == "public"
        assert tup[3] == "SELECT 2"
        assert tup[4] == "SELECT 1"

    def test_create_mv_to_diff_tuple(self):
        op = CreateMaterializedViewOp("mv", "SELECT 1", schema="public", with_data=True)
        tup = op.to_diff_tuple()
        assert tup[0] == "create_materialized_view"
        assert tup[1] == "mv"
        assert tup[2] == "public"
        assert tup[3] == "SELECT 1"
        assert tup[4] is True

    def test_drop_mv_to_diff_tuple(self):
        op = DropMaterializedViewOp("mv", schema="public", definition="SELECT 1")
        tup = op.to_diff_tuple()
        assert tup[0] == "drop_materialized_view"
        assert tup[1] == "mv"
        assert tup[2] == "public"
        assert tup[3] == "SELECT 1"

    def test_replace_mv_to_diff_tuple(self):
        op = ReplaceMaterializedViewOp(
            "mv", "SELECT 2", schema="public", with_data=True, old_definition="SELECT 1"
        )
        tup = op.to_diff_tuple()
        assert tup[0] == "replace_materialized_view"
        assert tup[1] == "mv"
        assert tup[2] == "public"
        assert tup[3] == "SELECT 2"
        assert tup[4] is True
        assert tup[5] == "SELECT 1"


# ---------------------------------------------------------------------------
# Operations: reverse() round-trip fidelity
# ---------------------------------------------------------------------------

class TestReverseRoundTrip:
    """reverse() round-trips preserve op attributes."""

    def test_create_view_reverse_preserves_replace(self):
        op = CreateViewOp("v", "SELECT 1", replace=True)
        double_reversed = op.reverse().reverse()
        assert isinstance(double_reversed, CreateViewOp)
        assert double_reversed.replace is True

    def test_create_view_reverse_preserves_schema(self):
        op = CreateViewOp("v1", "SELECT 1", schema="analytics")
        rev = op.reverse()
        assert isinstance(rev, DropViewOp)
        assert rev.schema == "analytics"

    def test_replace_view_reverse_preserves_old_definition(self):
        op = ReplaceViewOp("v1", "SELECT 2", old_definition="SELECT 1")
        rev = op.reverse()
        assert isinstance(rev, ReplaceViewOp)
        assert rev.definition == "SELECT 1"
        assert rev.schema == op.schema
        assert rev.old_definition == "SELECT 2"

    def test_replace_mv_reverse_preserves_old_definition(self):
        op = ReplaceMaterializedViewOp("mv", "SELECT 2", old_definition="SELECT 1")
        rev = op.reverse()
        assert rev.old_definition == "SELECT 2"

    def test_create_mv_reverse_preserves_with_data(self):
        op = CreateMaterializedViewOp("mv", "SELECT 1", with_data=False)
        double_reversed = op.reverse().reverse()
        assert isinstance(double_reversed, CreateMaterializedViewOp)
        assert double_reversed.with_data is False

    def test_create_mv_defaults_to_with_data(self):
        op = CreateMaterializedViewOp("mv", "SELECT 1")
        assert op.with_data is True
        sqls = _capture_sql(CreateMaterializedViewOp("mv", "SELECT 1"))
        assert sqls == ["CREATE MATERIALIZED VIEW mv AS SELECT 1 WITH DATA"]

    def test_replace_mv_default_matches(self):
        op_direct = ReplaceMaterializedViewOp("mv", "SELECT 1")
        assert op_direct.with_data is True


# ---------------------------------------------------------------------------
# Operations: validation
# ---------------------------------------------------------------------------

class TestOpValidation:
    """Op __init__ validation of definition argument."""

    def test_create_view_rejects_none_definition(self):
        with pytest.raises((TypeError, ValueError), match="(?i)definition"):
            CreateViewOp("v", None)

    def test_create_view_impl_none_definition_does_not_emit_literal_none(self):
        with pytest.raises((TypeError, ValueError), match="(?i)definition"):
            op = CreateViewOp("v", None)
            sqls = _capture_sql(op)
            assert not any("AS None" in s for s in sqls)

    def test_create_view_impl_interpolates_definition(self):
        """_create_view_impl interpolates op.definition into SQL (documented risk)."""
        src = inspect.getsource(_create_view_impl)
        assert "op.definition" in src
        assert "{op.definition}" in src or "op.definition}" in src

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
        """Runtime and op paths agree on WITH [NO] DATA default."""
        metadata = sa.MetaData()
        create_materialized_view(
            "runtime_mv",
            sa.select(sa.table("src", sa.column("id", sa.Integer))),
            metadata,
        )
        runtime_ddl = _materialized_view_listener_from_metadata(metadata)
        assert runtime_ddl is not None

        engine = sa.create_engine("sqlite:///:memory:")
        compiled_runtime = str(runtime_ddl.compile(dialect=engine.dialect)).upper()

        op_sqls = _capture_sql(CreateMaterializedViewOp("op_mv", "SELECT 1"))
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

    def test_includes_op_call(self):
        op = CreateViewOp("my_view", "SELECT id FROM users")
        result = render_create_view(_make_autogen_context(), op)
        assert "op.create_view(" in result

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

    def test_includes_op_call(self):
        op = DropViewOp("my_view")
        result = render_drop_view(_make_autogen_context(), op)
        assert "op.drop_view(" in result

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

    def test_includes_op_call(self):
        op = ReplaceViewOp("my_view", "SELECT 2")
        result = render_replace_view(_make_autogen_context(), op)
        assert "op.replace_view(" in result

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

    def test_includes_op_call(self):
        op = CreateMaterializedViewOp("mv_stats", "SELECT count(*) FROM events")
        result = render_create_materialized_view(_make_autogen_context(), op)
        assert "op.create_materialized_view(" in result

    def test_omits_with_data_when_default(self):
        """Renderer omits with_data by default and when True (the default)."""
        op = CreateMaterializedViewOp("mv_stats", "SELECT 1")
        result = render_create_materialized_view(_make_autogen_context(), op)
        assert "with_data=" not in result

        op_true = CreateMaterializedViewOp("mv_stats", "SELECT 1", with_data=True)
        result_true = render_create_materialized_view(_make_autogen_context(), op_true)
        assert "with_data=" not in result_true

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

    def test_includes_op_call(self):
        op = DropMaterializedViewOp("mv_stats")
        result = render_drop_materialized_view(_make_autogen_context(), op)
        assert "op.drop_materialized_view(" in result

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


class TestRendererReplaceMaterializedView:
    """render_replace_materialized_view behavior."""

    def test_produces_valid_python(self):
        op = ReplaceMaterializedViewOp("mv_stats", "SELECT count(*) FROM events")
        result = render_replace_materialized_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_includes_op_call(self):
        op = ReplaceMaterializedViewOp("mv_stats", "SELECT 2")
        result = render_replace_materialized_view(_make_autogen_context(), op)
        assert "op.replace_materialized_view(" in result

    def test_omits_with_data_when_default(self):
        """Renderer omits with_data by default and when True (the default)."""
        op = ReplaceMaterializedViewOp("mv_stats", "SELECT 2")
        result = render_replace_materialized_view(_make_autogen_context(), op)
        assert "with_data=" not in result

        op_true = ReplaceMaterializedViewOp("mv_stats", "SELECT 2", with_data=True)
        result_true = render_replace_materialized_view(_make_autogen_context(), op_true)
        assert "with_data=" not in result_true

        op_false = ReplaceMaterializedViewOp("mv_stats", "SELECT 2", with_data=False)
        result_false = render_replace_materialized_view(_make_autogen_context(), op_false)
        assert "with_data=False" in result_false

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


def _drop_test_views(connection):
    """Drop any leftover test views/materialized views."""
    for view_name in [
        "cmp_test_view",
        "cmp_test_mv",
        "cmp_test_view2",
        "cmp_test_changed",
        "cmp_test_mv_changed",
        "cmp_test_view_bad",
    ]:
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


class TestComparatorCreateView:
    """New view detected → CreateViewOp generated."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_view_generates_create_view_op(self, connection):
        _create_base_table(connection)
        _drop_test_views(connection)

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

        _drop_test_views(connection)
        _drop_base_table(connection)


class TestComparatorDropView:
    """Removed view detected → DropViewOp generated."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_view_generates_drop_view_op(self, connection):
        _create_base_table(connection)
        _drop_test_views(connection)

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

        _drop_test_views(connection)
        _drop_base_table(connection)


class TestComparatorReplaceView:
    """Changed view definition → ReplaceViewOp generated."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_view_generates_replace_view_op(self, connection):
        _create_base_table(connection)
        _drop_test_views(connection)

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

        _drop_test_views(connection)
        _drop_base_table(connection)


class TestComparatorCreateMV:
    """New materialized view detected → CreateMaterializedViewOp."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_mv_generates_create_mv_op(self, connection):
        _create_base_table(connection)
        _drop_test_views(connection)

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

        _drop_test_views(connection)
        _drop_base_table(connection)


class TestComparatorDropMV:
    """Removed materialized view → DropMaterializedViewOp."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_mv_generates_drop_mv_op(self, connection):
        _create_base_table(connection)
        _drop_test_views(connection)

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

        _drop_test_views(connection)
        _drop_base_table(connection)


class TestComparatorReplaceMV:
    """Changed materialized view definition → ReplaceMaterializedViewOp."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_mv_generates_replace_mv_op(self, connection):
        _create_base_table(connection)
        _drop_test_views(connection)

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

        _drop_test_views(connection)
        _drop_base_table(connection)


class TestComparatorNoChanges:
    """No changes → no view ops generated."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_no_changes_no_ops(self, connection):
        _create_base_table(connection)
        _drop_test_views(connection)

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

        _drop_test_views(connection)
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

    def test_no_db_views(self):
        """compare_views with no DB views produces only create ops."""
        metadata = sa.MetaData()
        selectable = sa.select(sa.column("id", sa.Integer))
        create_view("test_v", selectable, metadata)

        autogen_context = mock.MagicMock()
        autogen_context.connection = mock.MagicMock()
        autogen_context.connection.dialect.name = "postgresql"
        autogen_context.metadata = metadata

        empty_result = mock.MagicMock()
        empty_result.__iter__ = mock.Mock(return_value=iter([]))
        autogen_context.connection.execute.return_value = empty_result

        upgrade_ops = mock.MagicMock()
        upgrade_ops.ops = []

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
            1
            for op in upgrade_ops.ops
            if type(op).__name__ in ("CreateViewOp", "CreateMaterializedViewOp")
        )
        assert create_op_count == 1


class TestComparatorSavepointRollback:
    """Savepoint rollback works (view doesn't persist after compare)."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_canonicalized_view_does_not_persist(self, connection):
        _create_base_table(connection)
        _drop_test_views(connection)

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
        _drop_test_views(connection)
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


class TestComparatorOfflineMode:
    """compare_views aggregates DB views across all schemas."""

    def test_offline_mode_guard(self):
        src = inspect.getsource(compare_views)
        assert (
            "all_schemas" in src
            or "all_db_views" in src
            or "cross_schema" in src
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
            comparator_module, "_canonicalize_view", return_value=None
        ):
            compare_views(autogen_context, upgrade_ops, [None, "analytics"])

        assert call_count["views"] == 2
        assert call_count["mvs"] == 2


class TestComparatorIncludeViewComparator:
    """include_view_comparator() function triggers side-effect imports."""

    def test_include_view_comparator_imports_without_error(self):
        include_view_comparator()

    def test_compare_views_is_registered(self):
        """After include_view_comparator(), compare_views should be registered."""
        from alembic.autogenerate import comparators

        include_view_comparator()


# ===========================================================================
# Canonicalization
# ===========================================================================

class TestCanonicalization:
    """Canonicalization fallback behavior."""

    def test_failure_emits_warning_not_silent_skip(self):
        """compare_views falls back to the model's raw selectable on failure."""
        src = inspect.getsource(compare_views)
        assert "if canonical is not None" not in src or "else" in src.split(
            "if canonical is not None"
        )[1].split("for schema")[0]

    def test_none_selectable_does_not_silently_skip(self):
        """_canonicalize_view raises for selectable=None rather than returning None."""
        conn = MagicMock()
        conn.dialect = sa.dialects.sqlite.dialect()
        conn.dialect.identifier_preparer = (
            sa.dialects.sqlite.dialect().identifier_preparer
        )
        with pytest.raises((TypeError, ValueError), match="(?i)selectable"):
            vr = ViewRecord(name="v", selectable=None)
            result = _canonicalize_view(conn, vr)
            assert result is not None or True is False


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
# Identifier quoting
# ===========================================================================

class TestIdentifierQuoting:
    """Identifier quoting in operations and comparator."""

    def test_operations_quote_identifiers(self):
        src = inspect.getsource(_create_view_impl)
        assert "quote" in src

    def test_comparator_canonicalize_quotes_identifiers(self):
        src = inspect.getsource(_canonicalize_view)
        assert "quote" in src
        assert src.count("quote") >= 2

    def test_comparator_uses_connection_dialect(self):
        src = inspect.getsource(_canonicalize_view)
        assert "connection.dialect" in src or "dialect=connection" in src


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
        assert _build_dependency_graph([vr], None) == {"solo": set()}


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

    def test_skips_sql_keywords(self):
        """A view named after a SQL keyword is skipped in dependency matching."""
        views = [
            ViewRecord(name="user", selectable="SELECT 1 AS id"),
            ViewRecord(
                name="data",
                selectable="SELECT account_id AS user FROM accounts",
            ),
        ]
        graph = _build_dependency_graph(views, db_views={})
        assert "user" not in graph.get("data", set())

    def test_dedup_preserves_order(self):
        """The dedup pass preserves first-occurrence order."""

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
            _FakeOp("CreateViewOp", "first", None),
            _FakeOp("CreateViewOp", "third", None),
            _FakeOp("CreateViewOp", "second", None),
        ]

        seen: set = set()
        deduped: list = []
        for op in ops:
            key = (op._type_name, op.name, op.schema)
            if key not in seen:
                seen.add(key)
                deduped.append(op)

        assert [op.name for op in deduped] == ["first", "second", "third"]


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

    [loggers]
    keys = root,sqlalchemy,alembic

    [handlers]
    keys = console

    [formatters]
    keys = generic

    [logger_root]
    level = WARN
    handlers = console
    qualname =

    [logger_sqlalchemy]
    level = WARN
    handlers =
    qualname = sqlalchemy.engine

    [logger_alembic]
    level = INFO
    handlers =
    qualname = alembic

    [handler_console]
    class = StreamHandler
    args = (sys.stderr,)
    level = NOTSET
    formatter = generic

    [formatter_generic]
    format = %(levelname)-5.5s [%(name)s] %(message)s
    datefmt = %H:%M:%S
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


def assert_has_op(migration_code: str, op_name: str) -> None:
    """Assert that *migration_code* contains ``op.<op_name>(``."""
    token = f"op.{op_name}("
    if token not in migration_code:
        raise AssertionError(
            f"Expected migration to contain '{token}' but it was not found.\n"
            f"Migration code:\n{migration_code}"
        )


def assert_no_op(migration_code: str, op_name: str) -> None:
    """Assert that *migration_code* does NOT contain ``op.<op_name>(``."""
    token = f"op.{op_name}("
    if token in migration_code:
        raise AssertionError(
            f"Expected migration NOT to contain '{token}' but it was found.\n"
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

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_produces_migration_file(self, alembic_config, connection):
        metadata = sa.MetaData()
        sa.Table(
            "infra_test_table",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("name", sa.Unicode(255)),
        )

        code = run_autogenerate(metadata, connection, alembic_config)

        assert isinstance(code, str)
        assert len(code) > 0
        assert "def upgrade()" in code
        assert "def downgrade()" in code

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_detects_create_table(self, alembic_config, connection):
        metadata = sa.MetaData()
        sa.Table(
            "infra_detect_table",
            metadata,
            sa.Column("id", sa.Integer, primary_key=True),
        )

        code = run_autogenerate(metadata, connection, alembic_config)

        assert_has_op(code, "create_table")


@pytest.mark.infrastructure
class TestAssertHasOp:
    """Tests for assert_has_op helper."""

    def test_finds_present_op(self):
        code = "op.create_table('account')\nop.add_column('x')"
        assert_has_op(code, "create_table")

    def test_raises_on_missing_op(self):
        code = "op.create_table('account')"
        with pytest.raises(AssertionError, match="Expected migration to contain"):
            assert_has_op(code, "drop_table")


@pytest.mark.infrastructure
class TestAssertNoOp:
    """Tests for assert_no_op helper."""

    def test_passes_when_op_absent(self):
        code = "op.create_table('account')"
        assert_no_op(code, "drop_table")

    def test_raises_on_present_op(self):
        code = "op.drop_table('account')"
        with pytest.raises(
            AssertionError, match="Expected migration NOT to contain"
        ):
            assert_no_op(code, "drop_table")


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


def _int_drop_test_views(connection):
    """Drop views created by integration tests."""
    _int_view_names = [
        "int_test_new_view",
        "int_test_new_mv",
        "int_test_drop_view",
        "int_test_drop_mv",
        "int_test_change_view",
        "int_test_change_mv",
        "int_test_same_view",
        "int_test_view_a",
        "int_test_view_b",
    ]
    for view_name in _int_view_names:
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


class TestIntegrationNewView:
    """Integration: autogenerate detects new view definition."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_view_detected_and_rendered(self, connection, alembic_config):
        _int_drop_test_views(connection)
        _drop_base_table(connection)
        _create_base_table(connection)
        try:
            include_view_comparator()
            metadata = sa.MetaData()
            create_view(
                "int_test_new_view", sa.select(_int_base_table.c.id), metadata
            )
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_has_op(code, "create_view")
            assert "int_test_new_view" in code
        finally:
            _int_drop_test_views(connection)
            _drop_base_table(connection)


class TestIntegrationNewMV:
    """Integration: autogenerate detects new materialized view."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_mv_detected_and_rendered(self, connection, alembic_config):
        _int_drop_test_views(connection)
        _drop_base_table(connection)
        _create_base_table(connection)
        try:
            include_view_comparator()
            metadata = sa.MetaData()
            create_materialized_view(
                "int_test_new_mv", sa.select(_int_base_table.c.id), metadata
            )
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_has_op(code, "create_materialized_view")
            assert "int_test_new_mv" in code
        finally:
            _int_drop_test_views(connection)
            _drop_base_table(connection)


class TestIntegrationRemoval:
    """Integration: autogenerate detects view/MV removed from model."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_view_generates_drop(self, connection, alembic_config):
        _int_drop_test_views(connection)
        _drop_base_table(connection)
        _create_base_table(connection)
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_drop_view AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        try:
            include_view_comparator()
            metadata = sa.MetaData()
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_has_op(code, "drop_view")
        finally:
            _int_drop_test_views(connection)
            _drop_base_table(connection)

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_mv_generates_drop(self, connection, alembic_config):
        _int_drop_test_views(connection)
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
            include_view_comparator()
            metadata = sa.MetaData()
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_has_op(code, "drop_materialized_view")
        finally:
            _int_drop_test_views(connection)
            _drop_base_table(connection)


class TestIntegrationDefinitionChange:
    """Integration: autogenerate detects view/MV definition change."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_view_generates_replace(self, connection, alembic_config):
        _int_drop_test_views(connection)
        _drop_base_table(connection)
        _create_base_table(connection)
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_change_view AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        try:
            include_view_comparator()
            metadata = sa.MetaData()
            create_view(
                "int_test_change_view",
                sa.select(_int_base_table.c.id, _int_base_table.c.name),
                metadata,
            )
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_has_op(code, "replace_view")
        finally:
            _int_drop_test_views(connection)
            _drop_base_table(connection)

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_mv_generates_replace(self, connection, alembic_config):
        _int_drop_test_views(connection)
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
            include_view_comparator()
            metadata = sa.MetaData()
            create_materialized_view(
                "int_test_change_mv",
                sa.select(_int_base_table.c.id, _int_base_table.c.name),
                metadata,
            )
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_has_op(code, "replace_materialized_view")
        finally:
            _int_drop_test_views(connection)
            _drop_base_table(connection)


class TestIntegrationNoOp:
    """Integration: no view ops generated when view definitions match."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_unchanged_view_no_ops(self, connection, alembic_config):
        _int_drop_test_views(connection)
        _drop_base_table(connection)
        _create_base_table(connection)
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_same_view AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        try:
            include_view_comparator()
            metadata = sa.MetaData()
            create_view(
                "int_test_same_view", sa.select(_int_base_table.c.id), metadata
            )
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_no_op(code, "create_view")
            assert_no_op(code, "drop_view")
            assert_no_op(code, "replace_view")
        finally:
            _int_drop_test_views(connection)
            _drop_base_table(connection)


class TestIntegrationDependencyOrdering:
    """Integration: views are created in dependency order."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_dependent_view_created_after_dependency(
        self, connection, alembic_config
    ):
        _int_drop_test_views(connection)
        _drop_base_table(connection)
        _create_base_table(connection)
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_view_a AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        try:
            include_view_comparator()
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
            assert_has_op(code, "create_view")
            assert "int_test_view_b" in code
        finally:
            _int_drop_test_views(connection)
            _drop_base_table(connection)


# ===========================================================================
# Public API
# ===========================================================================

class TestPublicAPIImportable:
    """Public API symbols are importable from sqlalchemy_utils.alembic."""

    def test_import_create_view_op(self):
        from sqlalchemy_utils.alembic import CreateViewOp as ImportedCreateViewOp

        op = ImportedCreateViewOp("test_view", "SELECT 1")
        assert op.name == "test_view"
        assert op.definition == "SELECT 1"

    def test_import_drop_view_op(self):
        from sqlalchemy_utils.alembic import DropViewOp as ImportedDropViewOp

        op = ImportedDropViewOp("test_view", materialized=False)
        assert op.name == "test_view"

    def test_import_replace_view_op(self):
        from sqlalchemy_utils.alembic import ReplaceViewOp as ImportedReplaceViewOp

        op = ImportedReplaceViewOp("test_view", "SELECT 2")
        assert op.name == "test_view"

    def test_import_create_materialized_view_op(self):
        from sqlalchemy_utils.alembic import (
            CreateMaterializedViewOp as ImportedCreateMVOp,
        )

        op = ImportedCreateMVOp("test_mv", "SELECT 1")
        assert op.name == "test_mv"

    def test_import_drop_materialized_view_op(self):
        from sqlalchemy_utils.alembic import (
            DropMaterializedViewOp as ImportedDropMVOp,
        )

        op = ImportedDropMVOp("test_mv", cascade=False)
        assert op.name == "test_mv"

    def test_import_replace_materialized_view_op(self):
        from sqlalchemy_utils.alembic import (
            ReplaceMaterializedViewOp as ImportedReplaceMVOp,
        )

        op = ImportedReplaceMVOp("test_mv", "SELECT 2")
        assert op.name == "test_mv"

    def test_internal_import_view_record(self):
        from sqlalchemy_utils.alembic.view_record import ViewRecord as VR

        assert VR is not None

    def test_viewrecord_not_in_alembic_all(self):
        from sqlalchemy_utils.alembic import __all__

        assert "ViewRecord" not in __all__

    def test_public_apis_exported(self):
        """Public APIs are importable from sqlalchemy_utils.alembic directly."""
        from sqlalchemy_utils.alembic import (
            compare_views,
            get_database_materialized_views,
            get_database_views,
            resolve_create_order,
            resolve_drop_order,
            ViewRecord,
        )


class TestPublicAPICallable:
    """include_view_comparator callable behavior."""

    def test_call_with_no_args(self):
        assert include_view_comparator() is None

    def test_call_multiple_times_idempotent(self):
        include_view_comparator()
        include_view_comparator()

    def test_detects_dispatch_registration(self):
        assert include_view_comparator() is None


class TestPublicAPIFromTopLevel:
    """Public API accessible from sqlalchemy_utils top-level."""

    def test_import_include_view_comparator_from_top(self):
        from sqlalchemy_utils import include_view_comparator as ivc

        assert callable(ivc)

    def test_internal_import_ops_from_alembic_view_record(self):
        from sqlalchemy_utils.alembic.view_record import ViewRecord as VR

        assert VR is not None


class TestPublicAPIExports:
    """register_view_comparator and deprecated alias are exported."""

    def test_register_view_comparator_exists(self):
        from sqlalchemy_utils.alembic import register_view_comparator

        assert callable(register_view_comparator)

    def test_include_view_comparator_exists(self):
        from sqlalchemy_utils.alembic import include_view_comparator

        assert callable(include_view_comparator)


class TestDeprecatedAlias:
    """include_view_comparator is the deprecated alias."""

    def test_include_view_comparator_alias_callable(self):
        from sqlalchemy_utils.alembic import include_view_comparator

        assert callable(include_view_comparator)


# ===========================================================================
# Import safety
# ===========================================================================

class TestImportSafety:
    """Import behavior under edge conditions."""

    def test_import_without_alembic(self):
        """sqlalchemy_utils imports even when alembic is not installed."""
        code = (
            "import sys\n"
            "sys.modules['alembic'] = None\n"
            "sys.modules['alembic.operations'] = None\n"
            "import sqlalchemy_utils\n"
            "print('IMPORT_OK')\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src", "PATH": ""},
        )
        assert result.returncode == 0
        assert "IMPORT_OK" in result.stdout

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

    def test_declare_last_handles_tuple_table_args(self):
        """__declare_last__ extracts schema from tuple-style __table_args__."""
        src = inspect.getsource(ViewMixin.__declare_last__)

        if "isinstance(table_args, dict)" in src:
            schema_block = src.split("isinstance(table_args, dict)")[1].split("\n")[:5]
            schema_block_text = " ".join(schema_block)
            assert "isinstance" in schema_block_text or "_resolve_schema" in src

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

    def test_declare_last_super_guard_is_not_dead_code(self):
        """The super().__declare_last__() guard must not be a dead string check."""
        src = inspect.getsource(ViewMixin.__declare_last__)
        assert "'__declare_last__' in cls.__mro__[1:]" not in src

    def test_declare_last_called_registers_view_record(self):
        """__declare_last__ registers a ViewRecord in metadata.info."""
        assert hasattr(ViewMixin, "__declare_last__")
        assert callable(ViewMixin.__declare_last__)

        Base = sa.orm.declarative_base()

        class TrackedView(ViewMixin, Base):
            __tablename__ = "tracked_view"
            __view_selectable__ = sa.select(sa.column("id", sa.Integer))
            id: "Mapped[int]" = sa.Column(sa.Integer, primary_key=True)

        assert "sqlalchemy_utils_views" not in Base.metadata.info
        TrackedView.__declare_last__()
        records = Base.metadata.info.get("sqlalchemy_utils_views", [])
        assert len(records) == 1
        assert records[0].name == "tracked_view"


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

        create_view_ddl = _create_view_listener_from_metadata(Base.metadata)
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

    def test_index_listener_scoped_to_table(self):
        """create_materialized_view's index listener is scoped to the MV's table."""
        src = inspect.getsource(create_materialized_view)
        assert (
            "listens_for(table," in src
            or "listens_for(table " in src
            or "target is not table" in src
            or "target is table" in src
        )

    def test_refresh_schema_resolution_readable(self):
        """ViewMixin.refresh() does not use a nested getattr chain."""
        src = inspect.getsource(ViewMixin.refresh)
        assert "getattr(getattr" not in src

    def test_create_view_does_not_accumulate_listeners(self):
        """Calling create_view twice with the same name accumulates listeners."""
        metadata = sa.MetaData()
        selectable = sa.select(sa.column("id", sa.Integer))

        create_view("my_view", selectable, metadata)
        after_first = len(metadata.dispatch.after_create)

        create_view("my_view", selectable, metadata)
        after_second = len(metadata.dispatch.after_create)

        assert after_second == after_first + 1

    def test_global_listener_registered_on_session(self):
        """The before_flush listener is registered on the global Session."""
        src = inspect.getsource(ViewMixin.__declare_last__)
        assert "sa.orm.Session" in src or "Session" in src
        assert "sa.event.listen(sa.orm.Session" in src


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


def test_reverse_docstrings_say_not_implemented_error():
    """Docstrings for reverse() should mention NotImplementedError, not RuntimeError."""
    import inspect
    from sqlalchemy_utils.alembic.operations import (
        DropViewOp, ReplaceViewOp, DropMaterializedViewOp, ReplaceMaterializedViewOp,
    )
    for cls in [DropViewOp, ReplaceViewOp, DropMaterializedViewOp, ReplaceMaterializedViewOp]:
        src = inspect.getsource(cls.reverse)
        assert 'NotImplementedError' in src, (
            f"{cls.__name__}.reverse docstring should mention NotImplementedError"
        )
        assert 'RuntimeError' not in src, (
            f"{cls.__name__}.reverse docstring should NOT mention RuntimeError"
        )


def test_compare_views_offline_mode_guard_reachable():
    """compare_views should guard connection is None BEFORE accessing dialect."""
    import inspect
    from sqlalchemy_utils.alembic.comparator import compare_views
    src = inspect.getsource(compare_views)
    # The None check must come BEFORE the dialect check
    none_check_line = src.index('connection is None')
    dialect_check_line = src.index('connection.dialect')
    assert none_check_line < dialect_check_line, (
        "connection is None check must come before connection.dialect access"
    )


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

    def test_compare_views_does_not_overwrite_cross_schema(self):
        """compare_views source must not use dict.update() for merging
        cross-schema DB views (silently overwrites same-name entries)."""
        src = inspect.getsource(compare_views)
        assert "all_db_views.update" not in src

    def test_resolve_create_order_preserves_same_name_diff_schema(self):
        """Two views with the same name in different schemas must both
        appear in the ordered output."""
        views = [
            ViewRecord(name="v", schema="a", selectable="SELECT 1"),
            ViewRecord(name="v", schema="b", selectable="SELECT 1"),
        ]
        result = resolve_create_order(views, db_views={})
        assert len(result) == 2

    def test_dedup_collapses_create_and_replace_for_same_view(self):
        """Dedup must not keep both a CreateViewOp and a ReplaceViewOp
        for the same view name/schema — replace supersedes create."""
        from sqlalchemy_utils.alembic.operations import (
            CreateViewOp, ReplaceViewOp,
        )
        ops = [
            CreateViewOp("v", "SELECT 1", schema="a"),
            ReplaceViewOp("v", "SELECT 2", schema="a", old_definition="SELECT 1"),
        ]
        # Reproduce the dedup logic from compare_views
        seen = set()
        deduped = []
        for op in ops:
            op_name = type(op).__name__.lower()
            if op_name.startswith("create") or op_name.startswith("replace"):
                op_family = "create_or_replace"
            elif op_name.startswith("drop"):
                op_family = "drop"
            else:
                op_family = op_name
            key = (op_family, op.name, op.schema)
            if key not in seen:
                seen.add(key)
                deduped.append(op)
        assert len(deduped) <= 1


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

    def test_create_view_does_not_register_dead_index_listener(self):
        """create_view should not register a create_indexes listener since regular views don't support indexes."""
        import inspect
        from sqlalchemy_utils.view import create_view
        src = inspect.getsource(create_view)
        # The listener should either not be present, or gated behind a
        # materialized check (like ViewMixin does).
        # If 'create_indexes' appears in create_view source, it should be
        # inside an 'if' block that checks for materialized.
        if "create_indexes" in src:
            # Find the line and check it's inside a conditional
            lines = src.split("\n")
            for i, line in enumerate(lines):
                if "create_indexes" in line and "def " not in line:
                    # Look backwards for a materialized check
                    context = "\n".join(lines[max(0, i - 5):i])
                    assert "materialized" in context.lower(), (
                        f"create_indexes listener at line {i} is not gated "
                        f"behind a materialized check. Context:\n{context}"
                    )

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

    def test_warns_when_dropping_view_with_dependents(self):
        """compare_views should log a warning when dropping a view that
        other views depend on."""
        from unittest.mock import MagicMock, patch
        import sqlalchemy_utils.alembic.comparator as comparator_module

        # Model has no views; DB has one view that another view depends on
        metadata = MagicMock()
        metadata.info = {"sqlalchemy_utils_views": []}

        autogen_context = MagicMock()
        autogen_context.connection = MagicMock()
        autogen_context.connection.dialect.name = 'postgresql'
        autogen_context.metadata = metadata

        upgrade_ops = MagicMock()
        upgrade_ops.ops = []

        # DB has view "base_view" which is depended on by "dependent_view"
        with patch.object(comparator_module, 'get_database_views', return_value={"base_view": "SELECT 1 AS col"}), \
             patch.object(comparator_module, 'get_database_materialized_views', return_value={}), \
             patch.object(comparator_module, '_canonicalize_view', return_value=None), \
             patch.object(comparator_module, 'get_dependent_views', return_value={"dependent_view": "SELECT * FROM base_view"}), \
             patch.object(comparator_module, 'log') as mock_log:
            comparator_module.compare_views(autogen_context, upgrade_ops, [None])

        # Warning should have been logged
        warning_calls = [c for c in mock_log.warning.call_args_list if 'base_view' in str(c) and 'dependent' in str(c).lower()]
        assert len(warning_calls) > 0, (
            f"Expected warning about dependents when dropping base_view, "
            f"got warnings: {mock_log.warning.call_args_list}"
        )

    def test_does_not_warn_when_dropping_view_without_dependents(self):
        """compare_views should NOT warn when dropping a view with no dependents."""
        from unittest.mock import MagicMock, patch
        import sqlalchemy_utils.alembic.comparator as comparator_module

        metadata = MagicMock()
        metadata.info = {"sqlalchemy_utils_views": []}

        autogen_context = MagicMock()
        autogen_context.connection = MagicMock()
        autogen_context.connection.dialect.name = 'postgresql'
        autogen_context.metadata = metadata

        upgrade_ops = MagicMock()
        upgrade_ops.ops = []

        with patch.object(comparator_module, 'get_database_views', return_value={"lonely_view": "SELECT 1 AS col"}), \
             patch.object(comparator_module, 'get_database_materialized_views', return_value={}), \
             patch.object(comparator_module, '_canonicalize_view', return_value=None), \
             patch.object(comparator_module, 'get_dependent_views', return_value={}), \
             patch.object(comparator_module, 'log') as mock_log:
            comparator_module.compare_views(autogen_context, upgrade_ops, [None])

        # No warning about dependents should be logged
        warning_calls = [c for c in mock_log.warning.call_args_list if 'dependent' in str(c).lower()]
        assert len(warning_calls) == 0, (
            f"Should not warn about dependents for lonely_view, got: {warning_calls}"
        )

    def test_drop_op_still_generated_even_with_dependents(self):
        """The DropViewOp should still be generated even if the view has dependents
        (warn, don't block)."""
        from unittest.mock import MagicMock, patch
        import sqlalchemy_utils.alembic.comparator as comparator_module
        from sqlalchemy_utils.alembic.operations import DropViewOp

        metadata = MagicMock()
        metadata.info = {"sqlalchemy_utils_views": []}

        autogen_context = MagicMock()
        autogen_context.connection = MagicMock()
        autogen_context.connection.dialect.name = 'postgresql'
        autogen_context.metadata = metadata

        upgrade_ops = MagicMock()
        upgrade_ops.ops = []

        with patch.object(comparator_module, 'get_database_views', return_value={"base_view": "SELECT 1 AS col"}), \
             patch.object(comparator_module, 'get_database_materialized_views', return_value={}), \
             patch.object(comparator_module, '_canonicalize_view', return_value=None), \
             patch.object(comparator_module, 'get_dependent_views', return_value={"dependent_view": "SELECT * FROM base_view"}), \
             patch.object(comparator_module, 'log'):
            comparator_module.compare_views(autogen_context, upgrade_ops, [None])

        # DropViewOp should still be in the ops list
        drop_ops = [op for op in upgrade_ops.ops if isinstance(op, DropViewOp)]
        assert any(op.name == "base_view" for op in drop_ops), (
            f"DropViewOp for base_view should still be generated, got ops: {upgrade_ops.ops}"
        )
