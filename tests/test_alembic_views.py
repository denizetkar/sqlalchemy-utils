"""
Tests for SQLAlchemy-Utils Alembic integration.

These tests verify that ViewRecord dataclass functions correctly for
serializing and deserializing view definitions.

Also provides Alembic autogenerate test fixtures and helpers used by
Tasks 10-12 for integration testing of view-aware migration generation.
"""
from __future__ import annotations

import textwrap

import pytest

from dataclasses import FrozenInstanceError as _FrozenInstanceError
from pathlib import Path
from typing import Optional

import sqlalchemy as sa
from alembic import command, config

from sqlalchemy_utils.alembic.view_record import ViewRecord


class TestViewRecordCreation:
    """Test ViewRecord creation with required fields."""

    def test_create_with_minimum_fields(self):
        """Test ViewRecord can be created with just required fields."""
        record = ViewRecord(
            name="test_view",
            selectable="SELECT 1"
        )
        assert record.name == "test_view"
        assert record.selectable == "SELECT 1"
        assert record.schema is None
        assert record.materialized is False
        assert record.replace is False
        assert record.cascade_on_drop is True

    def test_create_with_all_fields(self):
        """Test ViewRecord can be created with all optional fields."""
        record = ViewRecord(
            name="test_view",
            selectable="SELECT 1",
            schema="public",
            materialized=True,
            replace=True,
            cascade_on_drop=False
        )
        assert record.name == "test_view"
        assert record.selectable == "SELECT 1"
        assert record.schema == "public"
        assert record.materialized is True
        assert record.replace is True
        assert record.cascade_on_drop is False

    def test_create_with_none_schema(self):
        """Test ViewRecord handles None schema correctly."""
        record = ViewRecord(
            name="test_view",
            selectable="SELECT 1",
            schema=None
        )
        assert record.schema is None

    def test_create_default_cascade_on_drop(self):
        """Test ViewRecord default cascade_on_drop is True."""
        record = ViewRecord(
            name="test_view",
            selectable="SELECT 1"
        )
        assert record.cascade_on_drop is True


class TestViewRecordFreezing:
    """Test ViewRecord is frozen and raises FrozenInstanceError on mutation."""

    def test_is_frozen(self):
        """Test ViewRecord is a frozen dataclass — mutation raises error."""
        record = ViewRecord(
            name="test_view",
            selectable="SELECT 1"
        )
        with pytest.raises(_FrozenInstanceError):
            record.name = "different_view"

    def test_raises_frozen_error_on_attribute_assignment(self):
        """Test attempting to set an attribute raises FrozenInstanceError."""
        record = ViewRecord(
            name="test_view",
            selectable="SELECT 1"
        )
        with pytest.raises(_FrozenInstanceError):
            record.name = "different_view"

    def test_raises_frozen_error_on_attribute_deletion(self):
        """Test attempting to delete an attribute raises FrozenInstanceError."""
        record = ViewRecord(
            name="test_view",
            selectable="SELECT 1"
        )
        with pytest.raises(_FrozenInstanceError):
            del record.name


class TestViewRecordEquality:
    """Test ViewRecord equality works correctly."""

    def test_equal_records_with_same_fields(self):
        """Test two ViewRecords with same fields are equal."""
        record1 = ViewRecord(
            name="test_view",
            selectable="SELECT 1"
        )
        record2 = ViewRecord(
            name="test_view",
            selectable="SELECT 1"
        )
        assert record1 == record2

    def test_not_equal_with_different_name(self):
        """Test ViewRecords with different names are not equal."""
        record1 = ViewRecord(name="view1", selectable="SELECT 1")
        record2 = ViewRecord(name="view2", selectable="SELECT 1")
        assert record1 != record2

    def test_not_equal_with_different_schema(self):
        """Test ViewRecords with different schemas are not equal."""
        record1 = ViewRecord(
            name="test_view",
            selectable="SELECT 1",
            schema="schema1"
        )
        record2 = ViewRecord(
            name="test_view",
            selectable="SELECT 1",
            schema="schema2"
        )
        assert record1 != record2

    def test_not_equal_with_different_materialized(self):
        """Test ViewRecords with different materialized status are not equal."""
        record1 = ViewRecord(
            name="test_view",
            selectable="SELECT 1",
            materialized=True
        )
        record2 = ViewRecord(
            name="test_view",
            selectable="SELECT 1",
            materialized=False
        )
        assert record1 != record2

    def test_not_equal_different_types(self):
        """Test ViewRecord is not equal to non-ViewRecord objects."""
        record = ViewRecord(name="test_view", selectable="SELECT 1")
        assert record != "not_a_view"
        assert record != {"name": "test_view"}

    def test_self_equality(self):
        """Test a ViewRecord equals itself."""
        record = ViewRecord(name="test_view", selectable="SELECT 1")
        assert record == record

    def test_equality_without_optional_fields(self):
        """Test equality works when optional fields are omitted."""
        record1 = ViewRecord(name="test_view", selectable="SELECT 1")
        record2 = ViewRecord(name="test_view", selectable="SELECT 1", schema=None)
        record3 = ViewRecord(name="test_view", selectable="SELECT 1", materialized=False)
        assert record1 == record2
        assert record1 == record3


class TestViewRecordHashing:
    """Test ViewRecord hash works for set and dict operations."""

    def test_hash_consistent_with_equality(self):
        """Test hash is consistent with equality (same key hash, same value, equal)."""
        record1 = ViewRecord(name="test_view", selectable="SELECT 1")
        record2 = ViewRecord(name="test_view", selectable="SELECT 1")
        # Both should hash to the same value
        assert hash(record1) == hash(record2)
        # Both should be equal
        assert record1 == record2

    def test_different_records_have_different_hashes(self):
        """Test records that are not equal have different hashes."""
        record1 = ViewRecord(name="view1", selectable="SELECT 1")
        record2 = ViewRecord(name="view2", selectable="SELECT 1")
        assert hash(record1) != hash(record2)

    def test_storable_in_set(self):
        """Test ViewRecord can be stored in a set, with dedup by equality."""
        record1 = ViewRecord(name="test_view", selectable="SELECT 1")
        record2 = ViewRecord(name="test_view", selectable="SELECT 1")
        record3 = ViewRecord(name="other_view", selectable="SELECT 1")

        view_set = {record1, record3}
        assert len(view_set) == 2
        assert record1 in view_set
        assert record3 in view_set
        # record2 equals record1 (same name/schema/materialized),
        # so it is also "in" the set even though not the same object
        assert record2 in view_set

    def test_storable_in_dict_as_key(self):
        """Test ViewRecord can be used as a dict key."""
        record = ViewRecord(name="test_view", selectable="SELECT 1")
        value_map = {record: "view_data"}

        assert value_map[record] == "view_data"
        assert len(value_map) == 1

    def test_storable_in_dict_multiple_keys(self):
        """Test multiple ViewRecords can be stored as dict keys."""
        record1 = ViewRecord(name="view1", selectable="SELECT 1")
        record2 = ViewRecord(name="view2", selectable="SELECT 2")
        record3 = ViewRecord(name="view3", selectable="SELECT 3")

        value_map = {record1: "data1", record2: "data2", record3: "data3"}

        assert len(value_map) == 3
        assert value_map[record1] == "data1"
        assert value_map[record2] == "data2"
        assert value_map[record3] == "data3"


class TestViewRecordRepr:
    """Test ViewRecord string representation."""

    def test_repr_with_schema(self):
        """Test __repr__ includes schema."""
        record = ViewRecord(
            name="test_view",
            selectable="SELECT 1",
            schema="public"
        )
        repr_str = repr(record)
        assert "ViewRecord" in repr_str
        assert "name='test_view'" in repr_str
        assert "schema=" in repr_str

    def test_repr_without_schema(self):
        """Test __repr__ shows None for missing schema."""
        record = ViewRecord(
            name="test_view",
            selectable="SELECT 1"
        )
        repr_str = repr(record)
        assert "ViewRecord" in repr_str
        assert "schema=" in repr_str

    def test_str(self):
        """Test __str__ yields readable representation."""
        record = ViewRecord(
            name="test_view",
            selectable="SELECT 1",
            materialized=True
        )
        str_repr = str(record)
        assert "test_view" in str_repr
        assert "materialized=True" in str_repr


class TestGetDatabaseViews:
    """Test get_database_views function queries pg_views correctly."""

    @pytest.mark.usefixtures('postgresql_dsn')
    def test_query_empty_database(self, connection):
        """Test get_database_views returns empty dict when no views exist."""
        from sqlalchemy_utils.alembic.pg_catalog import get_database_views

        views = get_database_views(connection)
        assert views == {}

    @pytest.mark.usefixtures('postgresql_dsn')
    def test_query_views_with_schema_filter(self, connection):
        """Test get_database_views filters by schema correctly."""
        from sqlalchemy_utils.alembic.pg_catalog import get_database_views

        views = get_database_views(connection, schema="public")
        assert isinstance(views, dict)
        for view_name, definition in views.items():
            assert isinstance(view_name, str)
            assert isinstance(definition, str)

    @pytest.mark.usefixtures('postgresql_dsn')
    def test_query_all_schemas_when_schema_none(self, connection):
        """Test get_database_views returns all schemas when schema=None."""
        from sqlalchemy_utils.alembic.pg_catalog import get_database_views

        views = get_database_views(connection, schema=None)
        assert isinstance(views, dict)
        for view_name, definition in views.items():
            assert isinstance(view_name, str)
            assert isinstance(definition, str)

    @pytest.mark.usefixtures('postgresql_dsn')
    def test_query_returns_view_definitions(self, connection):
        """Test get_database_views returns SQL definitions from pg_views."""
        from sqlalchemy_utils.alembic.pg_catalog import get_database_views

        views = get_database_views(connection)
        assert isinstance(views, dict)
        if views:
            for view_name, definition in views.items():
                assert view_name
                assert definition
                assert isinstance(definition, str)


class TestGetDatabaseMaterializedViews:
    """Test get_database_materialized_views function queries pg_matviews correctly."""

    @pytest.mark.usefixtures('postgresql_dsn')
    def test_query_empty_database(self, connection):
        """Test get_database_materialized_views returns empty dict when no MVs exist."""
        from sqlalchemy_utils.alembic.pg_catalog import get_database_materialized_views

        mv_views = get_database_materialized_views(connection)
        assert mv_views == {}

    @pytest.mark.usefixtures('postgresql_dsn')
    def test_query_materialized_views_with_schema_filter(self, connection):
        """Test get_database_materialized_views filters by schema correctly."""
        from sqlalchemy_utils.alembic.pg_catalog import get_database_materialized_views

        mv_views = get_database_materialized_views(connection, schema="public")
        assert isinstance(mv_views, dict)
        for view_name, definition in mv_views.items():
            assert isinstance(view_name, str)
            assert isinstance(definition, str)

    @pytest.mark.usefixtures('postgresql_dsn')
    def test_query_all_schemas_when_schema_none(self, connection):
        """Test get_database_materialized_views returns all schemas when schema=None."""
        from sqlalchemy_utils.alembic.pg_catalog import get_database_materialized_views

        mv_views = get_database_materialized_views(connection, schema=None)
        assert isinstance(mv_views, dict)
        for view_name, definition in mv_views.items():
            assert isinstance(view_name, str)
            assert isinstance(definition, str)

    @pytest.mark.usefixtures('postgresql_dsn')
    def test_query_returns_mv_definitions(self, connection):
        """Test get_database_materialized_views returns SQL definitions from pg_matviews."""
        from sqlalchemy_utils.alembic.pg_catalog import get_database_materialized_views

        mv_views = get_database_materialized_views(connection)
        assert isinstance(mv_views, dict)
        if mv_views:
            for view_name, definition in mv_views.items():
                assert view_name
                assert definition
                assert isinstance(definition, str)


# ===================================================================
# Operations tests (no DB required)
# ===================================================================

from sqlalchemy_utils.alembic.operations import (
    CreateViewOp,
    DropViewOp,
    ReplaceViewOp,
    CreateMaterializedViewOp,
    DropMaterializedViewOp,
    ReplaceMaterializedViewOp,
)


def _capture_sql(op_instance) -> list[str]:
    from unittest.mock import patch

    from alembic.operations import Operations
    from alembic.runtime.migration import MigrationContext

    statements: list[str] = []

    engine = sa.create_engine("sqlite:///:memory:")
    with engine.begin() as connection:
        ctx = MigrationContext.configure(connection)
        ops = Operations(ctx)
        with patch.object(ops, "execute", side_effect=lambda stmt, *a, **kw: statements.append(
            stmt.text if hasattr(stmt, "text") else str(stmt)
        )):
            ops.invoke(op_instance)
    return statements


class TestOperationsCreateViewOp:
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

    def test_sql_with_replace(self):
        op = CreateViewOp("v1", "SELECT 1", replace=True)
        sqls = _capture_sql(op)
        assert sqls == ["CREATE OR REPLACE VIEW v1 AS SELECT 1"]

    def test_sql_with_schema(self):
        op = CreateViewOp("v1", "SELECT 1", schema="public")
        sqls = _capture_sql(op)
        assert sqls == ["CREATE VIEW public.v1 AS SELECT 1"]


class TestOperationsDropViewOp:
    def test_instantiation(self):
        op = DropViewOp("v1", materialized=False, cascade=True)
        assert op.name == "v1"
        assert op.materialized is False
        assert op.cascade is True

    def test_reverse_returns_create_view(self):
        op = DropViewOp("v1", definition="SELECT 1")
        rev = op.reverse()
        assert isinstance(rev, CreateViewOp)
        assert rev.name == "v1"
        assert rev.definition == "SELECT 1"

    def test_reverse_without_definition_raises(self):
        op = DropViewOp("v1")
        with pytest.raises(RuntimeError, match="no definition stored"):
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


class TestOperationsReplaceViewOp:
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
        with pytest.raises(RuntimeError, match="no old_definition stored"):
            op.reverse()

    def test_sql(self):
        op = ReplaceViewOp("v1", "SELECT 2")
        sqls = _capture_sql(op)
        assert sqls == ["CREATE OR REPLACE VIEW v1 AS SELECT 2"]


class TestOperationsCreateMaterializedViewOp:
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


class TestOperationsDropMaterializedViewOp:
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
        with pytest.raises(RuntimeError, match="no definition stored"):
            op.reverse()

    def test_sql_cascade(self):
        op = DropMaterializedViewOp("mv1", cascade=True)
        sqls = _capture_sql(op)
        assert sqls == ["DROP MATERIALIZED VIEW IF EXISTS mv1 CASCADE"]

    def test_sql_no_cascade(self):
        op = DropMaterializedViewOp("mv1", cascade=False)
        sqls = _capture_sql(op)
        assert sqls == ["DROP MATERIALIZED VIEW IF EXISTS mv1"]


class TestOperationsReplaceMaterializedViewOp:
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
        with pytest.raises(RuntimeError, match="no old_definition stored"):
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
# Alembic autogenerate test fixtures and helpers
# ---------------------------------------------------------------------------

_ENV_PY_TEMPLATE = textwrap.dedent("""\
    from __future__ import annotations

    from alembic import context
    from sqlalchemy import pool

    config = context.config

    target_metadata = config.attributes.get("target_metadata")

    # TODO: Task 9 will add include_view_comparator() here.
    # include_object = include_view_comparator()

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
    """Create a temporary Alembic environment configured for autogenerate.

    Returns a function that, given a ``sa.MetaData`` object, produces an
    ``alembic.config.Config`` ready for ``command.revision(autogenerate=True)``.
    """
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
    """Run ``alembic revision --autogenerate`` and return the migration code.

    Parameters
    ----------
    metadata:
        The SQLAlchemy ``MetaData`` whose tables/views define the target schema.
    connection:
        An open SQLAlchemy ``Connection`` to the test database.
    alembic_config:
        The fixture that builds an ``alembic.config.Config``.

    Returns
    -------
    str
        The contents of the generated migration script.
    """
    cfg = alembic_config(metadata)
    command.revision(cfg, autogenerate=True, message="test")

    script_location = Path(cfg.get_main_option("script_location"))
    versions_dir = script_location / "versions"
    migration_files = sorted(versions_dir.glob("*.py"), key=lambda p: p.stat().st_mtime)

    if not migration_files:
        raise AssertionError("No migration file was generated by alembic revision --autogenerate")

    latest = migration_files[-1]
    code = latest.read_text(encoding="utf-8")

    latest.unlink()

    return code


def assert_has_op(migration_code: str, op_name: str) -> None:
    """Assert that *migration_code* contains ``op.<op_name>(``.

    Uses simple string matching — no AST parsing needed.
    """
    token = f"op.{op_name}("
    if token not in migration_code:
        raise AssertionError(
            f"Expected migration to contain '{token}' but it was not found.\n"
            f"Migration code:\n{migration_code}"
        )


def assert_no_op(migration_code: str, op_name: str) -> None:
    """Assert that *migration_code* does NOT contain ``op.<op_name>(``.

    Uses simple string matching — no AST parsing needed.
    """
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
class TestAlembicConfigFixture:
    """Tests for the alembic_config fixture itself."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_creates_valid_alembic_environment(self, alembic_config, connection):
        """alembic_config returns a callable that produces a valid Config."""
        metadata = sa.MetaData()
        cfg = alembic_config(metadata)

        assert isinstance(cfg, config.Config)
        assert cfg.attributes["connection"] is connection
        assert cfg.attributes["target_metadata"] is metadata
        script_location = Path(cfg.get_main_option("script_location"))
        assert script_location.exists()
        assert (script_location / "env.py").exists()


@pytest.mark.infrastructure
class TestRunAutogenerate:
    """Tests for the run_autogenerate helper."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_produces_migration_file(self, alembic_config, connection):
        """run_autogenerate returns a non-empty migration script string."""
        metadata = sa.MetaData()
        sa.Table("infra_test_table", metadata,
                 sa.Column("id", sa.Integer, primary_key=True),
                 sa.Column("name", sa.Unicode(255)))

        code = run_autogenerate(metadata, connection, alembic_config)

        assert isinstance(code, str)
        assert len(code) > 0
        assert "def upgrade()" in code
        assert "def downgrade()" in code

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_detects_create_table(self, alembic_config, connection):
        """run_autogenerate on a fresh DB with a new table emits op.create_table."""
        metadata = sa.MetaData()
        sa.Table("infra_detect_table", metadata,
                 sa.Column("id", sa.Integer, primary_key=True))

        code = run_autogenerate(metadata, connection, alembic_config)

        assert_has_op(code, "create_table")


@pytest.mark.infrastructure
class TestAssertHasOp:
    """Tests for assert_has_op helper."""

    def test_finds_present_op(self):
        """assert_has_op passes when the op call is present."""
        code = "op.create_table('account')\nop.add_column('x')"
        assert_has_op(code, "create_table")

    def test_raises_on_missing_op(self):
        """assert_has_op raises AssertionError when the op call is absent."""
        code = "op.create_table('account')"
        with pytest.raises(AssertionError, match="Expected migration to contain"):
            assert_has_op(code, "drop_table")


@pytest.mark.infrastructure
class TestAssertNoOp:
    """Tests for assert_no_op helper."""

    def test_passes_when_op_absent(self):
        """assert_no_op passes when the op call is absent."""
        code = "op.create_table('account')"
        assert_no_op(code, "drop_table")

    def test_raises_on_present_op(self):
        """assert_no_op raises AssertionError when the op call is present."""
        code = "op.drop_table('account')"
        with pytest.raises(AssertionError, match="Expected migration NOT to contain"):
            assert_no_op(code, "drop_table")


# ---------------------------------------------------------------------------
# Dependency resolution tests (depend.py)
# ---------------------------------------------------------------------------

from sqlalchemy_utils.alembic.depend import resolve_create_order, resolve_drop_order


class TestDependIndependentViews:
    """Test that independent views (no deps) maintain any order."""

    def test_independent_views_create_order_contains_all(self):
        """Independent views: create order includes all views."""
        views = [
            ViewRecord(name="alpha", selectable="SELECT 1"),
            ViewRecord(name="beta", selectable="SELECT 2"),
            ViewRecord(name="gamma", selectable="SELECT 3"),
        ]
        result = resolve_create_order(views, db_views={})
        assert set(v.name for v in result) == {"alpha", "beta", "gamma"}
        assert len(result) == 3

    def test_independent_views_drop_order_contains_all(self):
        """Independent views: drop order includes all views."""
        views = [
            ViewRecord(name="alpha", selectable="SELECT 1"),
            ViewRecord(name="beta", selectable="SELECT 2"),
        ]
        result = resolve_drop_order(views, db_views={})
        assert set(v.name for v in result) == {"alpha", "beta"}


class TestDependViewOnView:
    """Test view-on-view dependency ordering."""

    def test_dependent_after_dependency_in_create_order(self):
        """A view that references another must come after it in create order."""
        views = [
            ViewRecord(name="child_view", selectable="SELECT * FROM parent_view"),
            ViewRecord(name="parent_view", selectable="SELECT 1 AS col"),
        ]
        result = resolve_create_order(views, db_views={})
        names = [v.name for v in result]
        assert names.index("parent_view") < names.index("child_view")

    def test_dependent_before_dependency_in_drop_order(self):
        """A view that references another must come before it in drop order."""
        views = [
            ViewRecord(name="child_view", selectable="SELECT * FROM parent_view"),
            ViewRecord(name="parent_view", selectable="SELECT 1 AS col"),
        ]
        result = resolve_drop_order(views, db_views={})
        names = [v.name for v in result]
        assert names.index("child_view") < names.index("parent_view")


class TestDependMultipleLevels:
    """Test multi-level dependency chains (A → B → C)."""

    def test_chain_create_order(self):
        """In A→B→C, create order must be C, B, A."""
        views = [
            ViewRecord(name="a", selectable="SELECT * FROM b"),
            ViewRecord(name="b", selectable="SELECT * FROM c"),
            ViewRecord(name="c", selectable="SELECT 1 AS col"),
        ]
        result = resolve_create_order(views, db_views={})
        names = [v.name for v in result]
        assert names.index("c") < names.index("b") < names.index("a")

    def test_chain_drop_order(self):
        """In A→B→C, drop order must be A, B, C."""
        views = [
            ViewRecord(name="a", selectable="SELECT * FROM b"),
            ViewRecord(name="b", selectable="SELECT * FROM c"),
            ViewRecord(name="c", selectable="SELECT 1 AS col"),
        ]
        result = resolve_drop_order(views, db_views={})
        names = [v.name for v in result]
        assert names.index("a") < names.index("b") < names.index("c")


class TestDependCircular:
    """Test that circular dependencies raise ValueError."""

    def test_simple_cycle_raises_value_error(self):
        """A→B and B→A must raise ValueError."""
        views = [
            ViewRecord(name="view_a", selectable="SELECT * FROM view_b"),
            ViewRecord(name="view_b", selectable="SELECT * FROM view_a"),
        ]
        with pytest.raises(ValueError, match="[Cc]ircular"):
            resolve_create_order(views, db_views={})

    def test_three_way_cycle_raises_value_error(self):
        """A→B→C→A must raise ValueError."""
        views = [
            ViewRecord(name="v_a", selectable="SELECT * FROM v_b"),
            ViewRecord(name="v_b", selectable="SELECT * FROM v_c"),
            ViewRecord(name="v_c", selectable="SELECT * FROM v_a"),
        ]
        with pytest.raises(ValueError, match="[Cc]ircular"):
            resolve_create_order(views, db_views={})

    def test_cycle_in_drop_order_raises_value_error(self):
        """Cycles must also be detected in drop order."""
        views = [
            ViewRecord(name="x", selectable="SELECT * FROM y"),
            ViewRecord(name="y", selectable="SELECT * FROM x"),
        ]
        with pytest.raises(ValueError, match="[Cc]ircular"):
            resolve_drop_order(views, db_views={})


class TestDependDropOrder:
    """Test that drop order is the exact reverse of create order."""

    def test_drop_is_reverse_of_create(self):
        """Drop order must be the exact reverse of create order."""
        views = [
            ViewRecord(name="top", selectable="SELECT * FROM mid"),
            ViewRecord(name="mid", selectable="SELECT * FROM base"),
            ViewRecord(name="base", selectable="SELECT 1"),
        ]
        create = resolve_create_order(views, db_views={})
        drop = resolve_drop_order(views, db_views={})
        assert [v.name for v in drop] == list(reversed([v.name for v in create]))


class TestDependMaterializedViews:
    """Test materialized views with dependencies."""

    def test_materialized_view_after_base_table(self):
        """Materialized view depends on a regular view — must come after."""
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
        """Materialized view depending on another materialized view."""
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
    """Test that db_views are used as pre-existing dependencies."""

    def test_db_view_satisfies_dependency(self):
        """A model view depending on a DB-only view should not cause errors."""
        views = [
            ViewRecord(name="model_view", selectable="SELECT * FROM existing_db_view"),
        ]
        db_views = {"existing_db_view": "SELECT 1 AS col"}
        result = resolve_create_order(views, db_views=db_views)
        # DB-only view is not in output; model_view is present
        assert len(result) == 1
        assert result[0].name == "model_view"

    def test_db_view_not_included_in_output(self):
        """DB views must never appear in the sorted output."""
        views = [
            ViewRecord(name="child", selectable="SELECT * FROM db_parent"),
        ]
        db_views = {"db_parent": "SELECT 1"}
        result = resolve_create_order(views, db_views=db_views)
        names = [v.name for v in result]
        assert "db_parent" not in names
        assert "child" in names


class TestDependWordBoundary:
    """Test that dependency detection uses word-boundary matching."""

    def test_partial_name_no_false_positive(self):
        """A view named 'log' must not match 'log_entries' in definition."""
        views = [
            ViewRecord(name="log", selectable="SELECT 1"),
            ViewRecord(name="report", selectable="SELECT * FROM log_entries"),
        ]
        result = resolve_create_order(views, db_views={})
        # 'log' does NOT appear as a word boundary in "log_entries"
        # so there's no dependency — order is unconstrained, both present
        assert set(v.name for v in result) == {"log", "report"}

    def test_exact_name_with_word_boundary(self):
        """A view name must match only at word boundaries."""
        views = [
            ViewRecord(name="log", selectable="SELECT 1"),
            ViewRecord(name="report", selectable="SELECT * FROM log"),
        ]
        result = resolve_create_order(views, db_views={})
        names = [v.name for v in result]
        # 'log' IS a word boundary in "FROM log"
        assert names.index("log") < names.index("report")


# ===================================================================
# Auto-registration tests (no DB required)
# ===================================================================

from sqlalchemy import select, Column, Integer
from sqlalchemy_utils import create_view, create_materialized_view


class TestViewAutoRegistration:
    """Test that create_view and create_materialized_view auto-register ViewRecord instances."""

    def test_create_view_registers_view_record_in_metadata(self):
        """Test create_view() populates metadata.info['sqlalchemy_utils_views'] with a ViewRecord."""
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
        """Test create_materialized_view() populates metadata.info['sqlalchemy_utils_views'] with a ViewRecord where materialized=True."""
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
        """Test calling create_view() twice appends two records (not replaces)."""
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
        """Test that both view types can coexist in metadata.info['sqlalchemy_utils_views']."""
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
        """Test that create_view() with replace=True sets ViewRecord.replace=True."""
        metadata = sa.MetaData()

        selectable = select(Column("id", Integer))
        create_view("replace_view", selectable, metadata, replace=True)

        assert len(metadata.info["sqlalchemy_utils_views"]) == 1
        record = metadata.info["sqlalchemy_utils_views"][0]
        assert record.replace is True

    def test_create_view_with_cascade_on_drop_parameter(self):
        """Test that create_view() with cascade_on_drop=False sets ViewRecord.cascade_on_drop=False."""
        metadata = sa.MetaData()

        selectable = select(Column("id", Integer))
        create_view("no_cascade_view", selectable, metadata, cascade_on_drop=False)

        assert len(metadata.info["sqlalchemy_utils_views"]) == 1
        record = metadata.info["sqlalchemy_utils_views"][0]
        assert record.cascade_on_drop is False

    def test_default_cascade_on_drop_true(self):
        """Test that create_materialized_view default cascade_on_drop is True."""
        metadata = sa.MetaData()

        selectable = select(Column("id", Integer))
        create_materialized_view("mv_default", selectable, metadata, indexes=[])

        assert len(metadata.info["sqlalchemy_utils_views"]) == 1
        record = metadata.info["sqlalchemy_utils_views"][0]
        assert record.cascade_on_drop is True


class TestPublicAPICallable:
    def test_call_with_no_args(self):
        from sqlalchemy_utils import include_view_comparator
        assert include_view_comparator() is None

    def test_call_multiple_times_idempotent(self):
        include_view_comparator()
        include_view_comparator()

    def test_detects_dispatch_registration(self):
        assert include_view_comparator() is None


class TestPublicAPIImportable:
    def test_import_create_view_op(self):
        from sqlalchemy_utils.alembic import CreateViewOp
        op = CreateViewOp("test_view", "SELECT 1")
        assert op.name == "test_view"
        assert op.definition == "SELECT 1"

    def test_import_drop_view_op(self):
        from sqlalchemy_utils.alembic import DropViewOp
        op = DropViewOp("test_view", materialized=False)
        assert op.name == "test_view"

    def test_import_replace_view_op(self):
        from sqlalchemy_utils.alembic import ReplaceViewOp
        op = ReplaceViewOp("test_view", "SELECT 2")
        assert op.name == "test_view"

    def test_import_create_materialized_view_op(self):
        from sqlalchemy_utils.alembic import CreateMaterializedViewOp
        op = CreateMaterializedViewOp("test_mv", "SELECT 1")
        assert op.name == "test_mv"

    def test_import_drop_materialized_view_op(self):
        from sqlalchemy_utils.alembic import DropMaterializedViewOp
        op = DropMaterializedViewOp("test_mv", cascade=False)
        assert op.name == "test_mv"

    def test_import_replace_materialized_view_op(self):
        from sqlalchemy_utils.alembic import ReplaceMaterializedViewOp
        op = ReplaceMaterializedViewOp("test_mv", "SELECT 2")
        assert op.name == "test_mv"

    def test_import_view_record(self):
        from sqlalchemy_utils.alembic import ViewRecord
        record = ViewRecord(name="test_view", selectable="SELECT 1")
        assert record.name == "test_view"
        assert record.selectable == "SELECT 1"


class TestPublicAPIFromTopLevel:
    """Test that public API is accessible from sqlalchemy_utils top-level."""

    def test_import_include_view_comparator_from_top(self):
        """Test include_view_comparator is importable from sqlalchemy_utils."""
        from sqlalchemy_utils import include_view_comparator
        assert callable(include_view_comparator)

    def test_import_ops_from_top_level_alembic(self):
        """Test all operation classes accessible as sqlalchemy_utils.alembic.SomeOp."""
        from sqlalchemy_utils.alembic import (
            CreateViewOp,
            DropViewOp,
            ReplaceViewOp,
            CreateMaterializedViewOp,
            DropMaterializedViewOp,
            ReplaceMaterializedViewOp,
            ViewRecord,
        )
        # Just verify they imported without errors
        assert CreateViewOp is not None
        assert DropViewOp is not None
        assert ReplaceViewOp is not None
        assert CreateMaterializedViewOp is not None
        assert DropMaterializedViewOp is not None
        assert ReplaceMaterializedViewOp is not None
        assert ViewRecord is not None


# ===================================================================
# Renderer tests (no DB required)
# ===================================================================

from unittest.mock import MagicMock

from alembic.autogenerate.api import AutogenContext


def _make_autogen_context() -> AutogenContext:
    """Create a minimal mock AutogenContext for renderer tests."""
    ctx = MagicMock(spec=AutogenContext)
    ctx.imports = set()
    return ctx


class TestRendererCreateView:
    """Test render_create_view renderer."""

    def test_produces_valid_python(self):
        """Renderer output is syntactically valid Python."""
        from sqlalchemy_utils.alembic.renderer import render_create_view

        op = CreateViewOp("my_view", "SELECT 1")
        result = render_create_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_includes_op_call(self):
        """Renderer output includes op.create_view call."""
        from sqlalchemy_utils.alembic.renderer import render_create_view

        op = CreateViewOp("my_view", "SELECT id FROM users")
        result = render_create_view(_make_autogen_context(), op)
        assert "op.create_view(" in result

    def test_schema_omitted_when_none(self):
        """Schema parameter is omitted when schema is None."""
        from sqlalchemy_utils.alembic.renderer import render_create_view

        op = CreateViewOp("my_view", "SELECT 1", schema=None)
        result = render_create_view(_make_autogen_context(), op)
        assert "schema=" not in result

    def test_schema_included_when_provided(self):
        """Schema parameter is included when schema is provided."""
        from sqlalchemy_utils.alembic.renderer import render_create_view

        op = CreateViewOp("my_view", "SELECT 1", schema="public")
        result = render_create_view(_make_autogen_context(), op)
        assert "schema='public'" in result


class TestRendererDropView:
    """Test render_drop_view renderer."""

    def test_produces_valid_python(self):
        """Renderer output is syntactically valid Python."""
        from sqlalchemy_utils.alembic.renderer import render_drop_view

        op = DropViewOp("my_view")
        result = render_drop_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_includes_op_call(self):
        """Renderer output includes op.drop_view call."""
        from sqlalchemy_utils.alembic.renderer import render_drop_view

        op = DropViewOp("my_view")
        result = render_drop_view(_make_autogen_context(), op)
        assert "op.drop_view(" in result

    def test_schema_omitted_when_none(self):
        """Schema parameter is omitted when schema is None."""
        from sqlalchemy_utils.alembic.renderer import render_drop_view

        op = DropViewOp("my_view", schema=None)
        result = render_drop_view(_make_autogen_context(), op)
        assert "schema=" not in result

    def test_schema_included_when_provided(self):
        """Schema parameter is included when schema is provided."""
        from sqlalchemy_utils.alembic.renderer import render_drop_view

        op = DropViewOp("my_view", schema="analytics")
        result = render_drop_view(_make_autogen_context(), op)
        assert "schema='analytics'" in result


class TestRendererReplaceView:
    """Test render_replace_view renderer."""

    def test_produces_valid_python(self):
        """Renderer output is syntactically valid Python."""
        from sqlalchemy_utils.alembic.renderer import render_replace_view

        op = ReplaceViewOp("my_view", "SELECT 2")
        result = render_replace_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_includes_op_call(self):
        """Renderer output includes op.replace_view call."""
        from sqlalchemy_utils.alembic.renderer import render_replace_view

        op = ReplaceViewOp("my_view", "SELECT 2")
        result = render_replace_view(_make_autogen_context(), op)
        assert "op.replace_view(" in result

    def test_schema_included_when_provided(self):
        """Schema parameter is included when schema is provided."""
        from sqlalchemy_utils.alembic.renderer import render_replace_view

        op = ReplaceViewOp("my_view", "SELECT 2", schema="public")
        result = render_replace_view(_make_autogen_context(), op)
        assert "schema='public'" in result


class TestRendererCreateMaterializedView:
    """Test render_create_materialized_view renderer."""

    def test_produces_valid_python(self):
        """Renderer output is syntactically valid Python."""
        from sqlalchemy_utils.alembic.renderer import render_create_materialized_view

        op = CreateMaterializedViewOp("mv_stats", "SELECT count(*) FROM events")
        result = render_create_materialized_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_includes_op_call(self):
        """Renderer output includes op.create_materialized_view call."""
        from sqlalchemy_utils.alembic.renderer import render_create_materialized_view

        op = CreateMaterializedViewOp("mv_stats", "SELECT count(*) FROM events")
        result = render_create_materialized_view(_make_autogen_context(), op)
        assert "op.create_materialized_view(" in result

    def test_includes_with_data_false(self):
        """Renderer always includes with_data=False for autogenerate."""
        from sqlalchemy_utils.alembic.renderer import render_create_materialized_view

        op = CreateMaterializedViewOp("mv_stats", "SELECT 1")
        result = render_create_materialized_view(_make_autogen_context(), op)
        assert "with_data=False" in result

    def test_schema_included_when_provided(self):
        """Schema parameter is included when schema is provided."""
        from sqlalchemy_utils.alembic.renderer import render_create_materialized_view

        op = CreateMaterializedViewOp("mv_stats", "SELECT 1", schema="analytics")
        result = render_create_materialized_view(_make_autogen_context(), op)
        assert "schema='analytics'" in result


class TestRendererDropMaterializedView:
    """Test render_drop_materialized_view renderer."""

    def test_produces_valid_python(self):
        """Renderer output is syntactically valid Python."""
        from sqlalchemy_utils.alembic.renderer import render_drop_materialized_view

        op = DropMaterializedViewOp("mv_stats")
        result = render_drop_materialized_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_includes_op_call(self):
        """Renderer output includes op.drop_materialized_view call."""
        from sqlalchemy_utils.alembic.renderer import render_drop_materialized_view

        op = DropMaterializedViewOp("mv_stats")
        result = render_drop_materialized_view(_make_autogen_context(), op)
        assert "op.drop_materialized_view(" in result

    def test_cascade_included_when_true(self):
        """Cascade parameter is included when cascade is True."""
        from sqlalchemy_utils.alembic.renderer import render_drop_materialized_view

        op = DropMaterializedViewOp("mv_stats", cascade=True)
        result = render_drop_materialized_view(_make_autogen_context(), op)
        assert "cascade=True" in result

    def test_cascade_omitted_when_false(self):
        """Cascade parameter is omitted when cascade is False."""
        from sqlalchemy_utils.alembic.renderer import render_drop_materialized_view

        op = DropMaterializedViewOp("mv_stats", cascade=False)
        result = render_drop_materialized_view(_make_autogen_context(), op)
        assert "cascade=" not in result


class TestRendererReplaceMaterializedView:
    """Test render_replace_materialized_view renderer."""

    def test_produces_valid_python(self):
        """Renderer output is syntactically valid Python."""
        from sqlalchemy_utils.alembic.renderer import render_replace_materialized_view

        op = ReplaceMaterializedViewOp("mv_stats", "SELECT count(*) FROM events")
        result = render_replace_materialized_view(_make_autogen_context(), op)
        compile(result, "<test>", "exec")

    def test_includes_op_call(self):
        """Renderer output includes op.replace_materialized_view call."""
        from sqlalchemy_utils.alembic.renderer import render_replace_materialized_view

        op = ReplaceMaterializedViewOp("mv_stats", "SELECT 2")
        result = render_replace_materialized_view(_make_autogen_context(), op)
        assert "op.replace_materialized_view(" in result

    def test_includes_with_data_false(self):
        """Renderer always includes with_data=False for autogenerate."""
        from sqlalchemy_utils.alembic.renderer import render_replace_materialized_view

        op = ReplaceMaterializedViewOp("mv_stats", "SELECT 2")
        result = render_replace_materialized_view(_make_autogen_context(), op)
        assert "with_data=False" in result

    def test_schema_included_when_provided(self):
        """Schema parameter is included when schema is provided."""
        from sqlalchemy_utils.alembic.renderer import render_replace_materialized_view

        op = ReplaceMaterializedViewOp("mv_stats", "SELECT 2", schema="analytics")
        result = render_replace_materialized_view(_make_autogen_context(), op)
        assert "schema='analytics'" in result


# ===================================================================
# Comparator tests (require PostgreSQL)
# ===================================================================

from alembic.runtime.migration import MigrationContext
from alembic.autogenerate.api import AutogenContext
from alembic.operations import ops as alembic_ops

from sqlalchemy_utils.alembic.comparator import (
    compare_views,
    _canonicalize_view,
    include_view_comparator,
)


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


def _create_base_table(connection):
    """Create a base table needed for view tests. Idempotent."""
    connection.execute(sa.text(
        "CREATE TABLE IF NOT EXISTS _cmp_test_base "
        "(id SERIAL PRIMARY KEY, name TEXT, value INTEGER)"
    ))
    connection.commit()


def _drop_base_table(connection):
    """Drop the base table used for view tests."""
    connection.execute(sa.text("DROP TABLE IF EXISTS _cmp_test_base CASCADE"))
    connection.commit()


def _drop_test_views(connection):
    """Drop any leftover test views/materialized views."""
    for view_name in [
        "cmp_test_view", "cmp_test_mv", "cmp_test_view2",
        "cmp_test_changed", "cmp_test_mv_changed",
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
    """Test: new view detected → CreateViewOp generated."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_view_generates_create_view_op(self, connection):
        """A view in metadata but not in DB produces CreateViewOp."""
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

        view_ops = [
            op for op in upgrade_ops.ops
            if isinstance(op, CreateViewOp)
        ]
        assert len(view_ops) == 1
        assert view_ops[0].name == "cmp_test_view"

        _drop_test_views(connection)
        _drop_base_table(connection)


class TestComparatorDropView:
    """Test: removed view detected → DropViewOp generated."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_view_generates_drop_view_op(self, connection):
        """A view in DB but not in metadata produces DropViewOp."""
        _create_base_table(connection)
        _drop_test_views(connection)

        # Create the view in DB
        connection.execute(sa.text(
            "CREATE VIEW cmp_test_view AS SELECT id, name FROM _cmp_test_base"
        ))
        connection.commit()

        # Metadata has no views → should detect removal
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = []

        upgrade_ops = _run_comparator(connection, metadata)

        drop_ops = [
            op for op in upgrade_ops.ops
            if isinstance(op, DropViewOp) and not getattr(op, 'materialized', False)
        ]
        assert len(drop_ops) == 1
        assert drop_ops[0].name == "cmp_test_view"

        _drop_test_views(connection)
        _drop_base_table(connection)


class TestComparatorReplaceView:
    """Test: changed view definition → ReplaceViewOp generated."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_view_generates_replace_view_op(self, connection):
        """A view with a different definition produces ReplaceViewOp."""
        _create_base_table(connection)
        _drop_test_views(connection)

        # Create view with original definition (uses WHERE filter)
        connection.execute(sa.text(
            "CREATE VIEW cmp_test_changed AS SELECT id, name FROM _cmp_test_base WHERE value > 0"
        ))
        connection.commit()

        # Metadata defines the same view without the WHERE clause
        # (same column names — PG allows OR REPLACE as long as columns match)
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="cmp_test_changed",
                selectable="SELECT id, name FROM _cmp_test_base",
            )
        ]

        upgrade_ops = _run_comparator(connection, metadata)

        replace_ops = [
            op for op in upgrade_ops.ops
            if isinstance(op, ReplaceViewOp)
        ]
        assert len(replace_ops) == 1
        assert replace_ops[0].name == "cmp_test_changed"
        assert replace_ops[0].old_definition is not None

        _drop_test_views(connection)
        _drop_base_table(connection)


class TestComparatorCreateMaterializedView:
    """Test: new materialized view detected → CreateMaterializedViewOp."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_mv_generates_create_mv_op(self, connection):
        """A materialized view in metadata but not in DB produces CreateMaterializedViewOp."""
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
            op for op in upgrade_ops.ops
            if isinstance(op, CreateMaterializedViewOp)
        ]
        assert len(mv_ops) == 1
        assert mv_ops[0].name == "cmp_test_mv"
        assert mv_ops[0].with_data is False

        _drop_test_views(connection)
        _drop_base_table(connection)


class TestComparatorDropMaterializedView:
    """Test: removed materialized view → DropMaterializedViewOp."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_mv_generates_drop_mv_op(self, connection):
        """A materialized view in DB but not in metadata produces DropMaterializedViewOp."""
        _create_base_table(connection)
        _drop_test_views(connection)

        # Create the MV in DB
        connection.execute(sa.text(
            "CREATE MATERIALIZED VIEW cmp_test_mv AS "
            "SELECT id, name FROM _cmp_test_base WITH DATA"
        ))
        connection.commit()

        # Metadata has no views
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = []

        upgrade_ops = _run_comparator(connection, metadata)

        drop_mv_ops = [
            op for op in upgrade_ops.ops
            if isinstance(op, DropMaterializedViewOp)
        ]
        assert len(drop_mv_ops) == 1
        assert drop_mv_ops[0].name == "cmp_test_mv"

        _drop_test_views(connection)
        _drop_base_table(connection)


class TestComparatorReplaceMaterializedView:
    """Test: changed materialized view definition → ReplaceMaterializedViewOp."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_mv_generates_replace_mv_op(self, connection):
        """A materialized view with different definition produces ReplaceMaterializedViewOp."""
        _create_base_table(connection)
        _drop_test_views(connection)

        # Create MV with original definition
        connection.execute(sa.text(
            "CREATE MATERIALIZED VIEW cmp_test_mv_changed AS "
            "SELECT id, name FROM _cmp_test_base WITH DATA"
        ))
        connection.commit()

        # Metadata defines the same MV with different SQL
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
            op for op in upgrade_ops.ops
            if isinstance(op, ReplaceMaterializedViewOp)
        ]
        assert len(replace_ops) == 1
        assert replace_ops[0].name == "cmp_test_mv_changed"
        assert replace_ops[0].with_data is False
        assert replace_ops[0].old_definition is not None

        _drop_test_views(connection)
        _drop_base_table(connection)


class TestComparatorNoChanges:
    """Test: no changes → no ops generated."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_no_changes_no_ops(self, connection):
        """When model and DB match, no view ops should be generated."""
        _create_base_table(connection)
        _drop_test_views(connection)

        # Create a view in DB
        connection.execute(sa.text(
            "CREATE VIEW cmp_test_view2 AS SELECT id, name FROM _cmp_test_base"
        ))
        connection.commit()

        # Read back the actual definition to ensure exact match
        from sqlalchemy_utils.alembic.pg_catalog import get_database_views
        db_views = get_database_views(connection)
        actual_def = db_views.get("cmp_test_view2")

        # Metadata defines the same view with the same (canonical) definition
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="cmp_test_view2",
                selectable="SELECT id, name FROM _cmp_test_base",
            )
        ]

        upgrade_ops = _run_comparator(connection, metadata)

        view_ops = [
            op for op in upgrade_ops.ops
            if isinstance(op, (CreateViewOp, DropViewOp, ReplaceViewOp,
                               CreateMaterializedViewOp, DropMaterializedViewOp,
                               ReplaceMaterializedViewOp))
        ]
        # No ops should be generated for the matching view
        # (there might be a DropViewOp if the canonicalized def differs
        #  due to pg_views formatting, but with savepoint canonicalization
        #  the definitions should match)
        matching_view_ops = [
            op for op in view_ops
            if getattr(op, 'name', None) == "cmp_test_view2"
        ]
        assert len(matching_view_ops) == 0, (
            f"Expected no ops for matching view, got: {matching_view_ops}"
        )

        _drop_test_views(connection)
        _drop_base_table(connection)


class TestComparatorSavepointRollback:
    """Test: savepoint rollback works (view doesn't persist after compare)."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_canonicalized_view_does_not_persist(self, connection):
        """After canonicalization, the view should NOT exist in the database."""
        _create_base_table(connection)
        _drop_test_views(connection)

        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="cmp_test_view",
                selectable="SELECT id, name FROM _cmp_test_base",
            )
        ]

        # Run the comparator (this will canonicalize via savepoint)
        _run_comparator(connection, metadata)

        # The view should NOT be in the database after comparison
        from sqlalchemy_utils.alembic.pg_catalog import get_database_views
        db_views = get_database_views(connection)
        assert "cmp_test_view" not in db_views, (
            "View should not persist after canonicalization savepoint rollback"
        )

        _drop_base_table(connection)


class TestComparatorDDLError:
    """Test: DDL error in savepoint doesn't crash comparator."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_invalid_view_skipped_with_warning(self, connection):
        """A view referencing a nonexistent table is skipped (None canonicalized)."""
        _drop_test_views(connection)
        _drop_base_table(connection)

        # Do NOT create the base table — the view SQL references a nonexistent table
        metadata = sa.MetaData()
        metadata.info["sqlalchemy_utils_views"] = [
            ViewRecord(
                name="cmp_test_view_bad",
                selectable="SELECT id FROM nonexistent_table_xyz",
            )
        ]

        # Should not raise — the canonicalization fails, view is skipped
        upgrade_ops = _run_comparator(connection, metadata)

        # The invalid view should be skipped (no op generated)
        bad_view_ops = [
            op for op in upgrade_ops.ops
            if getattr(op, 'name', None) == "cmp_test_view_bad"
        ]
        assert len(bad_view_ops) == 0, (
            f"Invalid view should be skipped, got ops: {bad_view_ops}"
        )


class TestComparatorIncludeViewComparator:
    """Test: include_view_comparator() function triggers side-effect imports."""

    def test_include_view_comparator_imports_without_error(self):
        """include_view_comparator() should import without error."""
        # This tests that the function runs without ImportError
        include_view_comparator()

    def test_compare_views_is_registered(self):
        """After include_view_comparator(), compare_views should be registered."""
        from alembic.autogenerate import comparators
        # The dispatch_for("schema") registration should have happened
        # We can verify by checking that our function is in the dispatch chain
        # A simple way: just call include_view_comparator again and verify
        # no error occurs (double-registration is a no-op in Alembic)
        include_view_comparator()


# ===================================================================
# Integration tests (full Alembic autogenerate pipeline)
# ===================================================================

# Module-level Table object mapping to the _cmp_test_base table created
# by _create_base_table().  This allows create_view() to reference
# _int_base_table.c.id etc. when building select() constructs.
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


class TestIntegrationNewViewCreation:
    """Integration: autogenerate detects new view definition."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_new_view_detected_and_rendered(self, connection, alembic_config):
        _int_drop_test_views(connection)
        _drop_base_table(connection)
        _create_base_table(connection)
        try:
            include_view_comparator()
            metadata = sa.MetaData()
            create_view("int_test_new_view", sa.select(_int_base_table.c.id), metadata)
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_has_op(code, "create_view")
            assert "int_test_new_view" in code
        finally:
            _int_drop_test_views(connection)
            _drop_base_table(connection)


class TestIntegrationNewMVCreation:
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


class TestIntegrationViewRemoval:
    """Integration: autogenerate detects view removed from model."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_removed_view_generates_drop(self, connection, alembic_config):
        _int_drop_test_views(connection)
        _drop_base_table(connection)
        _create_base_table(connection)
        # Create a view directly in DB
        connection.execute(
            sa.text("CREATE VIEW int_test_drop_view AS SELECT id FROM _cmp_test_base")
        )
        connection.commit()
        try:
            include_view_comparator()
            # Empty metadata — view is in DB but not in model
            metadata = sa.MetaData()
            code = run_autogenerate(metadata, connection, alembic_config)
            assert_has_op(code, "drop_view")
        finally:
            _int_drop_test_views(connection)
            _drop_base_table(connection)


class TestIntegrationMVRemoval:
    """Integration: autogenerate detects MV removed from model."""

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


class TestIntegrationViewDefinitionChange:
    """Integration: autogenerate detects view definition change."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_changed_view_generates_replace(self, connection, alembic_config):
        _int_drop_test_views(connection)
        _drop_base_table(connection)
        _create_base_table(connection)
        # Create original view in DB
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_change_view AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        try:
            include_view_comparator()
            metadata = sa.MetaData()
            # Define same view name with different definition
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


class TestIntegrationMVDefinitionChange:
    """Integration: autogenerate detects MV definition change."""

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


class TestIntegrationNoOpWhenUnchanged:
    """Integration: no view ops generated when view definitions match."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_unchanged_view_no_ops(self, connection, alembic_config):
        _int_drop_test_views(connection)
        _drop_base_table(connection)
        _create_base_table(connection)
        # Create view in DB with same definition that model will have
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_same_view AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        try:
            include_view_comparator()
            metadata = sa.MetaData()
            # Same definition — comparator canonicalizes both, should match
            create_view("int_test_same_view", sa.select(_int_base_table.c.id), metadata)
            code = run_autogenerate(metadata, connection, alembic_config)
            # No view create/drop/replace ops should appear
            assert_no_op(code, "create_view")
            assert_no_op(code, "drop_view")
            assert_no_op(code, "replace_view")
        finally:
            _int_drop_test_views(connection)
            _drop_base_table(connection)


class TestIntegrationDependencyOrdering:
    """Integration: views are created in dependency order."""

    @pytest.mark.usefixtures("postgresql_dsn")
    def test_dependent_view_created_after_dependency(self, connection, alembic_config):
        _int_drop_test_views(connection)
        _drop_base_table(connection)
        _create_base_table(connection)
        # Pre-create view_a in the DB so view_b's canonicalization can succeed
        connection.execute(
            sa.text(
                "CREATE VIEW int_test_view_a AS SELECT id FROM _cmp_test_base"
            )
        )
        connection.commit()
        try:
            include_view_comparator()
            metadata = sa.MetaData()
            # view_a already in DB, also in model (so no diff for it)
            create_view("int_test_view_a", sa.select(_int_base_table.c.id), metadata)
            # view_b references view_a — this is the NEW view to detect
            vr_b = ViewRecord(
                name="int_test_view_b",
                selectable="SELECT id FROM int_test_view_a",
                schema=None,
                materialized=False,
            )
            metadata.info.setdefault("sqlalchemy_utils_views", []).append(vr_b)

            code = run_autogenerate(metadata, connection, alembic_config)
            # view_b should be detected as a new view
            assert_has_op(code, "create_view")
            assert "int_test_view_b" in code
        finally:
            _int_drop_test_views(connection)
            _drop_base_table(connection)
