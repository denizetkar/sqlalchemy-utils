"""
Alembic migration operations for database views.

Provides 6 MigrateOperation subclasses for creating, dropping, and replacing
both regular and materialized views, along with ``op.*`` helper functions.

Usage in Alembic migrations::

    def upgrade():
        op.create_view("my_view", "SELECT id, name FROM users")
        op.create_materialized_view("mv_stats", "SELECT count(*) FROM events")

    def downgrade():
        op.drop_view("my_view")
        op.drop_materialized_view("mv_stats")
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic.operations import MigrateOperation, Operations


# ---------------------------------------------------------------------------
# Helpers (identifier quoting — see view.py for the same pattern)
# ---------------------------------------------------------------------------

def _quote_identifier(dialect, name: str) -> str:
    """Quote *name* using the dialect's identifier preparer."""
    return dialect.identifier_preparer.quote(name)


def _quote_qualified_name(dialect, name: str, schema: str | None) -> str:
    """Return a schema-qualified, properly quoted identifier.

    When *schema* is given the result is ``"schema"."name"`` (both parts
    quoted by the dialect's identifier preparer); otherwise just ``"name"``.
    """
    if schema:
        return f"{_quote_identifier(dialect, schema)}.{_quote_identifier(dialect, name)}"
    return _quote_identifier(dialect, name)


# ===================================================================
# Regular view operations
# ===================================================================


@Operations.register_operation("create_view")
class CreateViewOp(MigrateOperation):
    """Operation that emits ``CREATE [OR REPLACE] VIEW``."""

    def __init__(
        self,
        name: str,
        definition: str,
        *,
        schema: str | None = None,
        replace: bool = False,
    ) -> None:
        self.name = name
        self.definition = definition
        self.schema = schema
        self.replace = replace

    @classmethod
    def create_view(
        cls,
        operations: Operations,
        name: str,
        definition: str,
        *,
        replace: bool = False,
        schema: str | None = None,
    ) -> None:
        """Programmatic entry-point for ``op.create_view()``."""
        op = CreateViewOp(name, definition, schema=schema, replace=replace)
        return operations.invoke(op)

    def reverse(self) -> DropViewOp:
        """Return the inverse: drop the view that was just created."""
        return DropViewOp(
            self.name,
            schema=self.schema,
            materialized=False,
            cascade=True,
            definition=self.definition,
        )

    def to_diff_tuple(self) -> tuple:
        return ("create_view", self.name, self.schema, self.definition)


@Operations.register_operation("drop_view")
class DropViewOp(MigrateOperation):
    """Operation that emits ``DROP [MATERIALIZED] VIEW IF EXISTS``."""

    def __init__(
        self,
        name: str,
        *,
        schema: str | None = None,
        materialized: bool = False,
        cascade: bool = True,
        definition: str | None = None,
    ) -> None:
        self.name = name
        self.schema = schema
        self.materialized = materialized
        self.cascade = cascade
        self.definition = definition

    @classmethod
    def drop_view(
        cls,
        operations: Operations,
        name: str,
        *,
        schema: str | None = None,
        cascade: bool = True,
        definition: str | None = None,
    ) -> None:
        """Programmatic entry-point for ``op.drop_view()``."""
        op = DropViewOp(
            name,
            schema=schema,
            materialized=False,
            cascade=cascade,
            definition=definition,
        )
        return operations.invoke(op)

    def reverse(self) -> CreateViewOp:
        """Return the inverse: re-create the view that was just dropped.

        Requires ``definition`` to have been stored at construction time;
        otherwise a ``RuntimeError`` is raised because the view definition
        is unknown.
        """
        if self.definition is None:
            raise RuntimeError(
                f"Cannot reverse DropViewOp for '{self.name}': "
                "no definition stored. Pass definition= to DropViewOp "
                "to enable automatic downgrade generation."
            )
        return CreateViewOp(self.name, self.definition, schema=self.schema)

    def to_diff_tuple(self) -> tuple:
        return ("drop_view", self.name, self.schema, self.definition)


@Operations.register_operation("replace_view")
class ReplaceViewOp(MigrateOperation):
    """Operation that emits ``CREATE OR REPLACE VIEW``."""

    def __init__(
        self,
        name: str,
        definition: str,
        *,
        schema: str | None = None,
        old_definition: str | None = None,
    ) -> None:
        self.name = name
        self.definition = definition
        self.schema = schema
        self.old_definition = old_definition

    @classmethod
    def replace_view(
        cls,
        operations: Operations,
        name: str,
        definition: str,
        *,
        schema: str | None = None,
        old_definition: str | None = None,
    ) -> None:
        """Programmatic entry-point for ``op.replace_view()``."""
        op = ReplaceViewOp(
            name, definition, schema=schema, old_definition=old_definition
        )
        return operations.invoke(op)

    def reverse(self) -> ReplaceViewOp:
        """Return the inverse: replace back to the old definition.

        Requires ``old_definition``; otherwise raises ``RuntimeError``.
        """
        if self.old_definition is None:
            raise RuntimeError(
                f"Cannot reverse ReplaceViewOp for '{self.name}': "
                "no old_definition stored. Pass old_definition= to "
                "ReplaceViewOp to enable automatic downgrade generation."
            )
        return ReplaceViewOp(
            self.name, self.old_definition, schema=self.schema
        )

    def to_diff_tuple(self) -> tuple:
        return ("replace_view", self.name, self.schema, self.definition, self.old_definition)


# ===================================================================
# Materialized view operations
# ===================================================================


@Operations.register_operation("create_materialized_view")
class CreateMaterializedViewOp(MigrateOperation):
    """Operation that emits ``CREATE MATERIALIZED VIEW``.

    .. note:: Materialized views are PostgreSQL-specific; other dialects
       will raise at execute time.
    """

    def __init__(
        self,
        name: str,
        definition: str,
        *,
        schema: str | None = None,
        with_data: bool = True,
    ) -> None:
        self.name = name
        self.definition = definition
        self.schema = schema
        self.with_data = with_data

    @classmethod
    def create_materialized_view(
        cls,
        operations: Operations,
        name: str,
        definition: str,
        *,
        schema: str | None = None,
        with_data: bool = True,
    ) -> None:
        """Programmatic entry-point for ``op.create_materialized_view()``."""
        op = CreateMaterializedViewOp(
            name, definition, schema=schema, with_data=with_data
        )
        return operations.invoke(op)

    def reverse(self) -> DropMaterializedViewOp:
        """Return the inverse: drop the materialized view just created."""
        return DropMaterializedViewOp(
            self.name,
            schema=self.schema,
            cascade=True,
            definition=self.definition,
        )

    def to_diff_tuple(self) -> tuple:
        return ("create_materialized_view", self.name, self.schema, self.definition, self.with_data)


@Operations.register_operation("drop_materialized_view")
class DropMaterializedViewOp(MigrateOperation):
    """Operation that emits ``DROP MATERIALIZED VIEW IF EXISTS``.

    .. note:: Materialized views are PostgreSQL-specific; other dialects
       will raise at execute time.
    """

    def __init__(
        self,
        name: str,
        *,
        schema: str | None = None,
        cascade: bool = True,
        definition: str | None = None,
    ) -> None:
        self.name = name
        self.schema = schema
        self.cascade = cascade
        self.definition = definition

    @classmethod
    def drop_materialized_view(
        cls,
        operations: Operations,
        name: str,
        *,
        schema: str | None = None,
        cascade: bool = True,
        definition: str | None = None,
    ) -> None:
        """Programmatic entry-point for ``op.drop_materialized_view()``."""
        op = DropMaterializedViewOp(
            name, schema=schema, cascade=cascade, definition=definition
        )
        return operations.invoke(op)

    def reverse(self) -> CreateMaterializedViewOp:
        """Return the inverse: re-create the materialized view.

        Requires ``definition``; otherwise raises ``RuntimeError``.
        """
        if self.definition is None:
            raise RuntimeError(
                f"Cannot reverse DropMaterializedViewOp for '{self.name}': "
                "no definition stored. Pass definition= to "
                "DropMaterializedViewOp to enable automatic downgrade "
                "generation."
            )
        return CreateMaterializedViewOp(
            self.name, self.definition, schema=self.schema
        )

    def to_diff_tuple(self) -> tuple:
        return ("drop_materialized_view", self.name, self.schema, self.definition)


@Operations.register_operation("replace_materialized_view")
class ReplaceMaterializedViewOp(MigrateOperation):
    """Operation that drops and re-creates a materialized view.

    PostgreSQL does not support ``CREATE OR REPLACE MATERIALIZED VIEW``
    so this operation issues a ``DROP`` followed by a ``CREATE``.

    .. note:: Materialized views are PostgreSQL-specific; other dialects
       will raise at execute time.
    """

    def __init__(
        self,
        name: str,
        definition: str,
        *,
        schema: str | None = None,
        with_data: bool = True,
        old_definition: str | None = None,
    ) -> None:
        self.name = name
        self.definition = definition
        self.schema = schema
        self.with_data = with_data
        self.old_definition = old_definition

    @classmethod
    def replace_materialized_view(
        cls,
        operations: Operations,
        name: str,
        definition: str,
        *,
        schema: str | None = None,
        with_data: bool = True,
        old_definition: str | None = None,
    ) -> None:
        """Programmatic entry-point for ``op.replace_materialized_view()``."""
        op = ReplaceMaterializedViewOp(
            name,
            definition,
            schema=schema,
            with_data=with_data,
            old_definition=old_definition,
        )
        return operations.invoke(op)

    def reverse(self) -> ReplaceMaterializedViewOp:
        """Return the inverse: replace back to old definition.

        Requires ``old_definition``; otherwise raises ``RuntimeError``.
        """
        if self.old_definition is None:
            raise RuntimeError(
                f"Cannot reverse ReplaceMaterializedViewOp for "
                f"'{self.name}': no old_definition stored. Pass "
                "old_definition= to ReplaceMaterializedViewOp to enable "
                "automatic downgrade generation."
            )
        return ReplaceMaterializedViewOp(
            self.name,
            self.old_definition,
            schema=self.schema,
            with_data=self.with_data,
        )

    def to_diff_tuple(self) -> tuple:
        return (
            "replace_materialized_view",
            self.name,
            self.schema,
            self.definition,
            self.with_data,
            self.old_definition,
        )


# ===================================================================
# SQL implementations
# ===================================================================


@Operations.implementation_for(CreateViewOp)
def _create_view_impl(operations: Operations, op: CreateViewOp) -> None:
    """Execute ``CREATE [OR REPLACE] VIEW`` via the migration connection."""
    dialect = operations.get_bind().dialect
    qualified = _quote_qualified_name(dialect, op.name, op.schema)
    replace_clause = "OR REPLACE " if op.replace else ""
    sql = f"CREATE {replace_clause}VIEW {qualified} AS {op.definition}"
    operations.execute(sa.text(sql))


@Operations.implementation_for(DropViewOp)
def _drop_view_impl(operations: Operations, op: DropViewOp) -> None:
    """Execute ``DROP [MATERIALIZED] VIEW IF EXISTS`` via the migration connection."""
    dialect = operations.get_bind().dialect
    qualified = _quote_qualified_name(dialect, op.name, op.schema)
    mat_clause = "MATERIALIZED " if op.materialized else ""
    cascade_clause = " CASCADE" if op.cascade else ""
    sql = f"DROP {mat_clause}VIEW IF EXISTS {qualified}{cascade_clause}"
    operations.execute(sa.text(sql))


@Operations.implementation_for(ReplaceViewOp)
def _replace_view_impl(operations: Operations, op: ReplaceViewOp) -> None:
    """Execute ``CREATE OR REPLACE VIEW`` via the migration connection."""
    dialect = operations.get_bind().dialect
    qualified = _quote_qualified_name(dialect, op.name, op.schema)
    sql = f"CREATE OR REPLACE VIEW {qualified} AS {op.definition}"
    operations.execute(sa.text(sql))


@Operations.implementation_for(CreateMaterializedViewOp)
def _create_materialized_view_impl(
    operations: Operations, op: CreateMaterializedViewOp
) -> None:
    """Execute ``CREATE MATERIALIZED VIEW … WITH [NO] DATA``."""
    dialect = operations.get_bind().dialect
    qualified = _quote_qualified_name(dialect, op.name, op.schema)
    data_clause = "WITH DATA" if op.with_data else "WITH NO DATA"
    sql = (
        f"CREATE MATERIALIZED VIEW {qualified} AS {op.definition} "
        f"{data_clause}"
    )
    operations.execute(sa.text(sql))


@Operations.implementation_for(DropMaterializedViewOp)
def _drop_materialized_view_impl(
    operations: Operations, op: DropMaterializedViewOp
) -> None:
    """Execute ``DROP MATERIALIZED VIEW IF EXISTS`` via the migration connection."""
    dialect = operations.get_bind().dialect
    qualified = _quote_qualified_name(dialect, op.name, op.schema)
    cascade_clause = " CASCADE" if op.cascade else ""
    sql = f"DROP MATERIALIZED VIEW IF EXISTS {qualified}{cascade_clause}"
    operations.execute(sa.text(sql))


@Operations.implementation_for(ReplaceMaterializedViewOp)
def _replace_materialized_view_impl(
    operations: Operations, op: ReplaceMaterializedViewOp
) -> None:
    """Drop then re-create a materialized view (PG has no OR REPLACE for MVs)."""
    dialect = operations.get_bind().dialect
    qualified = _quote_qualified_name(dialect, op.name, op.schema)
    data_clause = "WITH DATA" if op.with_data else "WITH NO DATA"

    drop_sql = f"DROP MATERIALIZED VIEW IF EXISTS {qualified} CASCADE"
    create_sql = (
        f"CREATE MATERIALIZED VIEW {qualified} AS {op.definition} "
        f"{data_clause}"
    )
    operations.execute(sa.text(drop_sql))
    operations.execute(sa.text(create_sql))
