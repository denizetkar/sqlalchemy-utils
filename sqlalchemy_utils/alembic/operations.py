"""
Alembic migration operations for database views.

Provides 7 MigrateOperation subclasses for creating, dropping, replacing,
and refreshing materialized views, along with ``op.*`` helper functions.

Usage in Alembic migrations::

    def upgrade():
        op.create_view("my_view", "SELECT id, name FROM users")
        op.create_materialized_view("mv_stats", "SELECT count(*) FROM events")

    def downgrade():
        op.drop_view("my_view")
        op.drop_materialized_view("mv_stats")
"""

from __future__ import annotations

import warnings

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


def _validate_definition(definition: str) -> None:
    if not isinstance(definition, str) or not definition:
        raise TypeError("definition must be a non-empty string")


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
        _validate_definition(definition)
        if replace:
            warnings.warn(
                "CreateViewOp(replace=True) is deprecated; use op.replace_view() "
                "or ReplaceViewOp instead. The reverse() of CreateViewOp(replace=True) "
                "emits a destructive DROP, while ReplaceViewOp.reverse() restores "
                "the prior definition.",
                DeprecationWarning,
                stacklevel=2,
            )
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
            replace=self.replace,
        )

    def to_diff_tuple(self) -> tuple:
        """Return ``("create_view", name, schema, definition)``."""
        return ("create_view", self.name, self.schema, self.definition)


@Operations.register_operation("drop_view")
class DropViewOp(MigrateOperation):
    """Operation that emits ``DROP [MATERIALIZED] VIEW IF EXISTS``."""

    def __init__(
        self,
        name: str,
        *,
        schema: str | None = None,
        # Internal-only: used by reverse() for round-trip. Autogenerate uses
        # DropMaterializedViewOp for MV drops; op.drop_view rejects
        # materialized=True. The renderer does not round-trip this flag.
        materialized: bool = False,
        # Note: named cascade for Alembic op consistency; corresponds to
        # cascade_on_drop in ViewMixin and create_view().
        cascade: bool = True,
        definition: str | None = None,
        replace: bool = False,
    ) -> None:
        self.name = name
        self.schema = schema
        self.materialized = materialized
        self.cascade = cascade
        self.definition = definition
        self.replace = replace

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
        """Programmatic entry-point for ``op.drop_view()``.

        .. note::
           This drops a regular (non-materialized) view.  Use
           ``op.drop_materialized_view()`` for materialized views.
        """
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
        otherwise a ``NotImplementedError`` is raised because the view definition
        is unknown.
        """
        if self.definition is None:
            raise NotImplementedError(
                f"Cannot reverse DropViewOp for '{self.name}': "
                "no definition stored. Pass definition= to DropViewOp "
                "to enable automatic downgrade generation."
            )
        return CreateViewOp(self.name, self.definition, schema=self.schema, replace=self.replace)

    def to_diff_tuple(self) -> tuple:
        """Return ``("drop_view", name, schema, definition)``."""
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
        _validate_definition(definition)
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

        Requires ``old_definition``; otherwise raises ``NotImplementedError``.
        """
        if self.old_definition is None:
            raise NotImplementedError(
                f"Cannot reverse ReplaceViewOp for '{self.name}': "
                "no old_definition stored. Pass old_definition= to "
                "ReplaceViewOp to enable automatic downgrade generation."
            )
        return ReplaceViewOp(
            self.name, self.old_definition, schema=self.schema,
            old_definition=self.definition,
        )

    def to_diff_tuple(self) -> tuple:
        """Return ``("replace_view", name, schema, definition, old_definition)``."""
        return ("replace_view", self.name, self.schema, self.definition, self.old_definition)


# ===================================================================
# Materialized view operations
# ===================================================================


@Operations.register_operation("create_materialized_view")
class CreateMaterializedViewOp(MigrateOperation):
    """Operation that emits ``CREATE MATERIALIZED VIEW``.

    .. note:: Materialized views are PostgreSQL-specific; other dialects
       will raise at execute time.

    .. note::
       Autogenerate always emits ``with_data=False`` (unpopulated MVs);
       manual ``op.create_materialized_view()`` also defaults to
       ``WITH NO DATA`` for consistency.
    """

    def __init__(
        self,
        name: str,
        definition: str,
        *,
        schema: str | None = None,
        with_data: bool = False,
    ) -> None:
        _validate_definition(definition)
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
        with_data: bool = False,
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
            with_data=self.with_data,
        )

    def to_diff_tuple(self) -> tuple:
        """Return ``("create_materialized_view", name, schema, definition, with_data)``."""
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
        # Note: named cascade for Alembic op consistency; corresponds to
        # cascade_on_drop in ViewMixin and create_view().
        cascade: bool = True,
        definition: str | None = None,
        with_data: bool = True,
    ) -> None:
        self.name = name
        self.schema = schema
        self.cascade = cascade
        self.definition = definition
        self.with_data = with_data

    @classmethod
    def drop_materialized_view(
        cls,
        operations: Operations,
        name: str,
        *,
        schema: str | None = None,
        cascade: bool = True,
        definition: str | None = None,
        with_data: bool = True,
    ) -> None:
        """Programmatic entry-point for ``op.drop_materialized_view()``."""
        op = DropMaterializedViewOp(
            name, schema=schema, cascade=cascade, definition=definition,
            with_data=with_data,
        )
        return operations.invoke(op)

    def reverse(self) -> CreateMaterializedViewOp:
        """Return the inverse: re-create the materialized view.

        Requires ``definition``; otherwise raises ``NotImplementedError``.
        """
        if self.definition is None:
            raise NotImplementedError(
                f"Cannot reverse DropMaterializedViewOp for '{self.name}': "
                "no definition stored. Pass definition= to "
                "DropMaterializedViewOp to enable automatic downgrade "
                "generation."
            )
        return CreateMaterializedViewOp(
            self.name, self.definition, schema=self.schema,
            with_data=self.with_data,
        )

    def to_diff_tuple(self) -> tuple:
        """Return ``("drop_materialized_view", name, schema, definition)``."""
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
        with_data: bool = False,
        old_definition: str | None = None,
    ) -> None:
        _validate_definition(definition)
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
        with_data: bool = False,
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

        Requires ``old_definition``; otherwise raises ``NotImplementedError``.
        """
        if self.old_definition is None:
            raise NotImplementedError(
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
            old_definition=self.definition,
        )

    def to_diff_tuple(self) -> tuple:
        """Return ``("replace_materialized_view", name, schema, definition, with_data, old_definition)``."""
        return (
            "replace_materialized_view",
            self.name,
            self.schema,
            self.definition,
            self.with_data,
            self.old_definition,
        )


# ===================================================================
# Refresh materialized view operation
# ===================================================================


@Operations.register_operation("refresh_materialized_view")
class RefreshMaterializedViewOp(MigrateOperation):
    """Operation that emits ``REFRESH MATERIALIZED VIEW``.

    .. note:: Materialized views are PostgreSQL-specific; other dialects
       will raise at execute time.
    """

    def __init__(
        self,
        name: str,
        *,
        schema: str | None = None,
        concurrently: bool = False,
    ) -> None:
        self.name = name
        self.schema = schema
        self.concurrently = concurrently

    @classmethod
    def refresh_materialized_view(
        cls,
        operations: Operations,
        name: str,
        *,
        schema: str | None = None,
        concurrently: bool = False,
    ) -> None:
        """Programmatic entry-point for ``op.refresh_materialized_view()``."""
        op = cls(name, schema=schema, concurrently=concurrently)
        return operations.invoke(op)

    def reverse(self) -> "RefreshMaterializedViewOp":
        """REFRESH MATERIALIZED VIEW is not meaningfully reversible.

        You cannot "un-refresh" a materialized view, so reverse()
        refuses rather than silently emit another REFRESH in the
        downgrade. Implement the downgrade step manually if needed.
        """
        raise NotImplementedError(
            "REFRESH MATERIALIZED VIEW is not meaningfully reversible; "
            "remove it from downgrade() or implement manually."
        )

    def to_diff_tuple(self) -> tuple:
        """Return ``("refresh_materialized_view", name, schema)``."""
        return ("refresh_materialized_view", self.name, self.schema)


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


@Operations.implementation_for(RefreshMaterializedViewOp)
def _refresh_materialized_view_impl(
    operations: Operations, op: RefreshMaterializedViewOp
) -> None:
    """Execute ``REFRESH MATERIALIZED VIEW`` via the migration connection."""
    dialect = operations.get_bind().dialect
    qualified = _quote_qualified_name(dialect, op.name, op.schema)
    concurrently = "CONCURRENTLY " if op.concurrently else ""
    sql = f"REFRESH MATERIALIZED VIEW {concurrently}{qualified}"
    operations.execute(sa.text(sql))
