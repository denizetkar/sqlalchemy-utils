from __future__ import annotations

from dataclasses import dataclass

import sqlalchemy as sa


@dataclass(frozen=True)
class ViewRecord:
    """
    Frozen dataclass representing a view definition for Alembic migrations.

    Mirrors the parameters of :func:`~sqlalchemy_utils.view.create_view`
    and :func:`~sqlalchemy_utils.view.create_materialized_view`
    to enable reconstruction of view definitions from serialized data.

    Equality is name-based: two ViewRecords with the same ``name`` and
    ``schema`` (and ``materialized`` flag) compare equal, regardless of the
    underlying selectable SQL.  Use :meth:`~ViewRecord.compiled_definition`
    to compare the actual SQL definitions.

    Users typically do not construct ``ViewRecord`` directly: instances are
    populated automatically by :func:`~sqlalchemy_utils.view.create_view`,
    :func:`~sqlalchemy_utils.view.create_materialized_view`, and
    :class:`~sqlalchemy_utils.view_mixin.ViewMixin` into
    ``metadata.info['sqlalchemy_utils_views']``, which the Alembic
    autogenerate comparator reads during ``alembic revision --autogenerate``.

    :param name: Name of the view.
    :param selectable: SQLAlchemy selectable (e.g. ``select()``) defining the view body.
    :param schema: Optional schema name; ``None`` means the default schema.
    :param materialized: When ``True``, this record describes a materialized view.
    :param replace: When ``True``, runtime DDL emits ``CREATE OR REPLACE VIEW``.
    :param cascade_on_drop: When ``True`` (default), appends ``CASCADE`` to
        ``DROP VIEW``/``DROP MATERIALIZED VIEW``.
    :param aliases: Optional dict mapping column names to alternative keys.

    :raises TypeError: if *selectable* is None.

    Example::

        import sqlalchemy as sa
        from sqlalchemy_utils.view_record import ViewRecord

        selectable = sa.select(sa.column("id", sa.Integer))
        record = ViewRecord(
            name="my_view",
            selectable=selectable,
            schema="public",
            materialized=False,
        )
        # `record.compiled_definition()` returns the compiled SQL string.
    """
    name: str
    selectable: sa.sql.ClauseElement
    schema: str | None = None
    materialized: bool = False
    replace: bool = False
    cascade_on_drop: bool = True
    aliases: dict[str, str] | None = None

    def __post_init__(self):
        if self.selectable is None:
            raise TypeError("selectable must not be None")
        # Normalize falsy schema (e.g. "") to None: "" would create the
        # view in current_schema() but fail the schema-match check
        # ("" != None), yielding a false DropViewOp.
        object.__setattr__(self, "schema", self.schema or None)

    def __eq__(self, other: object) -> bool:
        """Compare two ViewRecords for equality.

        Intentionally name-based (name/schema/materialized only) so that
        ViewRecords can serve as stable dict/set keys even when the
        underlying selectable SQL changes. Use :meth:`~ViewRecord.compiled_definition`
        to detect actual definition (selectable) changes.
        """
        if not isinstance(other, ViewRecord):
            return NotImplemented
        return (
            self.name == other.name
            and self.schema == other.schema
            and self.materialized == other.materialized
        )

    def __hash__(self) -> int:
        """Hash the ViewRecord for use in sets and dicts."""
        return hash((self.name, self.schema, self.materialized))

    def compiled_definition(self, *, dialect: sa.engine.Dialect | None = None) -> str:
        """Compile the selectable to a SQL string for comparison/dependency detection.

        If *dialect* is provided, compile against it; otherwise use default
        compilation.  String selectables are returned as-is.

        This is the single source of truth for selectable-to-string
        compilation used by
        ``sqlalchemy_utils.alembic.comparator._build_create_sql``
        and ``sqlalchemy_utils.alembic.depend._build_dependency_graph``.
        """
        sel = self.selectable
        if isinstance(sel, str):
            return sel
        compile_kwargs = {"literal_binds": True}
        if dialect is not None:
            return str(
                sel.compile(dialect=dialect, compile_kwargs=compile_kwargs)
            )
        return str(sel.compile(compile_kwargs=compile_kwargs))

    def __repr__(self) -> str:
        """Pretty string representation."""
        schema_str = f"{self.schema!r}" if self.schema else "None"
        return f"ViewRecord(name={self.name!r}, schema={schema_str}, materialized={self.materialized})"
