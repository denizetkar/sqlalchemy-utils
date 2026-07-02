from __future__ import annotations

from typing import Any
from dataclasses import dataclass


@dataclass(frozen=True)
class ViewRecord:
    """
    Frozen dataclass representing a view definition for Alembic migrations.

    Mirrors the parameters of create_view() and CreateView.__init__
    to enable reconstruction of view definitions from serialized data.

    Equality is name-based: two ViewRecords with the same ``name`` and
    ``schema`` (and ``materialized`` flag) compare equal, regardless of the
    underlying selectable SQL.  Use :meth:`definition_matches` to compare the
    actual SQL definitions.
    """
    name: str
    selectable: Any
    schema: str | None = None
    materialized: bool = False
    replace: bool = False
    cascade_on_drop: bool = True

    def __post_init__(self):
        if self.selectable is None:
            raise TypeError("selectable must not be None")

    def __eq__(self, other: object) -> bool:
        """Compare two ViewRecords for equality.

        Intentionally name-based (name/schema/materialized only) so that
        ViewRecords can serve as stable dict/set keys even when the
        underlying selectable SQL changes. Use :meth:`definition_matches`
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

    def definition_matches(self, other: 'ViewRecord') -> bool:
        """Compare view definitions (selectable SQL) for equality.

        Unlike __eq__ (which is name-based for dict/set key usage), this method
        compares the actual SQL of the selectables to detect definition changes.
        """
        if not isinstance(other, ViewRecord):
            return NotImplemented
        return self.compiled_definition() == other.compiled_definition()

    def compiled_definition(self, dialect=None) -> str:
        """Compile the selectable to a SQL string for comparison/dependency detection.

        If *dialect* is provided, compile against it; otherwise use default
        compilation.  String selectables are returned as-is.

        This is the single source of truth for selectable-to-string
        compilation used by :meth:`definition_matches`,
        :func:`sqlalchemy_utils.alembic.comparator._compile_selectable`
        and :func:`sqlalchemy_utils.alembic.depend._definition_str`.
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

    def _selectable_key(self) -> str:
        """Backward-compatible alias for :meth:`compiled_definition`."""
        return self.compiled_definition()

    def __repr__(self) -> str:
        """Pretty string representation."""
        schema_str = f"{self.schema!r}" if self.schema else "None"
        return f"ViewRecord(name={self.name!r}, schema={schema_str}, materialized={self.materialized})"
