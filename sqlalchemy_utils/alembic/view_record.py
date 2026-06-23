from __future__ import annotations

from typing import Any
from dataclasses import dataclass

try:
    from dataclasses import FrozenInstanceError
except ImportError:
    FrozenInstanceError = TypeError


@dataclass(frozen=True)
class ViewRecord:
    """
    Frozen dataclass representing a view definition for Alembic migrations.

    Mirrors the parameters of create_view() and CreateView.__init__
    to enable reconstruction of view definitions from serialized data.
    """
    name: str
    selectable: Any
    schema: str | None = None
    materialized: bool = False
    replace: bool = False
    cascade_on_drop: bool = True

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
        return self._selectable_key() == other._selectable_key()

    def _selectable_key(self) -> str:
        """Render the selectable to a stable string for comparison."""
        sel = self.selectable
        if isinstance(sel, str):
            return sel
        return str(sel.compile(compile_kwargs={"literal_binds": True}))

    def __repr__(self) -> str:
        """Pretty string representation."""
        schema_str = f"{self.schema!r}" if self.schema else "None"
        return f"ViewRecord(name={self.name!r}, schema={schema_str}, materialized={self.materialized})"
