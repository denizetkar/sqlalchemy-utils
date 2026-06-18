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

    @classmethod
    def from_create_view(cls, create_view: 'CreateView', schema: str | None = None, cascade_on_drop: bool = True) -> 'ViewRecord':
        """
        Create a ViewRecord from a CreateView instance.

        CreateView only stores name, selectable, materialized, and replace.
        The schema and cascade_on_drop must be passed separately since they
        are parameters of the create_view() function, not the CreateView DDLElement.
        """
        return cls(
            name=create_view.name,
            selectable=create_view.selectable,
            schema=schema,
            materialized=create_view.materialized,
            replace=create_view.replace,
            cascade_on_drop=cascade_on_drop,
        )

    def __eq__(self, other: object) -> bool:
        """Compare two ViewRecords for equality."""
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

    def __repr__(self) -> str:
        """Pretty string representation."""
        schema_str = f"{self.schema!r}" if self.schema else "None"
        return f"ViewRecord(name={self.name!r}, schema={schema_str}, materialized={self.materialized})"
