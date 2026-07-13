"""Alembic autogenerate integration for SQL views and materialized views.

Provides migration operations (create, drop, replace, refresh) and an
autogenerate comparator that detects view changes between the model
metadata and the database.

To enable autogenerate for views, call :func:`register_view_comparator`
in your Alembic ``env.py`` before ``context.configure()``.

Dependency-resolution and PostgreSQL catalog helpers are also importable
from their submodules for advanced use; see the View Migrations guide.
"""

from .comparator import register_view_comparator
from .depend import resolve_create_order, resolve_drop_order
from .operations import (CreateViewOp, DropViewOp, ReplaceViewOp,
                           CreateMaterializedViewOp, DropMaterializedViewOp,
                           ReplaceMaterializedViewOp, RefreshMaterializedViewOp)
from .pg_catalog import get_database_views, get_database_materialized_views, get_dependent_views
from ..view_record import ViewRecord

__all__ = [
    "register_view_comparator",
    "CreateViewOp", "DropViewOp", "ReplaceViewOp",
    "CreateMaterializedViewOp", "DropMaterializedViewOp", "ReplaceMaterializedViewOp",
    "RefreshMaterializedViewOp",
    "ViewRecord",
]
