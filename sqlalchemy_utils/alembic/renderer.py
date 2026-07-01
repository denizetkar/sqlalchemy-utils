"""
Alembic autogenerate renderers for view migration operations.

Registers 6 renderer functions with ``alembic.autogenerate.renderers.dispatch_for``
so that Alembic's autogenerate can render view operations as Python code strings
in migration scripts.
"""

from __future__ import annotations

from alembic.autogenerate import renderers
from alembic.autogenerate.api import AutogenContext

from sqlalchemy_utils.alembic.operations import (
    CreateViewOp,
    DropViewOp,
    ReplaceViewOp,
    CreateMaterializedViewOp,
    DropMaterializedViewOp,
    ReplaceMaterializedViewOp,
    RefreshMaterializedViewOp,
)


@renderers.dispatch_for(CreateViewOp)
def render_create_view(autogen_context: AutogenContext, op: CreateViewOp) -> str:
    """Render a CreateViewOp as op.create_view(...) code."""
    schema_part = f", schema={op.schema!r}" if op.schema is not None else ""
    replace_part = ", replace=True" if op.replace else ""
    return f"op.create_view({op.name!r}, {op.definition!r}{schema_part}{replace_part})"


@renderers.dispatch_for(DropViewOp)
def render_drop_view(autogen_context: AutogenContext, op: DropViewOp) -> str:
    """Render a DropViewOp as op.drop_view(...) code."""
    # materialized flag intentionally not rendered: DropViewOp is for
    # regular views only; MV drops use DropMaterializedViewOp, and
    # op.drop_view rejects materialized=True.
    schema_part = f", schema={op.schema!r}" if op.schema is not None else ""
    cascade_part = "" if op.cascade else ", cascade=False"
    definition_part = f", definition={op.definition!r}" if op.definition is not None else ""
    return f"op.drop_view({op.name!r}{schema_part}{cascade_part}{definition_part})"


@renderers.dispatch_for(ReplaceViewOp)
def render_replace_view(autogen_context: AutogenContext, op: ReplaceViewOp) -> str:
    """Render a ReplaceViewOp as op.replace_view(...) code."""
    schema_part = f", schema={op.schema!r}" if op.schema is not None else ""
    old_def_part = f", old_definition={op.old_definition!r}" if op.old_definition is not None else ""
    return f"op.replace_view({op.name!r}, {op.definition!r}{schema_part}{old_def_part})"


@renderers.dispatch_for(CreateMaterializedViewOp)
def render_create_materialized_view(
    autogen_context: AutogenContext, op: CreateMaterializedViewOp
) -> str:
    """Render a CreateMaterializedViewOp as op.create_materialized_view(...) code."""
    schema_part = f", schema={op.schema!r}" if op.schema is not None else ""
    with_data_part = "" if not op.with_data else ", with_data=True"
    return f"op.create_materialized_view({op.name!r}, {op.definition!r}{schema_part}{with_data_part})"


@renderers.dispatch_for(DropMaterializedViewOp)
def render_drop_materialized_view(
    autogen_context: AutogenContext, op: DropMaterializedViewOp
) -> str:
    """Render a DropMaterializedViewOp as op.drop_materialized_view(...) code."""
    schema_part = f", schema={op.schema!r}" if op.schema is not None else ""
    cascade_part = "" if op.cascade else ", cascade=False"
    definition_part = f", definition={op.definition!r}" if op.definition is not None else ""
    return f"op.drop_materialized_view({op.name!r}{schema_part}{cascade_part}{definition_part})"


@renderers.dispatch_for(ReplaceMaterializedViewOp)
def render_replace_materialized_view(
    autogen_context: AutogenContext, op: ReplaceMaterializedViewOp
) -> str:
    """Render a ReplaceMaterializedViewOp as op.replace_materialized_view(...) code."""
    schema_part = f", schema={op.schema!r}" if op.schema is not None else ""
    with_data_part = "" if not op.with_data else ", with_data=True"
    old_def_part = f", old_definition={op.old_definition!r}" if op.old_definition is not None else ""
    return f"op.replace_materialized_view({op.name!r}, {op.definition!r}{schema_part}{with_data_part}{old_def_part})"


@renderers.dispatch_for(RefreshMaterializedViewOp)
def render_refresh_materialized_view(
    autogen_context: AutogenContext, op: RefreshMaterializedViewOp
) -> str:
    """Render a RefreshMaterializedViewOp as op.refresh_materialized_view(...) code."""
    schema_part = f", schema={op.schema!r}" if op.schema is not None else ""
    concurrently_part = ", concurrently=True" if op.concurrently else ""
    return f"op.refresh_materialized_view({op.name!r}{schema_part}{concurrently_part})"
