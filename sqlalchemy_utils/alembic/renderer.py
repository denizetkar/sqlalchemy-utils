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
)


@renderers.dispatch_for(CreateViewOp)
def render_create_view(autogen_context: AutogenContext, op: CreateViewOp) -> str:
    schema_part = f", schema={op.schema!r}" if op.schema else ""
    return f"op.create_view({op.name!r}, {op.definition!r}{schema_part})"


@renderers.dispatch_for(DropViewOp)
def render_drop_view(autogen_context: AutogenContext, op: DropViewOp) -> str:
    schema_part = f", schema={op.schema!r}" if op.schema else ""
    cascade_part = ", cascade=True" if op.cascade else ""
    return f"op.drop_view({op.name!r}{schema_part}{cascade_part})"


@renderers.dispatch_for(ReplaceViewOp)
def render_replace_view(autogen_context: AutogenContext, op: ReplaceViewOp) -> str:
    schema_part = f", schema={op.schema!r}" if op.schema else ""
    return f"op.replace_view({op.name!r}, {op.definition!r}{schema_part})"


@renderers.dispatch_for(CreateMaterializedViewOp)
def render_create_materialized_view(
    autogen_context: AutogenContext, op: CreateMaterializedViewOp
) -> str:
    schema_part = f", schema={op.schema!r}" if op.schema else ""
    with_data_part = "" if op.with_data else ", with_data=False"
    return f"op.create_materialized_view({op.name!r}, {op.definition!r}{schema_part}{with_data_part})"


@renderers.dispatch_for(DropMaterializedViewOp)
def render_drop_materialized_view(
    autogen_context: AutogenContext, op: DropMaterializedViewOp
) -> str:
    schema_part = f", schema={op.schema!r}" if op.schema else ""
    cascade_part = ", cascade=False" if not op.cascade else ""
    return f"op.drop_materialized_view({op.name!r}{schema_part}{cascade_part})"


@renderers.dispatch_for(ReplaceMaterializedViewOp)
def render_replace_materialized_view(
    autogen_context: AutogenContext, op: ReplaceMaterializedViewOp
) -> str:
    schema_part = f", schema={op.schema!r}" if op.schema else ""
    with_data_part = ", with_data=True" if op.with_data else ""
    return f"op.replace_materialized_view({op.name!r}, {op.definition!r}{schema_part}{with_data_part})"
