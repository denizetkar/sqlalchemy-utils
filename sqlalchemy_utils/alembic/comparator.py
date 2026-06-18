"""
Alembic autogenerate comparator for database views.

Compares model-defined views (stored in ``metadata.info['sqlalchemy_utils_views']``)
against the current database state using **savepoint canonicalization**: each model
view is temporarily created inside a savepoint, its definition read back from
``pg_views``/``pg_matviews``, and the savepoint is rolled back so nothing persists.

Differences are emitted as :class:`CreateViewOp`, :class:`DropViewOp`,
:class:`ReplaceViewOp` (and their materialized-view equivalents).

Usage in ``env.py``::

    from sqlalchemy_utils.alembic.comparator import include_view_comparator
    include_view_comparator()   # must be called before context.configure()
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic.autogenerate import comparators
from alembic.autogenerate.api import AutogenContext

from sqlalchemy_utils.alembic.view_record import ViewRecord
from sqlalchemy_utils.alembic.pg_catalog import (
    get_database_views,
    get_database_materialized_views,
)
from sqlalchemy_utils.alembic.operations import (
    CreateViewOp,
    DropViewOp,
    ReplaceViewOp,
    CreateMaterializedViewOp,
    DropMaterializedViewOp,
    ReplaceMaterializedViewOp,
)
from sqlalchemy_utils.alembic.depend import resolve_create_order, resolve_drop_order

log = logging.getLogger(__name__)


def _canonicalize_view(
    connection: sa.engine.Connection,
    view_record: ViewRecord,
) -> str | None:
    """Create a view inside a savepoint, read its definition from pg_catalog,
    then roll back so the view never persists.

    Returns the canonical SQL definition string from PostgreSQL, or ``None``
    if the view cannot be created (e.g. depends on missing tables).
    """
    savepoint_name = "su_view_cmp"
    try:
        connection.execute(sa.text(f"SAVEPOINT {savepoint_name}"))

        sel = view_record.selectable
        if not isinstance(sel, str):
            definition = str(sel.compile(compile_kwargs={"literal_binds": True}))
        else:
            definition = sel

        prefix = f"{view_record.schema}." if view_record.schema else ""
        if view_record.materialized:
            # For MVs, if one already exists we must drop it first
            # (PG has no CREATE OR REPLACE MATERIALIZED VIEW)
            connection.execute(sa.text(
                f"DROP MATERIALIZED VIEW IF EXISTS {prefix}{view_record.name} CASCADE"
            ))
            sql = (
                f"CREATE MATERIALIZED VIEW {prefix}{view_record.name} "
                f"AS {definition} WITH NO DATA"
            )
        else:
            # Use OR REPLACE to avoid DuplicateTable if view already exists in DB
            sql = (
                f"CREATE OR REPLACE VIEW {prefix}{view_record.name} "
                f"AS {definition}"
            )

        connection.execute(sa.text(sql))

        if view_record.materialized:
            db_mvs = get_database_materialized_views(connection, view_record.schema)
            canonical = db_mvs.get(view_record.name)
        else:
            db_views = get_database_views(connection, view_record.schema)
            canonical = db_views.get(view_record.name)

        connection.execute(sa.text(f"ROLLBACK TO SAVEPOINT {savepoint_name}"))
        return canonical

    except Exception as exc:
        log.warning(
            "Failed to canonicalize view '%s': %s",
            view_record.name,
            exc,
        )
        try:
            connection.execute(sa.text(f"ROLLBACK TO SAVEPOINT {savepoint_name}"))
        except Exception:
            pass
        return None


@comparators.dispatch_for("schema")
def compare_views(
    autogen_context: AutogenContext,
    upgrade_ops,
    schemas,
) -> None:
    """Compare model-defined views against database state.

    This function is registered as an Alembic ``"schema"`` comparator and is
    called automatically during ``alembic revision --autogenerate``.

    It reads view definitions from ``metadata.info['sqlalchemy_utils_views']``
    (populated by the auto-registration mechanism in Task 6), canonicalizes
    each model view via savepoint simulation, and diffs against the live
    database.
    """
    connection = autogen_context.connection
    metadata = autogen_context.metadata

    model_records: list[ViewRecord] = metadata.info.get(
        "sqlalchemy_utils_views", []
    )

    for schema in schemas:
        db_views = get_database_views(connection, schema)
        db_mvs = get_database_materialized_views(connection, schema)

        # Canonicalize model views using savepoint simulation
        model_view_defs: dict[str, str] = {}
        model_mv_defs: dict[str, str] = {}

        for vr in model_records:
            if not _schema_matches(vr.schema, schema):
                continue

            canonical = _canonicalize_view(connection, vr)
            if canonical is not None:
                if vr.materialized:
                    model_mv_defs[vr.name] = canonical
                else:
                    model_view_defs[vr.name] = canonical

        # Diff model vs DB
        create_ops: list = []
        drop_ops: list = []

        # Regular views
        for name, definition in model_view_defs.items():
            if name not in db_views:
                create_ops.append(
                    CreateViewOp(name, definition, schema=schema)
                )
            elif db_views[name].strip() != definition.strip():
                create_ops.append(
                    ReplaceViewOp(
                        name,
                        definition,
                        schema=schema,
                        old_definition=db_views[name],
                    )
                )

        for name in db_views:
            if name not in model_view_defs:
                drop_ops.append(
                    DropViewOp(
                        name,
                        schema=schema,
                        materialized=False,
                        definition=db_views[name],
                    )
                )

        # Materialized views
        for name, definition in model_mv_defs.items():
            if name not in db_mvs:
                create_ops.append(
                    CreateMaterializedViewOp(
                        name, definition, schema=schema, with_data=False
                    )
                )
            elif db_mvs[name].strip() != definition.strip():
                create_ops.append(
                    ReplaceMaterializedViewOp(
                        name,
                        definition,
                        schema=schema,
                        with_data=False,
                        old_definition=db_mvs[name],
                    )
                )

        for name in db_mvs:
            if name not in model_mv_defs:
                drop_ops.append(
                    DropMaterializedViewOp(
                        name,
                        schema=schema,
                        definition=db_mvs[name],
                    )
                )

        # Order by dependency
        if create_ops:
            create_by_name = {op.name: op for op in create_ops}
            all_db = {**db_views, **db_mvs}
            try:
                ordered_records = resolve_create_order(model_records, all_db)
            except ValueError:
                log.warning(
                    "Circular view dependency detected; "
                    "creating views in model-definition order"
                )
                ordered_records = model_records

            for vr in ordered_records:
                if vr.name in create_by_name:
                    op = create_by_name.pop(vr.name)
                    upgrade_ops.ops.append(op)

            for op in create_by_name.values():
                upgrade_ops.ops.append(op)

        if drop_ops:
            drop_by_name = {op.name: op for op in drop_ops}
            all_db = {**db_views, **db_mvs}
            try:
                ordered_records = resolve_drop_order(model_records, all_db)
            except ValueError:
                log.warning(
                    "Circular view dependency detected; "
                    "dropping views in model-definition order"
                )
                ordered_records = model_records

            for vr in ordered_records:
                if vr.name in drop_by_name:
                    op = drop_by_name.pop(vr.name)
                    upgrade_ops.ops.append(op)

            # Drop any DB-only views not in model_records
            for op in drop_by_name.values():
                upgrade_ops.ops.append(op)


def _schema_matches(view_schema: str | None, loop_schema: str | None) -> bool:
    """Check whether a view's schema matches the current loop schema.

    Both ``None`` and ``'public'`` are treated as the default schema in PG.
    """
    if view_schema == loop_schema:
        return True
    # In PostgreSQL, schema=None maps to 'public'
    if view_schema is None and loop_schema == "public":
        return True
    if view_schema == "public" and loop_schema is None:
        return True
    return False


def include_view_comparator() -> None:
    """Activate view autogenerate support for Alembic.

    Call this in ``env.py`` **before** ``context.configure()``::

        from sqlalchemy_utils.alembic.comparator import include_view_comparator
        include_view_comparator()

    Imports the comparator, renderer, and operations modules to trigger
    ``@dispatch_for`` side-effect registrations.
    """
    from . import comparator, operations  # noqa: F401
    try:
        from . import renderer  # noqa: F401
    except ImportError:
        pass
