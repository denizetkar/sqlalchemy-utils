"""
Alembic autogenerate comparator for database views.

Compares model-defined views (stored in ``metadata.info['sqlalchemy_utils_views']``)
against the current database state using **savepoint canonicalization**: each model
view is temporarily created inside a savepoint, its definition read back from
``pg_views``/``pg_matviews``, and the savepoint is rolled back so nothing persists.

Differences are emitted as :class:`CreateViewOp`, :class:`DropViewOp`,
:class:`ReplaceViewOp` (and their materialized-view equivalents).

Usage in ``env.py``::

    from sqlalchemy_utils.alembic.comparator import register_view_comparator
    register_view_comparator()   # must be called before context.configure()
"""
from __future__ import annotations

import logging

import sqlalchemy as sa
from alembic.autogenerate import comparators
from alembic.autogenerate.api import AutogenContext

from sqlalchemy_utils.view_record import ViewRecord
from sqlalchemy_utils.alembic.pg_catalog import (
    get_database_views,
    get_database_materialized_views,
    get_dependent_views,
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
            definition = str(
                sel.compile(
                    dialect=connection.dialect,
                    compile_kwargs={"literal_binds": True},
                )
            )
        else:
            definition = sel

        ip = connection.dialect.identifier_preparer
        prefix = f"{ip.quote(view_record.schema)}." if view_record.schema else ""
        name = ip.quote(view_record.name)
        if view_record.materialized:
            # For MVs, if one already exists we must drop it first
            # (PG has no CREATE OR REPLACE MATERIALIZED VIEW)
            connection.execute(sa.text(
                f"DROP MATERIALIZED VIEW IF EXISTS {prefix}{name} CASCADE"
            ))
            sql = (
                f"CREATE MATERIALIZED VIEW {prefix}{name} "
                f"AS {definition} WITH NO DATA"
            )
        else:
            # Use OR REPLACE to avoid DuplicateTable if view already exists in DB
            sql = (
                f"CREATE OR REPLACE VIEW {prefix}{name} "
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

    if connection is None:
        log.warning(
            "View autogenerate comparison requires an online connection; "
            "skipping view diffing in offline mode."
        )
        return

    if connection.dialect.name != 'postgresql':
        log.warning(
            "View autogenerate comparison is only supported on PostgreSQL; "
            "skipping view diffing for non-PostgreSQL dialect '%s'.",
            connection.dialect.name,
        )
        return

    if schemas is None:
        schemas = [None]

    model_records: list[ViewRecord] = metadata.info.get(
        "sqlalchemy_utils_views", []
    )

    # Cross-schema dependency resolution requires DB state from all schemas.
    # Collect DB views for all schemas (single fetch per schema).
    all_db_views: dict[str, list[str]] = {}
    all_db_mvs: dict[str, list[str]] = {}
    db_views_by_schema: dict[str | None, dict[str, str]] = {}
    db_mvs_by_schema: dict[str | None, dict[str, str]] = {}

    for schema in schemas:
        db_views = get_database_views(connection, schema)
        db_mvs = get_database_materialized_views(connection, schema)
        db_views_by_schema[schema] = db_views
        db_mvs_by_schema[schema] = db_mvs
        for name, definition in db_views.items():
            all_db_views.setdefault(name, []).append(definition)
        for name, definition in db_mvs.items():
            all_db_mvs.setdefault(name, []).append(definition)
    # Flatten: for dependency resolution, any definition suffices
    all_db = {name: defs[0] for name, defs in {**all_db_views, **all_db_mvs}.items()}

    for schema in schemas:
        db_views = db_views_by_schema[schema]
        db_mvs = db_mvs_by_schema[schema]

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
                dependents = get_dependent_views(connection, name, schema=schema)
                if dependents:
                    log.warning(
                        "Dropping view %r which has %d dependent view(s): %s. "
                        "CASCADE will drop them automatically. "
                        "Remove the dependent views from your model first if this is unintended.",
                        name, len(dependents), ", ".join(sorted(dependents.keys())),
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
                dependents = get_dependent_views(connection, name, schema=schema)
                if dependents:
                    log.warning(
                        "Dropping materialized view %r which has %d dependent view(s): %s. "
                        "CASCADE will drop them automatically. "
                        "Remove the dependent views from your model first if this is unintended.",
                        name, len(dependents), ", ".join(sorted(dependents.keys())),
                    )

        # Order by dependency
        if create_ops:
            create_by_name = {op.name: op for op in create_ops}
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

    seen: set = set()
    deduped: list = []
    for op in upgrade_ops.ops:
        # Normalize op type to a family prefix (create/replace/drop)
        # so conflicting ops for the same view are deduped.
        op_name = type(op).__name__.lower()
        if op_name.startswith("create") or op_name.startswith("replace"):
            op_family = "create_or_replace"
        elif op_name.startswith("drop"):
            op_family = "drop"
        else:
            op_family = op_name
        key = (op_family, getattr(op, "name", None), getattr(op, "schema", None))
        if key not in seen:
            seen.add(key)
            deduped.append(op)
    upgrade_ops.ops = deduped


def _schema_matches(view_schema: str | None, loop_schema: str | None) -> bool:
    """Check whether a view's schema matches the current loop schema (exact match)."""
    return view_schema == loop_schema


def register_view_comparator() -> None:
    """Register view autogenerate hooks with Alembic.

    Registers the ``"schema"`` comparator (``compare_views``) plus the
    corresponding renderer and operation classes so that
    ``alembic revision --autogenerate`` detects database view changes
    (create / drop / replace for both regular and materialized views).

    Must be called in ``env.py`` **before** ``context.configure()``::

        from sqlalchemy_utils.alembic.comparator import register_view_comparator
        register_view_comparator()

    This function is idempotent (safe to call more than once).  The
    comparator is registered lazily — merely importing an Op class from
    :mod:`sqlalchemy_utils.alembic` does **not** activate autogenerate.
    """
    from . import comparator, operations  # noqa: F401
    comparators.dispatch_for("schema")(compare_views)
    try:
        from . import renderer  # noqa: F401
    except ImportError:
        pass
