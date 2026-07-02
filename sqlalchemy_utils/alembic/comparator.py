"""
Alembic autogenerate comparator for database views.

Compares model-defined views (stored in ``metadata.info['sqlalchemy_utils_views']``)
against the current database state using **single-savepoint batch canonicalization**:
all model views for a schema are created inside *one* outer savepoint (so
view-on-view dependencies resolve because every view exists simultaneously),
their definitions are read back from ``pg_views``/``pg_matviews`` in one batch,
and the outer savepoint is rolled back once so nothing persists.

Views whose ``CREATE`` fails are **skipped** (not dropped) — they are tracked
in a ``skipped`` set so the drop-detection loop does not emit a false
``DropViewOp`` for an existing view that merely failed to canonicalize.

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


# Outer savepoint name shared across all views in one canonicalization batch.
# A single outer savepoint (RELEASEd never — always rolled back at the end)
# fixes BUG-7 (per-view savepoint accumulation) and BUG-3 (A rolled back
# before B is canonicalized → B's CREATE fails).
_OUTER_SAVEPOINT = "su_view_cmp"


def _compile_selectable(connection: sa.engine.Connection, view_record: ViewRecord) -> str:
    sel = view_record.selectable
    if isinstance(sel, str):
        return sel
    return str(
        sel.compile(
            dialect=connection.dialect,
            compile_kwargs={"literal_binds": True},
        )
    )


def _build_create_sql(connection: sa.engine.Connection, view_record: ViewRecord) -> str:
    """Build the CREATE (OR REPLACE) VIEW / MATERIALIZED VIEW statement."""
    definition = _compile_selectable(connection, view_record)
    ip = connection.dialect.identifier_preparer
    prefix = f"{ip.quote(view_record.schema)}." if view_record.schema else ""
    name = ip.quote(view_record.name)
    if view_record.materialized:
        # PG has no CREATE OR REPLACE MATERIALIZED VIEW; drop first.
        # The drop happens inside the outer savepoint so it never persists.
        return (
            f"DROP MATERIALIZED VIEW IF EXISTS {prefix}{name}; "
            f"CREATE MATERIALIZED VIEW {prefix}{name} "
            f"AS {definition} WITH NO DATA"
        )
    return f"CREATE OR REPLACE VIEW {prefix}{name} AS {definition}"


def _canonicalize_all_views(
    connection: sa.engine.Connection,
    view_records: list[ViewRecord],
    db_views_for_deps: dict[str, str] | None,
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    """Canonicalize all *view_records* for one schema in a single savepoint.

    Creates ONE outer savepoint, creates each view (in dependency order) inside
    nested savepoints that are RELEASEd on success, reads all definitions back
    from pg_catalog in one batch, then rolls back the outer savepoint so
    nothing persists.

    Views that fail to CREATE are **skipped** — their names are returned in
    the ``skipped`` set so the caller can exclude them from drop detection
    (BUG-2: a failed canonicalization must NOT produce a false DropViewOp).

    Returns ``(view_defs, mv_defs, skipped)`` where ``view_defs`` /
    ``mv_defs`` map view name → canonical definition for regular /
    materialized views, and ``skipped`` holds names that failed to
    canonicalize.
    """
    view_defs: dict[str, str] = {}
    mv_defs: dict[str, str] = {}
    skipped: set[str] = set()

    if not view_records:
        return view_defs, mv_defs, skipped

    # Order by dependency so a view is created before any view that references
    # it (BUG-3 fix: all views coexist inside the single outer savepoint).
    try:
        ordered = resolve_create_order(view_records, db_views_for_deps)
    except ValueError:
        log.warning(
            "Circular view dependency detected; "
            "canonicalizing views in model-definition order"
        )
        ordered = view_records

    schema = view_records[0].schema
    connection.execute(sa.text(f"SAVEPOINT {_OUTER_SAVEPOINT}"))
    try:
        for vr in ordered:
            # Inner savepoint per view: on success RELEASE (view persists in
            # the outer savepoint so dependents can see it); on failure
            # ROLLBACK TO (cleans the aborted sub-transaction without
            # touching already-created views). Releasing avoids savepoint
            # accumulation (BUG-7).
            inner_sp = f"{_OUTER_SAVEPOINT}_v"
            connection.execute(sa.text(f"SAVEPOINT {inner_sp}"))
            try:
                connection.execute(sa.text(_build_create_sql(connection, vr)))
                connection.execute(sa.text(f"RELEASE SAVEPOINT {inner_sp}"))
            except (sa.exc.SQLAlchemyError, sa.exc.DBAPIError, OSError) as exc:
                log.warning(
                    "Failed to canonicalize view '%s': %s", vr.name, exc
                )
                try:
                    connection.execute(
                        sa.text(f"ROLLBACK TO SAVEPOINT {inner_sp}")
                    )
                    # ROLLBACK TO does not destroy the savepoint (PG
                    # semantics); RELEASE it so the next iteration can
                    # create a fresh one (BUG-10).
                    connection.execute(
                        sa.text(f"RELEASE SAVEPOINT {inner_sp}")
                    )
                except (sa.exc.SQLAlchemyError, sa.exc.DBAPIError, OSError):
                    pass
                skipped.add(vr.name)

        # Batch-read canonical definitions from pg_catalog in one query per
        # kind (regular + materialized) rather than one per view.
        db_views = get_database_views(connection, schema)
        db_mvs = get_database_materialized_views(connection, schema)
    finally:
        connection.execute(sa.text(f"ROLLBACK TO SAVEPOINT {_OUTER_SAVEPOINT}"))

    for vr in ordered:
        if vr.name in skipped:
            continue
        if vr.materialized:
            if vr.name in db_mvs:
                mv_defs[vr.name] = db_mvs[vr.name]
            else:
                skipped.add(vr.name)
        else:
            if vr.name in db_views:
                view_defs[vr.name] = db_views[vr.name]
            else:
                skipped.add(vr.name)
    return view_defs, mv_defs, skipped


def compare_views(
    autogen_context: AutogenContext,
    upgrade_ops,
    schemas,
) -> None:
    """Compare model-defined views against database state.

    This function is registered as an Alembic ``"schema"`` comparator and is
    called automatically during ``alembic revision --autogenerate``.

    It reads view definitions from ``metadata.info['sqlalchemy_utils_views']``
    (populated by `create_view()`, `create_materialized_view()`, and
    `ViewMixin.__declare_last__`), canonicalizes each model view via
    savepoint simulation, and diffs against the live database.
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

        # Batch-canonicalize all model views for this schema inside ONE outer
        # savepoint (BUG-3/BUG-7). Views that fail to canonicalize are
        # returned in `skipped` so drop detection can ignore them (BUG-2).
        schema_records = [
            vr for vr in model_records if _schema_matches(vr.schema, schema)
        ]
        model_view_defs, model_mv_defs, skipped = _canonicalize_all_views(
            connection, schema_records, all_db
        )

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

        # BUG-2: only drop views that are genuinely in the DB but NOT in the
        # model. Views in `skipped` failed canonicalization and must NOT be
        # dropped — they are still modeled, just not canonicalizable right now.
        for name in db_views:
            if name in model_view_defs or name in skipped:
                continue
            drop_ops.append(
                DropViewOp(
                    name,
                    schema=schema,
                    materialized=False,
                    definition=db_views[name],
                )
            )
            try:
                dependents = get_dependent_views(connection, name, schema=schema)
            except Exception as exc:
                log.warning(
                    "Failed to query dependent views for %r: %s", name, exc
                )
                dependents = {}
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
            if name in model_mv_defs or name in skipped:
                continue
            drop_ops.append(
                DropMaterializedViewOp(
                    name,
                    schema=schema,
                    definition=db_mvs[name],
                )
            )
            try:
                dependents = get_dependent_views(connection, name, schema=schema)
            except Exception as exc:
                log.warning(
                    "Failed to query dependent views for %r: %s", name, exc
                )
                dependents = {}
            if dependents:
                log.warning(
                    "Dropping materialized view %r which has %d dependent view(s): %s. "
                    "CASCADE will drop them automatically. "
                    "Remove the dependent views from your model first if this is unintended.",
                    name, len(dependents), ", ".join(sorted(dependents.keys())),
                )

        # Order by dependency
        if create_ops:
            create_by_name = {(op.name, op.schema): op for op in create_ops}
            try:
                ordered_records = resolve_create_order(model_records, all_db)
            except ValueError:
                log.warning(
                    "Circular view dependency detected; "
                    "creating views in model-definition order"
                )
                ordered_records = model_records

            for vr in ordered_records:
                key = (vr.name, vr.schema)
                if key in create_by_name:
                    op = create_by_name.pop(key)
                    upgrade_ops.ops.append(op)

            for op in create_by_name.values():
                upgrade_ops.ops.append(op)

        if drop_ops:
            drop_by_name = {(op.name, op.schema): op for op in drop_ops}
            try:
                ordered_records = resolve_drop_order(model_records, all_db)
            except ValueError:
                log.warning(
                    "Circular view dependency detected; "
                    "dropping views in model-definition order"
                )
                ordered_records = model_records

            for vr in ordered_records:
                key = (vr.name, vr.schema)
                if key in drop_by_name:
                    op = drop_by_name.pop(key)
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

    Registers the view comparator (``compare_views``) with Alembic's
    autogenerate system so that ``alembic revision --autogenerate``
    detects database view changes (create / drop / replace for both
    regular and materialized views). The comparator walks the model
    metadata, compares it against the live database, and emits the
    appropriate ``CreateViewOp`` / ``DropViewOp`` / ``ReplaceViewOp``
    (and materialized variants) operations.

    Call this once in your ``env.py`` **before** ``context.configure()``::

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
    except ImportError as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Failed to import renderer module; autogenerate will detect but "
            "not render view operations: %s",
            exc,
        )
