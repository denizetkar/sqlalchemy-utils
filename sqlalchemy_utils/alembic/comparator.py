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

**Iterative canonicalization** — when a view fails to create because it
references a column or table being added in the same migration, the
comparator applies the view-relevant DDL (``ADD COLUMN`` nullable, ``CREATE
TABLE`` empty) inside the same outer savepoint and re-attempts
canonicalization of the skipped views. This repeats to a fixpoint (no
progress) or a 10-iteration safety cap, so views that depend on
same-migration schema changes are detected instead of silently skipped.
See ``_canonicalize_all_views`` for the full contract and limitations
(notably the cross-schema table creation restriction).

Differences are emitted as :class:`~sqlalchemy_utils.alembic.operations.CreateViewOp`,
:class:`~sqlalchemy_utils.alembic.operations.DropViewOp`,
:class:`~sqlalchemy_utils.alembic.operations.ReplaceViewOp` (and their materialized-view equivalents).

Usage in ``env.py``::

    from sqlalchemy_utils import register_view_comparator
    register_view_comparator()   # must be called before context.configure()
"""
from __future__ import annotations

import logging
from collections.abc import Callable

import sqlalchemy as sa
from alembic.autogenerate import comparators
from alembic.autogenerate.api import AutogenContext
from alembic.operations.ops import UpgradeOps

from sqlalchemy_utils.view_record import ViewRecord
from sqlalchemy_utils.view import _quote_qualified_name
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
    RefreshMaterializedViewOp,
)
from sqlalchemy_utils.alembic.depend import resolve_create_order, resolve_drop_order

log = logging.getLogger(__name__)


# Outer savepoint name shared across all views in one canonicalization batch.
# A single outer savepoint (RELEASEd never — always rolled back at the end)
# avoids per-view savepoint accumulation and ensures that a view rolled back
# before a dependent is canonicalized does not break the dependent's CREATE.
_OUTER_SAVEPOINT = "su_view_cmp"

# Exception types treated as expected during view canonicalization.
_CANON_ERRORS = (sa.exc.SQLAlchemyError, OSError)

# Idempotency guard; Alembic runs single-threaded and dispatch_for is itself idempotent.
_registered = False


def _build_create_sql(connection: sa.engine.Connection, view_record: ViewRecord) -> list[str]:
    """Build the CREATE VIEW / MATERIALIZED VIEW statement(s).

    Returns a list of SQL strings so the caller can execute each statement
    separately. Multi-statement ``sa.text()`` relies on the simple-query
    protocol, which is driver-specific (psycopg2 supports it; asyncpg does
    not); returning a list keeps canonicalization portable across drivers.

    For both regular and materialized views returns two statements
    (``DROP`` then ``CREATE``). PG has no ``CREATE OR REPLACE MATERIALIZED
    VIEW``, and ``CREATE OR REPLACE VIEW`` fails when the new view's
    column structure differs from the existing view (e.g. removing or
    reordering columns) — which would skip the view and silently miss the
    definition change. DROP+CREATE avoids this because it runs inside a
    savepoint that is rolled back. CASCADE is needed so dependent views
    in the DB do not block the DROP.
    """
    definition = view_record.compiled_definition(dialect=connection.dialect)
    dialect = connection.dialect
    qualified = _quote_qualified_name(dialect, view_record.name, view_record.schema)
    if view_record.materialized:
        return [
            f"DROP MATERIALIZED VIEW IF EXISTS {qualified} CASCADE",
            f"CREATE MATERIALIZED VIEW {qualified} "
            f"AS {definition} WITH NO DATA",
        ]
    return [
        f"DROP VIEW IF EXISTS {qualified} CASCADE",
        f"CREATE VIEW {qualified} AS {definition}",
    ]


def _apply_view_relevant_ddl(
    connection: sa.engine.Connection,
    metadata: sa.MetaData,
    schema: str | None,
) -> int:
    """Apply view-relevant DDL (ADD COLUMN nullable, CREATE TABLE empty).

    Introspects model *metadata* against the current DB state and applies the
    minimal DDL needed for view canonicalization to succeed: a missing table
    is created with all model columns as nullable (no constraints, defaults,
    or keys); a missing column on an existing table is added as nullable with
    no default.

    Each DDL statement runs inside its own inner savepoint so that a single
    failure does not poison the outer transaction (PostgreSQL aborts the whole
    transaction on any statement error; only ``ROLLBACK TO SAVEPOINT``
    un-poisons it). Failures are logged and skipped; the count of
    successfully applied statements is returned.

    The inspector is built fresh on every call so it reflects DDL applied in
    prior iterations (a cached inspector would not see newly created tables).
    """
    applied = 0
    inner_sp = f"{_OUTER_SAVEPOINT}_ddl"
    preparer = connection.dialect.identifier_preparer

    for table in metadata.tables.values():
        if table.schema != schema:
            continue

        inspector = sa.inspect(connection)

        qualified = _quote_qualified_name(
            connection.dialect, table.name, schema
        )

        if not inspector.has_table(table.name, schema=schema):
            col_defs = ", ".join(
                f"{preparer.quote(col.name)} "
                f"{col.type.compile(dialect=connection.dialect)}"
                for col in table.columns
            )
            stmt = f"CREATE TABLE {qualified} ({col_defs})"
        else:
            db_columns = {
                col["name"]
                for col in inspector.get_columns(table.name, schema=schema)
            }
            missing = [
                col
                for col in table.columns
                if col.name not in db_columns
            ]
            if not missing:
                continue
            # One ALTER per column so a single failure does not block the rest.
            for col in missing:
                col_type = col.type.compile(dialect=connection.dialect)
                stmt = (
                    f"ALTER TABLE {qualified} ADD COLUMN "
                    f"{preparer.quote(col.name)} {col_type}"
                )
                if _execute_ddl_in_savepoint(
                    connection, inner_sp, stmt
                ):
                    applied += 1
            continue

        if _execute_ddl_in_savepoint(connection, inner_sp, stmt):
            applied += 1

    return applied


def _execute_ddl_in_savepoint(
    connection: sa.engine.Connection,
    savepoint: str,
    stmt: str,
) -> bool:
    """Execute a single DDL statement inside its own inner savepoint.

    Returns ``True`` if the statement was applied, ``False`` if it raised
    a ``_CANON_ERRORS`` (the savepoint is rolled back and the failure logged).
    """
    connection.execute(sa.text(f"SAVEPOINT {savepoint}"))
    try:
        connection.execute(sa.text(stmt))
        connection.execute(sa.text(f"RELEASE SAVEPOINT {savepoint}"))
        return True
    except _CANON_ERRORS as exc:
        log.warning("Failed to apply DDL %r: %s", stmt, exc)
        try:
            connection.execute(sa.text(f"ROLLBACK TO SAVEPOINT {savepoint}"))
            connection.execute(sa.text(f"RELEASE SAVEPOINT {savepoint}"))
        except _CANON_ERRORS:
            pass
        return False


def _canonicalize_all_views(
    connection: sa.engine.Connection,
    view_records: list[ViewRecord],
    db_views_for_deps: dict[str, str] | None,
    metadata: sa.MetaData | None = None,
) -> tuple[dict[str, str], dict[str, str], set[str]]:
    """Canonicalize all *view_records* for one schema in a single savepoint.

    Creates ONE outer savepoint, creates each view (in dependency order) inside
    nested savepoints that are RELEASEd on success, reads all definitions back
    from pg_catalog in one batch, then rolls back the outer savepoint so
    nothing persists.

    Views that fail to CREATE are **skipped** — their names are returned in
    the ``skipped`` set so the caller can exclude them from drop detection
    (a failed canonicalization must NOT produce a false DropViewOp).

    When *metadata* is provided and declares columns/tables that do not yet
    exist in the live DB (because they are being added in the same migration),
    pass 1 skips the views that reference them. After pass 1, an **iterative
    fixpoint loop** applies the view-relevant DDL inside the same outer
    savepoint and re-canonicalizes only the skipped views:

    - **DDL scope** — only ``ADD COLUMN`` (nullable, no default) and
      ``CREATE TABLE`` (empty: no indexes, constraints, foreign keys, and no
      ``ALTER COLUMN`` type changes) are applied. This is sufficient for view
      creation, which only needs the column to exist; full table shape is
      the migration's job.
    - **Re-canonicalization** — only views in the ``skipped`` set are retried
      after each DDL pass; successfully canonicalized views are not re-touched.
    - **Fixpoint termination** — the loop stops as soon as a pass applies no
      DDL and canonicalizes no new views (no progress), i.e. it terminates at
      the fixpoint where remaining skips cannot be resolved by further DDL.
    - **Safety cap** — at most 10 iterations are attempted; if the fixpoint
      is not reached, a warning is logged and the still-skipped views remain
      in ``skipped`` (no false ``DropViewOp`` is emitted for them).

    Limitation: a view in schema A that references a newly added table in
    schema B will not be rescued, because ``_apply_view_relevant_ddl`` filters
    tables by *schema* and only creates tables in the view's own schema.

    When *metadata* is ``None`` (legacy direct callers), the iterative loop is
    skipped and behavior is unchanged.

    Returns ``(view_defs, mv_defs, skipped)`` where ``view_defs`` /
    ``mv_defs`` map view name → canonical definition for regular /
    materialized views, and ``skipped`` holds names that failed to
    canonicalize.
    """
    view_defs: dict[str, str] = {}
    mv_defs: dict[str, str] = {}
    skipped: set[str] = set()
    processed: set[str] = set()

    if not view_records:
        return view_defs, mv_defs, skipped

    # Order by dependency so a view is created before any view that references
    # it (all views coexist inside the single outer savepoint).
    ordered = _safe_resolve(
        view_records,
        db_views_for_deps,
        resolve_create_order,
        "canonicalizing",
        dialect=connection.dialect,
    )

    schema = view_records[0].schema
    connection.execute(sa.text(f"SAVEPOINT {_OUTER_SAVEPOINT}"))
    try:
        for vr in ordered:
            # Inner savepoint per view: RELEASE on success, ROLLBACK TO on failure.
            inner_sp = f"{_OUTER_SAVEPOINT}_v"
            connection.execute(sa.text(f"SAVEPOINT {inner_sp}"))
            try:
                for stmt in _build_create_sql(connection, vr):
                    connection.execute(sa.text(stmt))
                connection.execute(sa.text(f"RELEASE SAVEPOINT {inner_sp}"))
            except _CANON_ERRORS as exc:
                log.warning(
                    "Failed to canonicalize view '%s': %s", vr.name, exc
                )
                try:
                    connection.execute(
                        sa.text(f"ROLLBACK TO SAVEPOINT {inner_sp}")
                    )
                    # ROLLBACK TO keeps the savepoint alive; RELEASE for reuse.
                    connection.execute(
                        sa.text(f"RELEASE SAVEPOINT {inner_sp}")
                    )
                except _CANON_ERRORS:
                    pass
                skipped.add(vr.name)

                # Probe the outer savepoint: a poisoned transaction would
                # silently skip all remaining views, so break early instead.
                try:
                    connection.execute(sa.text("SELECT 1"))
                except _CANON_ERRORS:
                    log.warning(
                        "Outer savepoint is in aborted state after failing to "
                        "canonicalize view '%s'; aborting canonicalization of "
                        "remaining views", vr.name
                    )
                    processed.add(vr.name)
                    # Unreached views go to `skipped` to avoid false DropViewOps.
                    for remaining_vr in ordered:
                        if remaining_vr.name not in processed:
                            skipped.add(remaining_vr.name)
                            log.warning(
                                "View %r in schema %r skipped (canonicalization aborted due to prior failure) — "
                                "no migration op will be generated for it.",
                                remaining_vr.name, schema,
                            )
                    break
            processed.add(vr.name)

        # Iterative fixpoint: apply view-relevant DDL from *metadata* inside
        # the same outer savepoint, re-canonicalize ONLY skipped views, repeat
        # until no progress (fixpoint) or 10-iteration safety cap. Skipped if
        # pass 1 poisoned the outer savepoint (probe above already marked
        # unreached views as skipped). Necessary because PG aborts the whole
        # transaction on any statement error, so a poisoned savepoint cannot
        # be recovered without ROLLBACK TO SAVEPOINT.
        if skipped and metadata is not None:
            outer_poisoned = False
            try:
                connection.execute(sa.text("SELECT 1"))
            except _CANON_ERRORS:
                outer_poisoned = True

            if not outer_poisoned:
                max_iterations = 10
                for _iteration in range(max_iterations):
                    ddl_count = _apply_view_relevant_ddl(
                        connection, metadata, schema
                    )

                    skipped_records = [
                        vr for vr in ordered if vr.name in skipped
                    ]
                    newly_canonicalized = 0
                    for vr in skipped_records:
                        inner_sp = f"{_OUTER_SAVEPOINT}_v"
                        connection.execute(
                            sa.text(f"SAVEPOINT {inner_sp}")
                        )
                        try:
                            for stmt in _build_create_sql(connection, vr):
                                connection.execute(sa.text(stmt))
                            connection.execute(
                                sa.text(f"RELEASE SAVEPOINT {inner_sp}")
                            )
                        except _CANON_ERRORS as exc:
                            log.warning(
                                "Failed to re-canonicalize view '%s': %s",
                                vr.name, exc,
                            )
                            try:
                                connection.execute(
                                    sa.text(
                                        f"ROLLBACK TO SAVEPOINT {inner_sp}"
                                    )
                                )
                                connection.execute(
                                    sa.text(
                                        f"RELEASE SAVEPOINT {inner_sp}"
                                    )
                                )
                            except _CANON_ERRORS:
                                pass
                            continue

                        skipped.discard(vr.name)
                        processed.add(vr.name)
                        newly_canonicalized += 1

                    if newly_canonicalized == 0:
                        break
                    if ddl_count == 0 and newly_canonicalized == 0:
                        break
                else:
                    log.warning(
                        "Iterative canonicalization reached max iterations "
                        "(%d) for schema %r",
                        max_iterations, schema,
                    )

        # Batch-read from pg_catalog inside the outer savepoint (before
        # rollback) so just-created views are visible.
        db_views: dict[str, str] = {}
        db_mvs: dict[str, str] = {}
        try:
            db_views = get_database_views(connection, schema)
            db_mvs = get_database_materialized_views(connection, schema)
        except _CANON_ERRORS as exc:
            # Catalog readback can fail if the transaction was poisoned
            # mid-loop without a per-view failure; treat as poisoned.
            log.warning(
                "Catalog readback failed for schema %r (transaction may be "
                "poisoned): %s. Marking all views skipped.", schema, exc
            )
            for vr in ordered:
                skipped.add(vr.name)
    finally:
        # Guarded rollback: a poisoned transaction would raise and mask
        # the original exception propagated from the body.
        try:
            connection.execute(
                sa.text(f"ROLLBACK TO SAVEPOINT {_OUTER_SAVEPOINT}")
            )
        except _CANON_ERRORS as exc:
            log.warning(
                "Failed to roll back outer savepoint %r: %s",
                _OUTER_SAVEPOINT,
                exc,
            )

    for vr in ordered:
        if vr.name in skipped or vr.name not in processed:
            continue
        source = db_mvs if vr.materialized else db_views
        target = mv_defs if vr.materialized else view_defs
        if vr.name in source:
            target[vr.name] = source[vr.name]
        else:
            log.warning(
                "View %r was processed but not found in catalog readback "
                "(may have been cascade-dropped); marking as skipped.",
                vr.name,
            )
            skipped.add(vr.name)
    return view_defs, mv_defs, skipped


def _diff_views(
    model_defs: dict[str, str],
    db_defs: dict[str, str],
    schema: str | None,
    is_materialized: bool,
    cascade_by_name: dict[tuple[str, str | None], bool],
) -> list:
    """Diff model view definitions against DB state, returning create/replace ops.

    Absent views become Create ops; changed definitions become Replace ops.
    Op class is selected by *is_materialized*; materialized ops use
    ``with_data=False`` (autogenerate emits ``WITH NO DATA``).

    ``cascade_by_name`` propagates each model's ``cascade_on_drop`` preference
    to the emitted Create/Replace ops (default ``True``, behavior-preserving).
    """
    ops: list = []
    for name, definition in model_defs.items():
        if name not in db_defs:
            if is_materialized:
                ops.append(
                    CreateMaterializedViewOp(
                        name, definition, schema=schema, with_data=False,
                        cascade_on_drop=cascade_by_name.get((name, schema), True),
                    )
                )
            else:
                ops.append(CreateViewOp(
                    name, definition, schema=schema,
                    cascade_on_drop=cascade_by_name.get((name, schema), True),
                ))
        elif db_defs[name].strip() != definition.strip():
            if is_materialized:
                ops.append(
                    ReplaceMaterializedViewOp(
                        name,
                        definition,
                        schema=schema,
                        with_data=False,
                        cascade=cascade_by_name.get((name, schema), True),
                        old_definition=db_defs[name],
                    )
                )
            else:
                ops.append(
                    ReplaceViewOp(
                        name,
                        definition,
                        schema=schema,
                        cascade=cascade_by_name.get((name, schema), True),
                        old_definition=db_defs[name],
                    )
                )
    return ops


def _warn_if_dependents(
    connection: sa.engine.Connection,
    name: str,
    schema: str | None,
    kind_label: str,
) -> None:
    """Warn if dropping *name* would cascade to dependent views.

    Queries ``get_dependent_views``; on failure logs a warning and proceeds
    (treats dependents as empty). When dependents exist, logs a warning naming
    *kind_label* (e.g. "view" or "materialized view") so the user knows the
    CASCADE will silently drop them.
    """
    try:
        dependents = get_dependent_views(connection, name, schema=schema)
    except sa.exc.SQLAlchemyError as exc:
        log.warning(
            "Failed to query dependent views for %r: %s", name, exc
        )
        dependents = {}
    if dependents:
        formatted = [
            f"{s}.{n}" if s else n
            for n, s in sorted(dependents)
        ]
        log.warning(
            "Dropping %s %r which has %d dependent view(s): %s. "
            "CASCADE will drop them automatically. "
            "Remove the dependent views from your model first if this is unintended.",
            kind_label,
            name,
            len(dependents),
            ", ".join(formatted),
        )


def _safe_resolve(
    records: list[ViewRecord],
    db: dict[str, str],
    resolver_fn: Callable[..., list[ViewRecord]],
    action_label: str,
    *,
    dialect: sa.engine.Dialect | None = None,
) -> list[ViewRecord]:
    """Resolve view ordering, falling back to model order on failure.

    Wraps *resolver_fn* (e.g. :func:`~sqlalchemy_utils.alembic.depend.resolve_create_order`) in a
    try/except. If a circular dependency (``ValueError``) is detected, or
    a view's ``ClauseElement`` fails compilation
    (``sa.exc.SQLAlchemyError`` — e.g. ``CompileError``,
    ``ArgumentError``), logs a warning naming *action_label* (e.g.
    ``"canonicalizing"``, ``"creating"``, ``"dropping"``) and returns
    *records* unchanged. Without the widened catch a single un-compilable
    view would propagate and abort the entire autogenerate run.

    *dialect* is forwarded to ``resolver_fn`` so dependency detection
    scans dialect-qualified SQL matching the comparator's emitted DDL.
    """
    try:
        return resolver_fn(records, db, dialect=dialect)
    except (ValueError, sa.exc.SQLAlchemyError):
        log.warning(
            "Circular view dependency or compilation error detected; "
            "%s views in model-definition order",
            action_label,
        )
        return records


def _is_create_family(op) -> bool:
    return isinstance(
        op,
        (CreateViewOp, CreateMaterializedViewOp, ReplaceViewOp, ReplaceMaterializedViewOp),
    )


def _is_drop_family(op) -> bool:
    return isinstance(op, (DropViewOp, DropMaterializedViewOp))


def _reorder_cross_type_drops_before_creates(ops: list) -> list:
    """Reorder so all cross-type drops precede all cross-type creates.

    When a view changes type (regular <-> materialized) the comparator emits
    a Create-family op for the new type and a Drop-family op for the old
    type. Creates are appended before drops, so without reordering the
    migration would CREATE while the old-type view still exists (which
    fails). For any (name, schema) present in BOTH a drop and a create op,
    all drops are emitted before all creates at the position of the first
    cross-type op, so non-cross-type ops keep their relative order.

    Drops are collected in their appearance order in ``ops`` — which is
    drop-order (dependents before dependencies) from
    :func:`~sqlalchemy_utils.alembic.depend.resolve_drop_order`. Creates
    are collected in their appearance order — create-order (dependencies
    before dependents) from
    :func:`~sqlalchemy_utils.alembic.depend.resolve_create_order`.
    """
    create_keys = {
        (getattr(op, "name", None), getattr(op, "schema", None))
        for op in ops if _is_create_family(op)
    }
    drop_keys = {
        (getattr(op, "name", None), getattr(op, "schema", None))
        for op in ops if _is_drop_family(op)
    }
    cross_keys = create_keys & drop_keys
    if not cross_keys:
        return ops

    all_drops: list = []
    cross_creates: list = []
    for op in ops:
        key = (getattr(op, "name", None), getattr(op, "schema", None))
        if _is_drop_family(op):
            all_drops.append(op)
        elif _is_create_family(op) and key in cross_keys:
            cross_creates.append(op)

    result: list = []
    inserted = False
    for op in ops:
        key = (getattr(op, "name", None), getattr(op, "schema", None))
        if _is_drop_family(op) or key in cross_keys:
            if not inserted:
                result.extend(all_drops)
                result.extend(cross_creates)
                inserted = True
            continue
        result.append(op)
    return result


def _order_ops(ops, records, db, resolver_fn, action_label, *, dialect=None):
    """Order *ops* by dependency using *resolver_fn*.

    Builds a ``{(name, schema): op}`` mapping from *ops*, resolves the
    ordering of *records* via :func:`_safe_resolve`, appends each matching
    op (in dependency order) to the result, then appends any remaining
    (DB-only) ops. Returns the ordered list of ops.

    *dialect* is forwarded to ``resolver_fn`` for dialect-aware dependency
    detection.
    """
    by_name = {(op.name, op.schema): op for op in ops}
    ordered_records = _safe_resolve(
        records, db, resolver_fn, action_label, dialect=dialect
    )
    result = []
    for vr in ordered_records:
        key = (vr.name, vr.schema)
        if key in by_name:
            result.append(by_name.pop(key))
    for op in by_name.values():
        result.append(op)
    return result


def _collect_drop_ops(
    db_defs,
    model_defs,
    skipped,
    schema,
    cascade_by_name,
    op_class,
    kind_label,
    connection,
    drop_ops,
):
    """Collect drop ops for DB views not in the model, warning about dependents."""
    for name in db_defs:
        if name in model_defs or name in skipped:
            continue
        drop_ops.append(
            op_class(
                name,
                schema=schema,
                cascade=cascade_by_name.get((name, schema), True),
                definition=db_defs[name],
            )
        )
        _warn_if_dependents(connection, name, schema, kind_label)


def compare_views(
    autogen_context: AutogenContext,
    upgrade_ops: UpgradeOps,
    schemas: list[str | None] | None = None,
) -> None:
    """Compare model-defined views against database state.

    .. note:: Internal — registered as the Alembic ``"schema"`` comparator by :func:`register_view_comparator`. Not part of the stable public API.

    This function is registered as an Alembic ``"schema"`` comparator and is
    called automatically during ``alembic revision --autogenerate``.

    It reads view definitions from ``metadata.info['sqlalchemy_utils_views']``
    (populated by :func:`~sqlalchemy_utils.view.create_view`,
    :func:`~sqlalchemy_utils.view.create_materialized_view`, and
    :meth:`~sqlalchemy_utils.view_mixin.ViewMixin.__declare_last__`),
    canonicalizes each model view via
    savepoint simulation, and diffs against the live database.

    :param autogen_context:
        The Alembic :class:`~alembic.autogenerate.api.AutogenContext`
        providing the live database connection and model metadata.
    :param upgrade_ops:
        The :class:`~alembic.operations.ops.UpgradeOps` into which
        detected ``CreateViewOp`` / ``DropViewOp`` / ``ReplaceViewOp``
        (and materialized variants) are appended.
    :param schemas:
        List of schema names to compare. ``None`` is treated as
        ``[None]`` (i.e. the connection's default schema only); pass an
        explicit list of schema names to compare non-default schemas.
    :returns: ``None``. Detected differences are appended to
        *upgrade_ops* in place.
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
    db_views_by_schema: dict[str | None, dict[str, str]] = {}
    db_mvs_by_schema: dict[str | None, dict[str, str]] = {}
    all_db: dict[str, str] = {}

    for schema in schemas:
        db_views = get_database_views(connection, schema)
        db_mvs = get_database_materialized_views(connection, schema)
        db_views_by_schema[schema] = db_views
        db_mvs_by_schema[schema] = db_mvs
        all_db.update(db_views)
        all_db.update(db_mvs)

    all_create_ops: list = []
    all_drop_ops: list = []

    for schema in schemas:
        db_views = db_views_by_schema[schema]
        db_mvs = db_mvs_by_schema[schema]

        # Batch-canonicalize all model views for this schema inside ONE outer
        # savepoint. Views that fail to canonicalize are returned in
        # `skipped` so drop detection can ignore them.
        schema_records = [
            vr for vr in model_records if vr.schema == schema
        ]
        model_view_defs, model_mv_defs, skipped = _canonicalize_all_views(
            connection, schema_records, all_db, metadata
        )

        # Diff model vs DB
        create_ops: list = []
        drop_ops: list = []

        # Propagate cascade_on_drop from each ViewRecord so both replace
        # and drop ops honor the model's cascade preference. Missing
        # entries default to True (behavior-preserving).
        cascade_by_name = {
            (vr.name, vr.schema): vr.cascade_on_drop for vr in schema_records
        }

        # Regular views
        create_ops.extend(
            _diff_views(
                model_view_defs, db_views, schema,
                is_materialized=False, cascade_by_name=cascade_by_name,
            )
        )
        # Only drop views that are genuinely in the DB but NOT in the
        # model. Views in `skipped` failed canonicalization and must NOT be
        # dropped — they are still modeled, just not canonicalizable right now.
        _collect_drop_ops(
            db_views, model_view_defs, skipped, schema, cascade_by_name,
            DropViewOp, "view", connection, drop_ops,
        )

        # Materialized views
        create_ops.extend(
            _diff_views(
                model_mv_defs, db_mvs, schema,
                is_materialized=True, cascade_by_name=cascade_by_name,
            )
        )
        _collect_drop_ops(
            db_mvs, model_mv_defs, skipped, schema, cascade_by_name,
            DropMaterializedViewOp, "materialized view", connection, drop_ops,
        )

        all_create_ops.extend(create_ops)
        all_drop_ops.extend(drop_ops)

    # Order by dependency — call once with ALL ops from ALL schemas so
    # cross-schema view-on-view dependencies are resolved correctly.
    if all_create_ops:
        upgrade_ops.ops.extend(
            _order_ops(
                all_create_ops,
                model_records,
                all_db,
                resolve_create_order,
                "creating",
                dialect=connection.dialect,
            )
        )

    if all_drop_ops:
        upgrade_ops.ops.extend(
            _order_ops(
                all_drop_ops,
                model_records,
                all_db,
                resolve_drop_order,
                "dropping",
                dialect=connection.dialect,
            )
        )

    # Cross-type conflict resolution: when a view changes from regular to
    # materialized (or vice versa), the above per-schema loops emit BOTH a
    # Create op (new type) and a Drop op (old type). Because creates are
    # extended before drops, the migration would CREATE while the old-type
    # view still exists, which fails. For any view name that has BOTH a
    # create-family op and a drop-family op, move the drop before the create.
    upgrade_ops.ops = _reorder_cross_type_drops_before_creates(upgrade_ops.ops)

    seen: set = set()
    deduped: list = []
    for op in upgrade_ops.ops:
        # Only view ops participate in dedup. Non-view ops (Alembic
        # built-ins like CreateTableOp) and RefreshMaterializedViewOp
        # (which is neither create- nor drop-family) pass through here
        # unchanged — refresh ops are idempotent side effects, not state
        # transitions that conflict with create/drop.
        if not (_is_create_family(op) or _is_drop_family(op)):
            deduped.append(op)
            continue
        # Normalize op type to a family prefix (create/replace/drop)
        # so conflicting ops for the same view are deduped.
        if _is_create_family(op):
            op_family = "create_or_replace"
        else:
            op_family = "drop"
        key = (op_family, getattr(op, "name", None), getattr(op, "schema", None))
        if key not in seen:
            seen.add(key)
            deduped.append(op)
    upgrade_ops.ops = deduped


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

        from sqlalchemy_utils import register_view_comparator
        register_view_comparator()

    This function is idempotent (safe to call more than once).  The
    comparator is registered explicitly — merely importing an Op class from
    :mod:`sqlalchemy_utils.alembic` does **not** activate autogenerate.
    """
    global _registered
    if _registered:
        return
    from . import renderer  # noqa: F401
    comparators.dispatch_for("schema")(compare_views)
    _registered = True
