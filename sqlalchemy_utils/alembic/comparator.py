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

Differences are emitted as :class:`~sqlalchemy_utils.alembic.operations.CreateViewOp`,
:class:`~sqlalchemy_utils.alembic.operations.DropViewOp`,
:class:`~sqlalchemy_utils.alembic.operations.ReplaceViewOp` (and their materialized-view equivalents).

Usage in ``env.py``::

    from sqlalchemy_utils import register_view_comparator
    register_view_comparator()   # must be called before context.configure()
"""
from __future__ import annotations

import logging

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

_VIEW_OP_TYPES: tuple[type, ...] = (
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
# ``DBAPIError`` is a subclass of ``SQLAlchemyError`` so listing both is
# redundant; programming errors (``TypeError``, etc.) must propagate.
_CANON_ERRORS = (sa.exc.SQLAlchemyError, OSError)

# Idempotency guard: register_view_comparator may be called multiple times
# (e.g. across env.py reloads); only the first call registers the hook.
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
    (a failed canonicalization must NOT produce a false DropViewOp).

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
            # Inner savepoint per view: on success RELEASE (view persists in
            # the outer savepoint so dependents can see it); on failure
            # ROLLBACK TO (cleans the aborted sub-transaction without
            # touching already-created views). Releasing avoids savepoint
            # accumulation.
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
                    # ROLLBACK TO does not destroy the savepoint (PG
                    # semantics); RELEASE it so the next iteration can
                    # create a fresh one.
                    connection.execute(
                        sa.text(f"RELEASE SAVEPOINT {inner_sp}")
                    )
                except _CANON_ERRORS:
                    pass
                skipped.add(vr.name)

                # Probe: is the outer savepoint still usable? A DB-level
                # error may have poisoned the transaction, in which case
                # every subsequent SAVEPOINT will fail and every remaining
                # view would be silently skipped. Break early with a
                # warning instead.
                try:
                    connection.execute(sa.text("SELECT 1"))
                except _CANON_ERRORS:
                    log.warning(
                        "Outer savepoint is in aborted state after failing to "
                        "canonicalize view '%s'; aborting canonicalization of "
                        "remaining views", vr.name
                    )
                    processed.add(vr.name)
                    # Unreached views must land in `skipped` so drop
                    # detection does not treat them as DB-only and emit a
                    # false DropViewOp (data loss).
                    for remaining_vr in ordered:
                        if remaining_vr.name not in processed:
                            skipped.add(remaining_vr.name)
                    break
            processed.add(vr.name)

        # Batch-read canonical definitions from pg_catalog in one query per
        # kind (regular + materialized) rather than one per view. This
        # readback happens INSIDE the outer savepoint (before the
        # finally-block rollback) so the just-created views are visible.
        db_views: dict[str, str] = {}
        db_mvs: dict[str, str] = {}
        try:
            db_views = get_database_views(connection, schema)
            db_mvs = get_database_materialized_views(connection, schema)
        except _CANON_ERRORS as exc:
            # The catalog readback itself can fail if the transaction was
            # poisoned mid-loop (the inner-savepoint probe in the loop only
            # fires after a view CREATE failure; a DB-level abort that
            # poisons the outer savepoint without a per-view failure would
            # land here). Treat as poisoned and skip readback.
            log.warning(
                "Catalog readback failed for schema %r (transaction may be "
                "poisoned): %s. Marking all views skipped.", schema, exc
            )
            for vr in ordered:
                skipped.add(vr.name)
    finally:
        # Roll back the outer savepoint so nothing persists.  Guarded
        # because if the transaction is already poisoned (e.g. a DB-level
        # error aborted the outer savepoint itself) the ROLLBACK TO would
        # raise and mask the original exception propagated from the body.
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


def _diff_views(
    model_defs: dict[str, str],
    db_defs: dict[str, str],
    schema: str | None,
    is_materialized: bool,
    cascade_by_name: dict[tuple[str, str | None], bool] | None = None,
) -> list:
    """Diff model view definitions against DB state, returning create/replace ops.

    For each model view name: if absent from the DB, emit a Create op; if the
    canonical definition differs from the DB, emit a Replace op. The op class
    (``CreateViewOp``/``ReplaceViewOp`` vs ``CreateMaterializedViewOp``/
    ``ReplaceMaterializedViewOp``) is selected by *is_materialized*.

    Materialized-view ops are created with ``with_data=False`` (matches the
    previous inline behavior — autogenerate emits ``WITH NO DATA``).

    ``cascade_by_name`` is consulted for materialized-view Create and
    Replace ops so the user's ``cascade_on_drop`` preference propagates
    from the model to the emitted ``CreateMaterializedViewOp`` /
    ``ReplaceMaterializedViewOp``. Regular-view Create ops also consult
    ``cascade_by_name`` (default ``True``) so a model view created during
    autogenerate honors the ``cascade_on_drop`` preference. Regular-view
    Replace ops do not carry a ``cascade`` field (``CREATE OR REPLACE
    VIEW`` does not drop). Missing entries default to ``True``
    (behavior-preserving).
    """
    if cascade_by_name is None:
        cascade_by_name = {}
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
    except Exception as exc:
        log.warning(
            "Failed to query dependent views for %r: %s", name, exc
        )
        dependents = {}
    if dependents:
        formatted = []
        for k in sorted(dependents):
            if isinstance(k, tuple) and len(k) == 2:
                dep_name, dep_schema = k
                formatted.append(
                    f"{dep_schema}.{dep_name}" if dep_schema else dep_name
                )
            else:
                formatted.append(str(k))
        log.warning(
            "Dropping %s %r which has %d dependent view(s): %s. "
            "CASCADE will drop them automatically. "
            "Remove the dependent views from your model first if this is unintended.",
            kind_label,
            name,
            len(dependents),
            ", ".join(formatted),
        )


def _safe_resolve(records, db, resolver_fn, action_label, *, dialect=None):
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
    """Reorder so a Drop for a view name precedes a Create for the same name.

    When a view changes type (regular <-> materialized) the comparator emits
    a Create-family op for the new type and a Drop-family op for the old
    type. Creates are appended before drops, so without reordering the
    migration would CREATE while the old-type view still exists (which
    fails). For any (name, schema) present in BOTH a drop and a create op,
    the drop is moved immediately before the create op for that key.

    The drop+create pair for each cross-type key is inserted AT THE POSITION
    where the FIRST cross-type op for that key originally appeared in
    ``ops`` (rather than buffering all cross-type ops to the end of the
    result). This preserves the relative dependency ordering: a non-
    cross-type op that was positioned after a cross-type op stays after it,
    so a dependent view's create is not reordered before its dependency's
    new type exists (which would cause a cascade-drop to silently drop the
    dependent view).
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

    # Walk `ops` in order so cross-type keys are emitted in the order they
    # FIRST appear. Iterating `cross_keys` (a set) would give non-deterministic
    # order under Python hash randomization, which can reorder drop+create
    # pairs of dependent views and break migrations.
    buffered: dict = {}
    first_position: dict = {}
    cross_order: list = []
    for idx, op in enumerate(ops):
        key = (getattr(op, "name", None), getattr(op, "schema", None))
        if key not in cross_keys:
            continue
        buffered.setdefault(key, []).append(op)
        if key not in first_position:
            first_position[key] = idx
            cross_order.append(key)

    result: list = []
    inserted: set = set()
    for idx, op in enumerate(ops):
        for key in cross_order:
            if key in inserted:
                continue
            if first_position[key] != idx:
                continue
            drops = [o for o in buffered[key] if _is_drop_family(o)]
            creates = [o for o in buffered[key] if _is_create_family(o)]
            result.extend(drops)
            result.extend(creates)
            inserted.add(key)
        key = (getattr(op, "name", None), getattr(op, "schema", None))
        if key in cross_keys:
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


def compare_views(
    autogen_context: AutogenContext,
    upgrade_ops: UpgradeOps,
    schemas: list[str | None] | None = None,
) -> None:
    """Compare model-defined views against database state.

    This function is registered as an Alembic ``"schema"`` comparator and is
    called automatically during ``alembic revision --autogenerate``.

    It reads view definitions from ``metadata.info['sqlalchemy_utils_views']``
    (populated by `create_view()`, `create_materialized_view()`, and
    `ViewMixin.__declare_last__`), canonicalizes each model view via
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
            connection, schema_records, all_db
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
        for name in db_views:
            if name in model_view_defs or name in skipped:
                continue
            drop_ops.append(
                DropViewOp(
                    name,
                    schema=schema,
                    cascade=cascade_by_name.get((name, schema), True),
                    definition=db_views[name],
                )
            )
            _warn_if_dependents(connection, name, schema, "view")

        # Materialized views
        create_ops.extend(
            _diff_views(
                model_mv_defs, db_mvs, schema,
                is_materialized=True, cascade_by_name=cascade_by_name,
            )
        )
        for name in db_mvs:
            if name in model_mv_defs or name in skipped:
                continue
            drop_ops.append(
                DropMaterializedViewOp(
                    name,
                    schema=schema,
                    cascade=cascade_by_name.get((name, schema), True),
                    definition=db_mvs[name],
                )
            )
            _warn_if_dependents(connection, name, schema, "materialized view")

        # Order by dependency
        if create_ops:
            upgrade_ops.ops.extend(
                _order_ops(
                    create_ops,
                    model_records,
                    all_db,
                    resolve_create_order,
                    "creating",
                    dialect=connection.dialect,
                )
            )

        if drop_ops:
            upgrade_ops.ops.extend(
                _order_ops(
                    drop_ops,
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
        # built-ins like CreateTableOp) lack a ``name`` attribute, so
        # getattr returns None for all of them — without this guard every
        # non-view op collides on ("create_or_replace"|"drop", None, None)
        # and all but the first are silently discarded.
        if not isinstance(op, _VIEW_OP_TYPES):
            deduped.append(op)
            continue
        # Normalize op type to a family prefix (create/replace/drop)
        # so conflicting ops for the same view are deduped. Refresh ops
        # are excluded from dedup entirely — they are idempotent side
        # effects, not state transitions that conflict with create/drop.
        if isinstance(op, RefreshMaterializedViewOp):
            deduped.append(op)
            continue
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
    comparator is registered lazily — merely importing an Op class from
    :mod:`sqlalchemy_utils.alembic` does **not** activate autogenerate.
    """
    global _registered
    if _registered:
        return
    from . import comparator, operations  # noqa: F401
    comparators.dispatch_for("schema")(compare_views)
    try:
        from . import renderer  # noqa: F401
    except ImportError as exc:
        log.warning(
            "Failed to import renderer module; autogenerate will detect but "
            "not render view operations: %s",
            exc,
        )
    _registered = True
