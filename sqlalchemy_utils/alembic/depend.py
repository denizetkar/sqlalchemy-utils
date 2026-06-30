"""Topological sort for view-on-view dependency ordering.

Provides ``resolve_create_order`` and ``resolve_drop_order`` which sort
:class:`~sqlalchemy_utils.view_record.ViewRecord` instances so that
views are created / dropped in a safe order even when they reference each
other.

Dependency detection uses simple word-boundary matching (``\\b{name}\\b``)
against the stringified view definition.  This is intentionally v1 — no
cross-schema or SQL-AST parsing yet.
"""
from __future__ import annotations

import re
from graphlib import TopologicalSorter, CycleError

from sqlalchemy_utils.view_record import ViewRecord


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Common SQL keywords / identifiers skipped during view-name dependency
# matching to avoid false positives (e.g. a view named ``user`` matching the
# column alias ``AS user`` in another view's definition).
_SQL_KEYWORDS = frozenset({
    'select', 'from', 'where', 'as', 'and', 'or', 'not', 'in', 'is',
    'join', 'on', 'group', 'order', 'by', 'having', 'limit', 'offset',
    'user', 'id', 'name', 'data', 'table', 'column', 'view', 'create',
    'drop', 'insert', 'update', 'delete', 'set', 'values', 'into',
    'with', 'case', 'when', 'then', 'else', 'end', 'null', 'true', 'false',
    'count', 'sum', 'avg', 'min', 'max', 'distinct', 'all', 'any',
    'asc', 'desc', 'union', 'intersect', 'except', 'exists', 'between',
    'like', 'inner', 'left', 'right', 'outer', 'cross', 'using', 'natural',
    'cast', 'coalesce', 'nullif', 'over', 'partition', 'row', 'rows',
    'schema', 'index', 'sequence', 'function', 'procedure', 'trigger',
    'foreign', 'primary', 'key', 'unique', 'check', 'default', 'constraint',
})


def _definition_str(view_record: ViewRecord) -> str:
    """Return the SQL definition string for *view_record*.

    If ``selectable`` is already a string it is returned as-is; otherwise
    it is compiled with ``literal_binds`` (matching
    :meth:`ViewRecord._selectable_key`).
    """
    sel = view_record.selectable
    if isinstance(sel, str):
        return sel
    return str(sel.compile(compile_kwargs={"literal_binds": True}))


def _build_dependency_graph(
    view_records: list[ViewRecord],
    db_views: dict[str, str] | None,
) -> dict[str, set[str]]:
    """Build a ``{name: {dep_name, ...}}`` mapping.

    For every view in *view_records*, we scan its definition for references
    to **other** view names (from *view_records* or *db_views*) using
    word-boundary regex matching.

    *db_views* represents views that already exist in the database — they
    are potential dependencies but are NOT included in the sort output.
    """
    if db_views is None:
        db_views = {}
    # All known view names (model + DB) for matching
    model_names: set[str] = {vr.name for vr in view_records}
    db_names: set[str] = set(db_views.keys()) - model_names  # only DB-only
    all_known = model_names | db_names

    graph: dict[str, set[str]] = {}

    for vr in view_records:
        definition = _definition_str(vr)
        deps: set[str] = set()
        for other_name in all_known:
            if other_name == vr.name:
                continue  # skip self-reference
            if other_name.lower() in _SQL_KEYWORDS:
                continue  # skip SQL keywords / common words
            if re.search(rf"\b{re.escape(other_name)}\b", definition):
                deps.add(other_name)
        graph[vr.name] = deps

    return graph


def _records_by_name(
    view_records: list[ViewRecord],
) -> dict[str, list[ViewRecord]]:
    """Return a ``{name: [ViewRecord, ...]}`` lookup preserving all records
    with the same name (e.g. same name in different schemas)."""
    result: dict[str, list[ViewRecord]] = {}
    for vr in view_records:
        result.setdefault(vr.name, []).append(vr)
    return result


def _toposort(
    view_records: list[ViewRecord],
    db_views: dict[str, str] | None,
    *,
    reverse: bool = False,
) -> list[ViewRecord]:
    """Core topological sort with cycle detection.

    Parameters
    ----------
    view_records:
        The model views to sort.
    db_views:
        Current database view definitions (name → SQL).  These are
        considered as pre-existing dependencies.
    reverse:
        If *True*, return drop order (dependents before dependencies).

    Returns
    -------
    list[ViewRecord]
        Sorted view records.

    Raises
    ------
    ValueError
        If a cycle is detected among the view dependencies.
    """
    if db_views is None:
        db_views = {}
    graph = _build_dependency_graph(view_records, db_views)
    sorter = TopologicalSorter(graph)

    try:
        # ``static_order()`` returns an iterator; ``prepare()`` would also
        # catch cycles but ``static_order`` is the convenient public API.
        sorted_names = list(sorter.static_order())
    except CycleError as exc:
        # exc.args typically contains (cycle_nodes..., message) — we
        # extract what we can for a helpful error.
        cycle_info = exc.args if exc.args else ()
        raise ValueError(
            f"Circular dependency detected among views: {cycle_info}"
        ) from exc

    if reverse:
        sorted_names = list(reversed(sorted_names))

    name_to_record = _records_by_name(view_records)

    # Filter out any names that are only in db_views (not in model records)
    result: list[ViewRecord] = []
    for name in sorted_names:
        if name in name_to_record:
            result.extend(name_to_record[name])
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_create_order(
    view_records: list[ViewRecord],
    db_views: dict[str, str] | None,
) -> list[ViewRecord]:
    """Sort *view_records* so that dependencies come before dependents.

    This is the order in which views should be **created** (or recreated
    during a migration).

    Parameters
    ----------
    view_records:
        List of :class:`ViewRecord` instances representing the desired views.
    db_views:
        Mapping of ``{view_name: sql_definition}`` for views that already
        exist in the database.  These are treated as pre-satisfied
        dependencies — a model view may depend on a DB view, but the DB
        view is not included in the output.

    Returns
    -------
    list[ViewRecord]
        Views in safe creation order.

    Raises
    ------
    ValueError
        If a circular dependency is detected.
    """
    return _toposort(view_records, db_views, reverse=False)


def resolve_drop_order(
    view_records: list[ViewRecord],
    db_views: dict[str, str] | None,
) -> list[ViewRecord]:
    """Sort *view_records* so that dependents come before dependencies.

    This is the **reverse** of :func:`resolve_create_order` — views that
    depend on others are dropped first, so no dangling references remain.

    Parameters
    ----------
    view_records:
        List of :class:`ViewRecord` instances.
    db_views:
        Mapping of ``{view_name: sql_definition}`` for existing database
        views.  Used for dependency detection but not included in output.

    Returns
    -------
    list[ViewRecord]
        Views in safe drop order.

    Raises
    ------
    ValueError
        If a circular dependency is detected.
    """
    return _toposort(view_records, db_views, reverse=True)
