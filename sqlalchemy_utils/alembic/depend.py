"""Topological sort for view-on-view dependency ordering.

Provides ``resolve_create_order`` and ``resolve_drop_order`` which sort
:class:`~sqlalchemy_utils.view_record.ViewRecord` instances so that
views are created / dropped in a safe order even when they reference each
other.

Dependency detection uses word-boundary matching against the stringified
view definition.  Cross-schema name matching is supported at the
comparator level (all schemas' view names are pooled); SQL-AST parsing
is not yet implemented.
"""
from __future__ import annotations

import re
from graphlib import TopologicalSorter, CycleError

import sqlalchemy as sa

from sqlalchemy_utils.view_record import ViewRecord


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_dependency_graph(
    view_records: list[ViewRecord],
    db_views: dict[str, str] | None,
    *,
    dialect: sa.engine.Dialect | None = None,
) -> dict[tuple[str, str | None], set[tuple[str, str | None]]]:
    """Build ``{(name, schema): {(dep_name, dep_schema), ...}}`` by
    word-boundary matching each view's definition against other view names
    (model + DB).  *db_views* names are potential dependencies but excluded
    from sort output.  *dialect* is forwarded to ``compiled_definition`` so
    scanning matches emitted SQL.

    Graph keys are ``(name, schema)`` tuples so that cross-schema same-name
    views remain distinct nodes.  Regex matching stays bare-name based (a
    known limitation); when a matched bare name maps to multiple model views
    in different schemas, ALL of them are added as dependency targets.
    """
    if db_views is None:
        db_views = {}
    model_names: set[str] = {vr.name for vr in view_records}
    db_names: set[str] = set(db_views.keys()) - model_names  # only DB-only
    all_known = model_names | db_names

    # Map bare name -> set of (name, schema) keys for model views.
    # DB-only views have no schema, so they stay as bare-name targets that
    # are filtered out of the result by _toposort.
    name_to_keys: dict[str, set[tuple[str, str | None]]] = {}
    for vr in view_records:
        name_to_keys.setdefault(vr.name, set()).add((vr.name, vr.schema))

    graph: dict[tuple[str, str | None], set[tuple[str, str | None]]] = {}

    for vr in view_records:
        definition = vr.compiled_definition(dialect=dialect)
        deps: set[tuple[str, str | None]] = set()
        for other_name in all_known:
            if other_name == vr.name:
                continue  # skip self-reference
            if re.search(rf"\b{re.escape(other_name)}\b", definition):
                # Resolve the matched bare name to graph keys. For model
                # views, add all (name, schema) variants. For DB-only views,
                # other_name is not in name_to_keys, so we add the bare name
                # as a placeholder edge target (filtered out of the result
                # by _toposort since it has no matching ViewRecord).
                matched_keys = name_to_keys.get(other_name)
                if matched_keys is not None:
                    deps.update(matched_keys)
                else:
                    deps.add((other_name, None))
        graph.setdefault((vr.name, vr.schema), set()).update(deps)

    return graph


def _records_by_key(
    view_records: list[ViewRecord],
) -> dict[tuple[str, str | None], list[ViewRecord]]:
    """Return ``{(name, schema): [ViewRecord, ...]}`` preserving all records."""
    result: dict[tuple[str, str | None], list[ViewRecord]] = {}
    for vr in view_records:
        result.setdefault((vr.name, vr.schema), []).append(vr)
    return result


def _reverse_edges(
    graph: dict[tuple[str, str | None], set[tuple[str, str | None]]],
) -> dict[tuple[str, str | None], set[tuple[str, str | None]]]:
    """Return a new graph with every edge reversed.

    *graph* maps ``{node: {predecessor, ...}}`` (node depends on its
    predecessors).  The returned graph swaps the direction of each edge so
    that ``TopologicalSorter`` on it yields dependents before dependencies.
    """
    reversed_graph: dict[tuple[str, str | None], set[tuple[str, str | None]]] = {
        node: set() for node in graph
    }
    for node, predecessors in graph.items():
        for predecessor in predecessors:
            reversed_graph.setdefault(predecessor, set()).add(node)
    return reversed_graph


def _toposort(
    view_records: list[ViewRecord],
    db_views: dict[str, str] | None,
    *,
    reverse: bool = False,
    dialect: sa.engine.Dialect | None = None,
) -> list[ViewRecord]:
    """Core topological sort with cycle detection.

    *reverse* returns drop order (dependents first).  *dialect* is forwarded
    to ``compiled_definition``.  Raises ``ValueError`` on cycles."""
    if db_views is None:
        db_views = {}
    graph = _build_dependency_graph(view_records, db_views, dialect=dialect)
    sort_graph = _reverse_edges(graph) if reverse else graph
    sorter = TopologicalSorter(sort_graph)

    try:
        # ``static_order()`` returns an iterator; ``prepare()`` would also
        # catch cycles but ``static_order`` is the convenient public API.
        sorted_keys = list(sorter.static_order())
    except CycleError as exc:
        # exc.args is ``(message, cycle_nodes)`` where ``cycle_nodes`` is a
        # list like ``[('view_a', None), ('view_b', None), ('view_a', None)]``.
        # Format it as a readable chain for a helpful error message.
        cycle_nodes = exc.args[1] if len(exc.args) > 1 else exc.args
        if cycle_nodes:
            cycle_chain = " -> ".join(
                f"{n[0]}.{n[1]}" if isinstance(n, tuple) and n[1] else str(n)
                for n in cycle_nodes
            )
            msg = f"Circular dependency detected among views: {cycle_chain}"
        else:
            msg = "Circular dependency detected among views"
        raise ValueError(msg) from exc

    key_to_record = _records_by_key(view_records)

    # Filter out any keys that are only in db_views (not in model records)
    result: list[ViewRecord] = []
    for key in sorted_keys:
        if key in key_to_record:
            result.extend(key_to_record[key])
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def resolve_create_order(
    view_records: list[ViewRecord],
    db_views: dict[str, str] | None,
    *,
    dialect: sa.engine.Dialect | None = None,
) -> list[ViewRecord]:
    """Sort *view_records* so that dependencies come before dependents.

    This is the order in which views should be **created** (or recreated
    during a migration).

    :param view_records: List of :class:`ViewRecord` instances
        representing the desired views.
    :param db_views: Mapping of ``{view_name: sql_definition}`` for views
        that already exist in the database.  These are treated as
        pre-satisfied dependencies — a model view may depend on a DB view,
        but the DB view is not included in the output.
    :param dialect: Optional SQLAlchemy dialect forwarded to
        :meth:`ViewRecord.compiled_definition` so dependency detection
        scans the same dialect-qualified SQL the comparator emits.  When
        *None*, default compilation is used.
    :returns: Views in safe creation order.
    :raises ValueError: If a circular dependency is detected.

    Example::

        >>> ordered = resolve_create_order(view_records, db_views)
        # ordered[0] has no dependencies; ordered[-1] depends on others
    """
    return _toposort(view_records, db_views, reverse=False, dialect=dialect)


def resolve_drop_order(
    view_records: list[ViewRecord],
    db_views: dict[str, str] | None,
    *,
    dialect: sa.engine.Dialect | None = None,
) -> list[ViewRecord]:
    """Sort *view_records* so that dependents come before dependencies.

    This is the **reverse** of :func:`resolve_create_order` — views that
    depend on others are dropped first, so no dangling references remain.

    :param view_records: List of :class:`ViewRecord` instances.
    :param db_views: Mapping of ``{view_name: sql_definition}`` for
        existing database views.  Used for dependency detection but not
        included in output.
    :param dialect: Optional SQLAlchemy dialect forwarded to
        :meth:`ViewRecord.compiled_definition` so dependency detection
        scans the same dialect-qualified SQL the comparator emits.  When
        *None*, default compilation is used.
    :returns: Views in safe drop order.
    :raises ValueError: If a circular dependency is detected.

    Example::

        >>> ordered = resolve_drop_order(view_records, db_views)
        # ordered[0] is a dependent view; ordered[-1] has no dependents
    """
    return _toposort(view_records, db_views, reverse=True, dialect=dialect)
