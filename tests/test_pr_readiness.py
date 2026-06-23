"""
PR-readiness regression tests for bugs A-K.

Each test in this file is intentionally written to FAIL against the current
(unfixed) source code, proving each bug is real. The tests will start passing
once the corresponding bug is fixed in a later wave.

Run:
    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \\
        tests/test_pr_readiness.py -v
"""
from __future__ import annotations

import inspect
from unittest import mock

import pytest
import sqlalchemy as sa

from sqlalchemy_utils.alembic.comparator import (
    _canonicalize_view,
    _schema_matches,
    compare_views,
)
from sqlalchemy_utils.alembic.depend import _build_dependency_graph
from sqlalchemy_utils.alembic.operations import (
    CreateViewOp,
    DropViewOp,
    _create_view_impl,
)
from sqlalchemy_utils.alembic.pg_catalog import get_database_views
from sqlalchemy_utils.alembic.view_record import ViewRecord
from sqlalchemy_utils.view_mixin import ViewMixin


# ---------------------------------------------------------------------------
# BUG-A: ViewMixin.refresh() does not resolve schema from __table_args__
#        when __view_schema__ is not set.
#
# When a view model declares schema via __table_args__={'schema': 'analytics'}
# but does NOT set __view_schema__, refresh() passes schema=None to
# refresh_materialized_view instead of the resolved 'analytics'.
# ---------------------------------------------------------------------------
def test_refresh_uses_resolved_schema_from_table_args():
    """ViewMixin.refresh must use the schema resolved from __table_args__
    when __view_schema__ is not explicitly set."""

    class AnalyticsView(ViewMixin):
        __tablename__ = 'analytics_mv'
        __view_selectable__ = sa.select(sa.column('id', sa.Integer))
        __view_materialized__ = True
        # __view_schema__ intentionally NOT set — schema comes from table_args
        __table_args__ = {'schema': 'analytics'}
        metadata = sa.MetaData()
        id = sa.Column(sa.Integer, primary_key=True)

    session = mock.MagicMock(name='session')

    with mock.patch(
        'sqlalchemy_utils.view_mixin.refresh_materialized_view'
    ) as mock_refresh:
        AnalyticsView.refresh(session)

    mock_refresh.assert_called_once()
    _, kwargs = mock_refresh.call_args
    # BUG-A: refresh() passes cls.__view_schema__ (which is None) instead of
    # the schema resolved from __table_args__.
    assert kwargs.get('schema') == 'analytics', (
        "ViewMixin.refresh() should resolve schema from __table_args__ when "
        "__view_schema__ is not set, but got "
        f"schema={kwargs.get('schema')!r}"
    )


# ---------------------------------------------------------------------------
# BUG-B: Alembic operations do not quote identifiers (reserved words break).
#
# _schema_prefix() and the operation implementations use bare f-strings like
# f"{prefix}{op.name}" without running the identifier through the dialect's
# identifier_preparer.quote(). A view named "order" (a SQL reserved word)
# produces invalid SQL.
# ---------------------------------------------------------------------------
def test_operations_quote_identifiers():
    """The operations module must use identifier_preparer.quote() for both
    schema and name, so reserved words like 'order' produce valid SQL.

    This test inspects the source of _create_view_impl and asserts that
    quoting is applied. (The old _schema_prefix helper has been removed;
    quoting is now done via _qualified_name / _quote_identifier.)
    """
    src = inspect.getsource(_create_view_impl)

    # The implementation should call identifier_preparer.quote() (or equivalent)
    # on the view name. Currently it just embeds {op.name} directly.
    assert 'quote' in src, (
        "_create_view_impl must quote identifiers via "
        "identifier_preparer.quote() to support reserved-word view names "
        "like 'order'. Source does not contain 'quote'."
    )


# ---------------------------------------------------------------------------
# BUG-C: _canonicalize_view does not quote identifiers when building the
#        CREATE VIEW / CREATE MATERIALIZED VIEW SQL.
#
# The function builds SQL via f-strings using {prefix}{view_record.name}
# without running through identifier_preparer.quote().
# ---------------------------------------------------------------------------
def test_comparator_canonicalize_quotes_identifiers():
    """_canonicalize_view must use identifier_preparer.quote() for schema
    and name when constructing CREATE VIEW SQL."""
    src = inspect.getsource(_canonicalize_view)

    assert 'quote' in src, (
        "_canonicalize_view must use identifier_preparer.quote() for schema "
        "and name identifiers; current source does not contain 'quote'."
    )

    # And it must quote both the schema (prefix) and the view name.
    assert src.count('quote') >= 2, (
        "_canonicalize_view should quote both schema and name (>=2 quote "
        "calls); found fewer."
    )


# ---------------------------------------------------------------------------
# BUG-D: _schema_matches treats None and 'public' as equivalent, causing
#        duplicate processing of views that are genuinely in non-public
#        schemas during autogenerate diff loops scoped to 'public'.
# ---------------------------------------------------------------------------
def test_no_duplicate_ops_for_none_public_schemas():
    """_schema_matches uses exact match only; None != 'public'.

    Previously a view with schema=None matched BOTH a None-schema loop AND
    a 'public'-schema loop, leading to duplicate operations during
    autogenerate when schemas=[None, 'public'] was iterated. With the fix,
    _schema_matches is an exact equality check, so None matches only None.
    """
    # None view-schema no longer matches 'public' loop-schema — bug fixed.
    assert _schema_matches(None, 'public') is False, (
        "BUG-D: _schema_matches(None, 'public') should be False after fix; "
        "treating None and 'public' as equivalent causes duplicate ops."
    )

    # None still matches None (same schema).
    assert _schema_matches(None, None) is True, (
        "Expected _schema_matches(None, None) == True (exact match)."
    )

    # 'public' matches 'public' (exact match).
    assert _schema_matches('public', 'public') is True, (
        "Expected _schema_matches('public', 'public') == True (exact match)."
    )

    # 'analytics' does not match 'public'.
    assert _schema_matches('analytics', 'public') is False, (
        "Expected _schema_matches('analytics', 'public') == False (exact match)."
    )

    # A view with schema=None is processed ONLY in the None loop, not the
    # 'public' loop — no duplicate ops.
    both_match = (
        _schema_matches(None, None) and _schema_matches(None, 'public')
    )
    assert not both_match, (
        "BUG-D: _schema_matches(None, None) and _schema_matches(None, "
        "'public') both return True, causing duplicate ops when "
        "schemas=[None, 'public'] is iterated."
    )


# ---------------------------------------------------------------------------
# BUG-E: The materialized-view index-creation listener is scoped to the
#        whole metadata instead of to the specific MV's table, so it fires
#        for every table.create_all() call and can raise "index already
#        exists" errors when multiple MVs share a metadata.
# ---------------------------------------------------------------------------
def test_index_listener_scoped_to_table():
    """The create_indexes listener in create_materialized_view must be scoped
    to the MV's table (via sa.event.listens_for(table, 'after_create')), not
    to the whole metadata. Currently it listens on metadata, so it fires for
    EVERY table/MV created via that metadata, re-attempting index creation
    and raising 'index already exists' when multiple objects share metadata.
    """
    from sqlalchemy_utils.view import create_materialized_view

    src = inspect.getsource(create_materialized_view)

    # The bug: the listener is registered via
    #   @sa.event.listens_for(metadata, 'after_create')
    # without a guard, so it fires for every table created via that metadata.
    # A correct implementation either listens on the table directly OR guards
    # the listener body with a target identity check.
    #
    # We accept either: listens on table, or has a target-is-table guard.
    assert "listens_for(table," in src or "listens_for(table " in src or "target is not table" in src or "target is table" in src, (
        "BUG-E: create_materialized_view's create_indexes listener is "
        "registered on `metadata` (not the MV's `table`), so it fires for "
        "every table created via that metadata and re-creates the MV "
        "indexes, raising 'index already exists'. Expected "
        "'listens_for(table, ...)' in source."
    )


# ---------------------------------------------------------------------------
# BUG-F: get_database_views hardcodes 'public' when schema=None instead of
#        querying ALL non-system schemas.
# ---------------------------------------------------------------------------
def test_pg_catalog_returns_all_schemas_when_none():
    """When schema=None, get_database_views should query all non-system
    schemas, not just 'public'. Currently the SQL clause is:
        (:schema IS NULL AND schemaname = 'public')
    which limits None to 'public' only.
    """
    src = inspect.getsource(get_database_views)

    # The bug: the SQL contains "schemaname = 'public'" as the None branch.
    # A correct implementation would exclude system schemas (pg_*, information_schema)
    # rather than hardcoding 'public'.
    assert "schemaname = 'public'" not in src, (
        "get_database_views hardcodes 'public' for the schema=None branch; "
        "it should query all non-system schemas instead. Source contains "
        "\"schemaname = 'public'\"."
    )


# ---------------------------------------------------------------------------
# BUG-G: _canonicalize_view does not pass the connection's dialect to
#        sel.compile(), so the canonicalized definition may use the wrong
#        dialect's SQL syntax.
# ---------------------------------------------------------------------------
def test_comparator_uses_connection_dialect():
    """_canonicalize_view must pass dialect=connection.dialect to
    sel.compile() so the canonical definition matches the actual DB dialect."""
    src = inspect.getsource(_canonicalize_view)

    assert 'connection.dialect' in src or 'dialect=connection' in src, (
        "_canonicalize_view must pass dialect=connection.dialect (or "
        "dialect=connection) to sel.compile(); current source does not."
    )


# ---------------------------------------------------------------------------
# BUG-H: _build_dependency_graph uses word-boundary regex matching that
#        treats SQL keywords-as-view-names (e.g. a view named "user") as
#        matching any definition containing the word "user", creating
#        false dependencies.
# ---------------------------------------------------------------------------
def test_depend_regex_skips_sql_keywords():
    """A view named 'user' must not be flagged as a dependency of every
    view whose definition happens to contain the word 'user' as a SQL
    keyword (e.g. 'CREATE USER ...'). Currently it does."""
    # Two views: one named 'user', one named 'data' with a definition that
    # contains the standalone word 'user' as a column alias, NOT a view ref.
    user_view = ViewRecord(
        name='user',
        selectable='SELECT 1 AS id',
        schema=None,
        materialized=False,
    )
    data_view = ViewRecord(
        name='data',
        # 'user' appears here as a column alias (standalone word), not as a
        # view reference. The regex \buser\b matches it anyway.
        selectable='SELECT account_id AS user FROM accounts',
        schema=None,
        materialized=False,
    )

    graph = _build_dependency_graph([user_view, data_view], {})

    # 'data' should NOT depend on 'user' — the word 'user' is a column
    # alias here, not a reference to the 'user' view. The regex can't tell
    # the difference, so it creates a false dependency.
    assert 'user' not in graph.get('data', set()), (
        "BUG-H: _build_dependency_graph falsely reports 'user' as a "
        f"dependency of 'data' (graph={graph!r}). The regex matches the "
        "standalone word 'user' even when it's a column alias, not a view "
        "reference."
    )


# ---------------------------------------------------------------------------
# BUG-I: compare_views only collects db_views/db_mvs for the CURRENT schema
#        loop iteration before calling resolve_create_order, so cross-schema
#        view dependencies (a view in schema A depending on a view in schema B)
#        are not resolved and may produce a wrong creation order.
# ---------------------------------------------------------------------------
def test_cross_schema_dependency_resolution():
    """compare_views must collect all_db across ALL schemas before calling
    resolve_create_order, so cross-schema dependencies are visible to the
    topological sort. Currently all_db only contains the current schema's
    views."""
    src = inspect.getsource(compare_views)

    # The bug: resolve_create_order is called with `all_db = {**db_views, **db_mvs}`
    # where db_views/db_mvs are scoped to the CURRENT schema loop only.
    # A correct implementation would aggregate across all schemas first.
    #
    # We check that the source does NOT build a cross-schema aggregate before
    # calling resolve_create_order. Specifically, the call should pass an
    # accumulator rather than a per-schema dict.
    assert 'all_schemas' in src or 'all_db_views' in src or 'cross_schema' in src, (
        "compare_views should aggregate db views across ALL schemas before "
        "calling resolve_create_order to support cross-schema dependencies; "
        "current source only uses the current-schema db_views/db_mvs."
    )


# ---------------------------------------------------------------------------
# BUG-J: ViewMixin does not define __view_replace__ as a class attribute,
#        so getattr(cls, '__view_replace__', False) always returns the
#        default False even when a subclass sets it via __view_replace__ = True.
#        (Actually the deeper bug: the attribute is missing from the class
#        body, so hasattr(ViewMixin, '__view_replace__') is False.)
# ---------------------------------------------------------------------------
def test_view_replace_is_class_attribute():
    """ViewMixin must define __view_replace__ as a class attribute (default
    False) so that subclasses can override it and the code path
    getattr(cls, '__view_replace__', False) finds the class attribute."""
    assert hasattr(ViewMixin, '__view_replace__'), (
        "BUG-J: ViewMixin does not define __view_replace__ as a class "
        "attribute; hasattr(ViewMixin, '__view_replace__') is False."
    )
    assert ViewMixin.__view_replace__ is False, (
        "BUG-J: ViewMixin.__view_replace__ should default to False."
    )


# ---------------------------------------------------------------------------
# BUG-K: ViewRecord does not implement a definition_matches() method to
#        compare its selectable against a database definition string.
#        Autogenerate relies on string equality, which is fragile.
# ---------------------------------------------------------------------------
def test_viewrecord_definition_matches_method():
    """ViewRecord must expose a definition_matches() method that compares
    the view's selectable against a database definition string in a
    dialect-aware, normalization-tolerant way."""
    assert hasattr(ViewRecord, 'definition_matches'), (
        "BUG-K: ViewRecord does not define a definition_matches() method; "
        "hasattr(ViewRecord, 'definition_matches') is False."
    )
    assert callable(getattr(ViewRecord, 'definition_matches')), (
        "BUG-K: ViewRecord.definition_matches exists but is not callable."
    )
