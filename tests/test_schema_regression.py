"""
Regression tests for validated schema-parameter and related bugs.

These tests document bugs that exist in the current codebase. They are
intentionally written to FAIL against the unfixed code, proving each bug
is real. Each test names the bug it covers (Bugs 1-10).

Run:
    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
        tests/test_schema_regression.py -v
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path
from unittest import mock

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from sqlalchemy_utils.alembic.comparator import _schema_matches
from sqlalchemy_utils.alembic.view_record import ViewRecord
from sqlalchemy_utils.view import (
    DropView,
    RefreshMaterializedView,
    create_materialized_view,
    create_view,
    refresh_materialized_view,
)
from sqlalchemy_utils.view_mixin import ViewMixin


TESTS_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Bug 1: create_view() does not accept a schema parameter
# ---------------------------------------------------------------------------
def test_create_view_accepts_schema_param():
    """create_view must accept schema= and propagate it to the view record."""
    metadata = sa.MetaData()
    selectable = sa.select(sa.column('id', sa.Integer))

    # Bug: create_view signature is (name, selectable, metadata,
    #       cascade_on_drop, replace) — no schema kwarg.
    # Expect TypeError until schema is added.
    table = create_view('v', selectable, metadata, schema='analytics')

    view_records = metadata.info.get('sqlalchemy_utils_views', [])
    assert view_records, "Expected a ViewRecord to be registered"
    assert view_records[-1].schema == 'analytics'


# ---------------------------------------------------------------------------
# Bug 2: create_materialized_view() does not accept a schema parameter
# ---------------------------------------------------------------------------
def test_create_materialized_view_accepts_schema_param():
    """create_materialized_view must accept schema= and propagate it."""
    metadata = sa.MetaData()
    selectable = sa.select(sa.column('id', sa.Integer))

    # Bug: create_materialized_view signature is
    #       (name, selectable, metadata, indexes, aliases) — no schema kwarg.
    table = create_materialized_view(
        'mv', selectable, metadata, schema='analytics'
    )

    view_records = metadata.info.get('sqlalchemy_utils_views', [])
    assert view_records, "Expected a ViewRecord to be registered"
    assert view_records[-1].schema == 'analytics'


# ---------------------------------------------------------------------------
# Bug 3: refresh_materialized_view() does not accept a schema parameter
# ---------------------------------------------------------------------------
def test_refresh_materialized_view_accepts_schema_param():
    """refresh_materialized_view must accept schema= without TypeError."""
    session = mock.MagicMock(name='session')

    # Bug: refresh_materialized_view signature is
    #       (session, name, concurrently=False) — no schema kwarg.
    refresh_materialized_view(session, 'mv', schema='analytics')

    # If we got here, no TypeError was raised. Verify the compiled statement
    # carried the schema by inspecting the executed RefreshMaterializedView.
    session.execute.assert_called_once()
    executed = session.execute.call_args.args[0]
    assert isinstance(executed, RefreshMaterializedView)
    assert getattr(executed, 'schema', None) == 'analytics'


def test_refresh_materialized_view_compiler_includes_schema():
    """The RefreshMaterializedView compiler must qualify the view with schema."""
    # Bug: RefreshMaterializedView.__init__ only takes (name, concurrently);
    #       the compiler emits `REFRESH MATERIALIZED VIEW <name>` with no schema.
    element = RefreshMaterializedView(
        'mv', concurrently=False, schema='analytics'
    )
    dialect = postgresql.dialect()
    compiled = str(
        element.compile(dialect=dialect)
    )
    assert compiled == 'REFRESH MATERIALIZED VIEW analytics.mv'


# ---------------------------------------------------------------------------
# Bug 4: ViewMixin.refresh() does not pass schema to refresh_materialized_view
# ---------------------------------------------------------------------------
def test_viewmixin_refresh_passes_schema():
    """ViewMixin.refresh must forward __view_schema__ to refresh_materialized_view."""
    class MySchemaView(ViewMixin):
        __tablename__ = 'my_schema_mv'
        __view_selectable__ = sa.select(sa.column('id', sa.Integer))
        __view_materialized__ = True
        __view_schema__ = 'analytics'
        metadata = sa.MetaData()
        id = sa.Column(sa.Integer, primary_key=True)

    session = mock.MagicMock(name='session')

    with mock.patch(
        'sqlalchemy_utils.view_mixin.refresh_materialized_view'
    ) as mock_refresh:
        MySchemaView.refresh(session)

    mock_refresh.assert_called_once()
    _, kwargs = mock_refresh.call_args
    assert kwargs.get('schema') == 'analytics'


# ---------------------------------------------------------------------------
# Bug 5: ViewRecord.__eq__ ignores the selectable field (known limitation)
# ---------------------------------------------------------------------------
def test_viewrecord_eq_ignores_selectable():
    """Two ViewRecords with same name/schema/materialized but different
    selectable compare equal — documents the known limitation in __eq__."""
    selectable_a = sa.select(sa.column('id', sa.Integer))
    selectable_b = sa.select(sa.column('other_id', sa.Integer))

    record_a = ViewRecord(
        name='v',
        selectable=selectable_a,
        schema='analytics',
        materialized=False,
    )
    record_b = ViewRecord(
        name='v',
        selectable=selectable_b,
        schema='analytics',
        materialized=False,
    )

    # The selectables differ, but __eq__ only compares name/schema/materialized.
    # This assertion documents the limitation: the records ARE equal under
    # the current implementation, masking real differences in the view body.
    assert record_a == record_b


# ---------------------------------------------------------------------------
# Bug 6: _schema_matches treats None and 'public' as equivalent
# ---------------------------------------------------------------------------
def test_schema_matches_none_public_equivalence():
    """Documents that _schema_matches(None, 'public') is True, which masks
    views that are genuinely in a non-default schema from autogenerate diff
    loops scoped to 'public'."""
    # None (no schema on view) maps to 'public' loop — treated as match.
    assert _schema_matches(None, 'public') is True

    # But a view explicitly in 'analytics' against a None loop is NOT a match,
    # showing the asymmetry: None is a wildcard for 'public' only.
    assert _schema_matches('analytics', None) is False


# ---------------------------------------------------------------------------
# Bug 7: Stale TODO marker left in test_alembic_views.py
# ---------------------------------------------------------------------------
def test_no_stale_todo_in_test_alembic_views():
    """Regression guard: test_alembic_views.py must not carry stale
    `TODO: Task` markers (Bug 7 was fixed in a prior wave)."""
    target = TESTS_DIR / 'test_alembic_views.py'
    text = target.read_text()
    matches = re.findall(r'TODO:\s*Task', text)
    assert matches == [], (
        f"Found {len(matches)} stale 'TODO: Task' marker(s) in "
        f"{target.name}; they should be removed once the work is done."
    )


# ---------------------------------------------------------------------------
# Bug 8: Typo 'test_viewmixn' (should be 'test_viewmixin') in test_view_mixin.py
# ---------------------------------------------------------------------------
def test_no_viewmixn_typos_in_test_view_mixin():
    """Regression guard: test_view_mixin.py must not contain the
    'test_viewmixn' typo (Bug 8 was fixed in a prior wave)."""
    target = TESTS_DIR / 'test_view_mixin.py'
    text = target.read_text()
    matches = re.findall(r'test_viewmixn', text)
    assert matches == [], (
        f"Found {len(matches)} 'test_viewmixn' typo(s) in "
        f"{target.name}; rename to 'test_viewmixin'."
    )


# ---------------------------------------------------------------------------
# Bug 9: docs/alembic.rst contains a Python code block that is not valid Python
# ---------------------------------------------------------------------------
def test_alembic_rst_example_is_valid_python():
    """Regression guard: the Python code block in docs/alembic.rst must be
    syntactically valid (Bug 9 was fixed in a prior wave)."""
    rst_path = (
        TESTS_DIR.parent / 'docs' / 'alembic.rst'
    )
    rst_text = rst_path.read_text()

    # Extract the first `.. code-block:: python` block.
    pattern = re.compile(
        r'\.\.\s+code-block::\s+python\n\n((?:    .*\n|\n)+)',
        re.MULTILINE,
    )
    match = pattern.search(rst_text)
    assert match is not None, "Expected a python code-block in alembic.rst"

    block = match.group(1)
    # Dedent (strip the leading 4-space indent of the rst code block).
    code = textwrap.dedent(block)

    # Should parse as valid Python. Currently it does NOT because of an
    # invalid MetaData(...) constructor invocation in the example.
    ast.parse(code)


# ---------------------------------------------------------------------------
# Bug 10: create_materialized_view ignores cascade_on_drop — DropView always
#         receives the default cascade=True and callers cannot override it.
# ---------------------------------------------------------------------------
def test_materialized_view_dropview_ignores_cascade():
    """The DropView registered by create_materialized_view must honor a
    caller-supplied cascade_on_drop setting rather than always defaulting to
    cascade=True."""
    metadata = sa.MetaData()
    selectable = sa.select(sa.column('id', sa.Integer))

    # Bug: create_materialized_view has no cascade_on_drop kwarg, so the
    # DropView listener is always registered with cascade=True.
    create_materialized_view(
        'mv',
        selectable,
        metadata,
        cascade_on_drop=False,
    )

    # Inspect the registered `before_drop` event listeners on the metadata.
    drop_views = _collect_drop_views(metadata)
    assert drop_views, (
        "Expected at least one DropView listener registered on the metadata"
    )
    # The caller asked for cascade_on_drop=False; the DropView must reflect it.
    cascades = [dv.cascade for dv in drop_views]
    assert all(c is False for c in cascades), (
        "create_materialized_view did not honor cascade_on_drop=False; "
        f"DropView.cascade values: {cascades}"
    )


def _collect_drop_views(metadata):
    """Yield DropView instances registered as `before_drop` listeners."""
    found = []
    dispatch = getattr(metadata, 'dispatch', None)
    if dispatch is None:
        return found
    before_drop = getattr(dispatch, 'before_drop', None)
    if before_drop is None:
        return found
    # DDLElement listeners are stored in a list; each entry may be the
    # DropView itself or a wrapper exposing it via __self__.
    collection = (
        getattr(before_drop, 'objs', None)
        or getattr(before_drop, '_listeners', None)
        or getattr(before_drop, 'listeners', None)
        or []
    )
    for item in collection:
        if isinstance(item, DropView):
            found.append(item)
            continue
        target = getattr(item, '__self__', None)
        if isinstance(target, DropView):
            found.append(target)
    return found
