"""
Regression tests for validated schema-parameter and related bugs.

These tests document bugs that exist in the current codebase. They are
intentionally written to FAIL against the unfixed code, proving each bug
is real. Each test names the bug it covers.

Run:
    PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
        tests/test_schema_regression.py -v
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import pytest
import sqlalchemy as sa

from sqlalchemy_utils.view_record import ViewRecord
from sqlalchemy_utils.view import (
    DropView,
    create_materialized_view,
    create_view,
)


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
