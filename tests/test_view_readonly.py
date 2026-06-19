"""
Tests for ViewMixin read-only enforcement.

Verifies that the before_flush listener prevents INSERT, UPDATE, and DELETE
operations on view-backed ORM instances while leaving regular (non-view)
classes unaffected.
"""
import pytest
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, declarative_base, mapped_column

from sqlalchemy_utils.exceptions import ViewReadonlyError
from sqlalchemy_utils.view_mixin import ViewMixin, _view_before_flush


def _make_view_base():
    """Create a fresh Base + source table + View class for testing."""
    Base = declarative_base()
    source = sa.Table(
        'readonly_src', Base.metadata,
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String),
    )

    class ReadOnlyView(ViewMixin, Base):
        __tablename__ = 'readonly_view'
        __view_selectable__ = sa.select(source.c.id, source.c.name)
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
        name: Mapped[str] = mapped_column(sa.String)

    ReadOnlyView.__declare_last__()
    return Base, source, ReadOnlyView


def _fake_session(new=None, dirty=None, deleted=None):
    """Build a lightweight session-like object for _view_before_flush."""
    return type('FakeSession', (), {
        'new': new or set(),
        'dirty': dirty or set(),
        'deleted': deleted or set(),
    })()


def test_add_raises_viewreadonlyerror():
    """session.add(view_instance) raises ViewReadonlyError on flush."""
    from sqlalchemy.orm import Session

    Base, source, ReadOnlyView = _make_view_base()
    engine = sa.create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        # Seed a row in the source table so the view is queryable
        session.execute(source.insert().values(id=1, name='seed'))
        session.commit()

        # Attempt to INSERT into the view
        view_instance = ReadOnlyView(id=2, name='illegal')
        session.add(view_instance)
        with pytest.raises(ViewReadonlyError):
            session.flush()


def test_modify_raises_viewreadonlyerror():
    """Dirty view instance in session raises ViewReadonlyError."""
    Base, source, ReadOnlyView = _make_view_base()

    fake_instance = ReadOnlyView(id=1, name='modified')
    fs = _fake_session(dirty={fake_instance})

    with pytest.raises(ViewReadonlyError):
        _view_before_flush(fs, None, None)


def test_delete_raises_viewreadonlyerror():
    """Deleted view instance in session raises ViewReadonlyError."""
    Base, source, ReadOnlyView = _make_view_base()

    fake_instance = ReadOnlyView(id=1, name='doomed')
    fs = _fake_session(deleted={fake_instance})

    with pytest.raises(ViewReadonlyError):
        _view_before_flush(fs, None, None)


def test_non_view_class_unaffected():
    """Non-ViewMixin classes pass through _view_before_flush without error."""
    Base = declarative_base()

    class RegularTable(Base):
        __tablename__ = 'regular_table'
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
        name: Mapped[str] = mapped_column(sa.String)

    fake_instance = RegularTable(id=1, name='normal')
    fs = _fake_session(new={fake_instance})

    # Must NOT raise — non-view class is unaffected
    _view_before_flush(fs, None, None)
