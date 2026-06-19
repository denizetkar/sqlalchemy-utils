import logging

import pytest
import sqlalchemy as sa
from sqlalchemy.orm import declarative_base, Mapped, mapped_column

from sqlalchemy_utils.exceptions import ViewReadonlyError
from sqlalchemy_utils.view_mixin import ViewMixin, _view_before_flush


def _src(*columns):
    return sa.table('src', *columns)


def test_viewreadonlyerror_importable():
    assert ViewReadonlyError is not None


def test_viewreadonlyerror_raisable_catchable():
    try:
        raise ViewReadonlyError("test")
    except ViewReadonlyError as e:
        assert str(e) == "test"


def test_viewmixn_class_creation_with_valid_selectable():
    Base = declarative_base()

    class MyView(ViewMixin, Base):
        __tablename__ = 'my_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    MyView.__declare_last__()

    assert MyView.__table__ is not None
    assert MyView.__table__.name == 'my_view'
    assert 'id' in MyView.__table__.columns


def test_viewmixn_none_selectable_raises_typeerror():
    """__view_selectable__ = None raises TypeError."""
    class NoneView(ViewMixin):
        __tablename__ = 'none_view'
        __view_selectable__ = None
        metadata = sa.MetaData()

    with pytest.raises(TypeError, match="not None"):
        NoneView.__declare_last__()


def test_viewmixn_string_selectable_raises_typeerror():
    """String __view_selectable__ raises TypeError."""
    class StringView(ViewMixin):
        __tablename__ = 'string_view'
        __view_selectable__ = "SELECT 1 AS id"
        metadata = sa.MetaData()

    with pytest.raises(TypeError, match="not a string"):
        StringView.__declare_last__()


def test_viewmixn_column_name_mismatch_raises_valueerror():
    Base = declarative_base()

    class MismatchView(ViewMixin, Base):
        __tablename__ = 'mismatch_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
        wrong_name: Mapped[str] = mapped_column(sa.String)

    with pytest.raises(ValueError, match="not found in selectable"):
        MismatchView.__declare_last__()


def test_viewmixn_type_drift_produces_warning(caplog):
    Base = declarative_base()

    class DriftView(ViewMixin, Base):
        __tablename__ = 'drift_view'
        __view_selectable__ = sa.select(
            sa.cast(sa.column('id', sa.Integer), sa.String).label('id')
        )
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    with caplog.at_level(logging.WARNING, logger="sqlalchemy_utils.view_mixin"):
        DriftView.__declare_last__()

    assert any("Type drift" in rec.message for rec in caplog.records)


def test_viewmixn_viewrecord_auto_registered():
    Base = declarative_base()

    class RegView(ViewMixin, Base):
        __tablename__ = 'reg_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    RegView.__declare_last__()

    records = Base.metadata.info.get('sqlalchemy_utils_views', [])
    assert any(vr.name == 'reg_view' for vr in records)


def test_viewmixn_table_set_on_class():
    Base = declarative_base()

    class TableView(ViewMixin, Base):
        __tablename__ = 'table_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    TableView.__declare_last__()

    assert TableView.__table__ is not None
    assert TableView.__table__.name == 'table_view'


def test_viewmixn_ddl_listeners_registered():
    Base = declarative_base()

    source = sa.Table(
        'source', Base.metadata,
        sa.Column('id', sa.Integer, primary_key=True),
    )

    class DDLView(ViewMixin, Base):
        __tablename__ = 'ddl_view'
        __view_selectable__ = sa.select(source.c.id)
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    DDLView.__declare_last__()

    engine = sa.create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    with engine.connect() as conn:
        result = conn.execute(
            sa.text("SELECT name FROM sqlite_master WHERE type='view' AND name='ddl_view'")
        )
        assert result.fetchone() is not None


def test_viewmixn_materialized_flag():
    Base = declarative_base()

    class MatView(ViewMixin, Base):
        __tablename__ = 'mat_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        __view_materialized__ = True
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    MatView.__declare_last__()

    records = Base.metadata.info.get('sqlalchemy_utils_views', [])
    mat_record = next((vr for vr in records if vr.name == 'mat_view'), None)
    assert mat_record is not None
    assert mat_record.materialized is True


def test_viewmixn_before_flush_raises_for_dirty_instance():
    Base = declarative_base()

    class FlushView(ViewMixin, Base):
        __tablename__ = 'flush_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    FlushView.__declare_last__()

    class FakeViewInstance(ViewMixin):
        pass

    fake_instance = FakeViewInstance()

    session = type('FakeSession', (), {
        'new': set(),
        'dirty': {fake_instance},
        'deleted': set(),
    })()

    with pytest.raises(ViewReadonlyError):
        _view_before_flush(session, None, None)


def test_viewmixn_extra_selectable_column_produces_warning(caplog):
    Base = declarative_base()

    class ExtraView(ViewMixin, Base):
        __tablename__ = 'extra_view'
        __view_selectable__ = sa.select(
            _src(
                sa.column('id', sa.Integer),
                sa.column('name', sa.String),
            )
        )
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    with caplog.at_level(logging.WARNING, logger="sqlalchemy_utils.view_mixin"):
        ExtraView.__declare_last__()

    assert any("not declared on" in rec.message for rec in caplog.records)


def test_viewmixn_importable_from_top_level():
    from sqlalchemy_utils import ViewMixin
    assert ViewMixin is not None


def test_viewreadonlyerror_importable_from_top_level():
    from sqlalchemy_utils import ViewReadonlyError
    assert ViewReadonlyError is not None


def test_viewmixn_autogenerate_integration():
    """End-to-end: ViewMixin class registers ViewRecord in metadata.info
    so that the Alembic comparator can detect views for autogenerate.

    The comparator reads ``metadata.info['sqlalchemy_utils_views']`` to
    produce ``op.create_view`` / ``op.drop_view`` operations.  This test
    verifies the critical prerequisite: that the ViewRecord is correctly
    populated with all fields needed by the comparator.
    """
    from sqlalchemy_utils.alembic.view_record import ViewRecord

    Base = declarative_base()

    source = sa.Table(
        'autogen_src', Base.metadata,
        sa.Column('id', sa.Integer, primary_key=True),
    )

    class AutoGenView(ViewMixin, Base):
        __tablename__ = 'autogen_view'
        __view_selectable__ = sa.select(source.c.id)
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    AutoGenView.__declare_last__()

    records = Base.metadata.info.get('sqlalchemy_utils_views', [])
    matching = [vr for vr in records if vr.name == 'autogen_view']
    assert len(matching) >= 1, (
        f"Expected a ViewRecord named 'autogen_view', got {[vr.name for vr in records]}"
    )

    vr = matching[0]
    assert isinstance(vr, ViewRecord)
    assert vr.schema is None
    assert vr.materialized is False
    assert vr.replace is False
    assert vr.cascade_on_drop is True
    # The selectable must be present and compilable
    compiled = str(vr.selectable.compile(compile_kwargs={"literal_binds": True}))
    assert 'autogen_src' in compiled
