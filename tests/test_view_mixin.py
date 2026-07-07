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


def test_viewmixin_class_creation_with_valid_selectable():
    Base = declarative_base()

    class MyView(ViewMixin, Base):
        __tablename__ = 'my_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    MyView.__declare_last__()

    assert MyView.__table__ is not None
    assert MyView.__table__.name == 'my_view'
    assert 'id' in MyView.__table__.columns


def test_viewmixin_none_selectable_raises_typeerror():
    """__view_selectable__ = None raises TypeError."""
    class NoneView(ViewMixin):
        __tablename__ = 'none_view'
        __view_selectable__ = None
        metadata = sa.MetaData()

    with pytest.raises(TypeError, match="not None"):
        NoneView.__declare_last__()


def test_viewmixin_string_selectable_raises_typeerror():
    """String __view_selectable__ raises TypeError with helpful message."""
    class StringView(ViewMixin):
        __tablename__ = 'string_view'
        __view_selectable__ = "SELECT 1 AS id"
        metadata = sa.MetaData()

    with pytest.raises(TypeError, match="SQLAlchemy selectable"):
        StringView.__declare_last__()


def test_viewmixin_column_name_mismatch_raises_valueerror():
    Base = declarative_base()

    class MismatchView(ViewMixin, Base):
        __tablename__ = 'mismatch_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
        wrong_name: Mapped[str] = mapped_column(sa.String)

    with pytest.raises(ValueError, match="not found in selectable"):
        MismatchView.__declare_last__()


def test_viewmixin_type_drift_produces_warning(caplog):
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


def test_viewmixin_viewrecord_auto_registered():
    Base = declarative_base()

    class RegView(ViewMixin, Base):
        __tablename__ = 'reg_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    RegView.__declare_last__()

    records = Base.metadata.info.get('sqlalchemy_utils_views', [])
    assert any(vr.name == 'reg_view' for vr in records)


def test_viewmixin_table_set_on_class():
    Base = declarative_base()

    class TableView(ViewMixin, Base):
        __tablename__ = 'table_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    TableView.__declare_last__()

    assert TableView.__table__ is not None
    assert TableView.__table__.name == 'table_view'


def test_viewmixin_ddl_listeners_registered():
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


def test_viewmixin_materialized_flag():
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


def test_viewmixin_before_flush_raises_for_dirty_instance():
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


def test_viewmixin_extra_selectable_column_produces_warning(caplog):
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


def test_viewmixin_importable_from_top_level():
    from sqlalchemy_utils import ViewMixin
    assert ViewMixin is not None


def test_viewreadonlyerror_importable_from_top_level():
    from sqlalchemy_utils import ViewReadonlyError
    assert ViewReadonlyError is not None


def test_viewmixin_autogenerate_integration():
    """End-to-end: ViewMixin class registers ViewRecord in metadata.info
    so that the Alembic comparator can detect views for autogenerate.

    The comparator reads ``metadata.info['sqlalchemy_utils_views']`` to
    produce ``op.create_view`` / ``op.drop_view`` operations.  This test
    verifies the critical prerequisite: that the ViewRecord is correctly
    populated with all fields needed by the comparator.
    """
    from sqlalchemy_utils.view_record import ViewRecord

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


def test_view_cascade_default():
    """ViewMixin.__view_cascade__ defaults to True."""
    assert ViewMixin.__view_cascade__ is True


def test_old_cascade_attr_removed():
    """Old __view_cascade_on_drop__ is no longer a class default."""
    assert not hasattr(ViewMixin, '__view_cascade_on_drop__')


def test_view_schema_default():
    """ViewMixin.__view_schema__ defaults to None."""
    assert ViewMixin.__view_schema__ is None


def test_view_schema_propagated_to_viewrecord():
    """Custom __view_schema__ is stored in ViewRecord after __declare_last__."""
    Base = declarative_base()
    class SchemaView(ViewMixin, Base):
        __tablename__ = 'schema_test_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        __view_schema__ = 'analytics'
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    SchemaView.__declare_last__()
    records = Base.metadata.info.get('sqlalchemy_utils_views', [])
    vr = [r for r in records if r.name == 'schema_test_view'][0]
    assert vr.schema == 'analytics'


def test_table_args_schema_fallback():
    """__table_args__['schema'] used when __view_schema__ not set."""
    Base = declarative_base()
    class FallbackView(ViewMixin, Base):
        __tablename__ = 'fallback_test_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        __table_args__ = {'schema': 'reporting'}
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    FallbackView.__declare_last__()
    records = Base.metadata.info.get('sqlalchemy_utils_views', [])
    vr = [r for r in records if r.name == 'fallback_test_view'][0]
    assert vr.schema == 'reporting'


def test_resolve_schema_list_style():
    """`_resolve_schema` handles list-style __table_args__.

    Regression test: when __table_args__ is a list whose
    final element is a dict containing 'schema' (e.g.
    ``__table_args__ = [{"schema": "public"}]``), `_resolve_schema`
    previously returned None because it only checked
    ``isinstance(table_args, tuple)``.

    SQLAlchemy's declarative scanner rejects a bare list as
    ``__table_args__`` at class-creation time, so the list is assigned
    after declaration to simulate programmatic / runtime mutation. The
    same code path is also exercised when ``__table_args__`` is a list
    that originated from a ``@declared_attr``.
    """
    Base = declarative_base()

    class ListView(ViewMixin, Base):
        __tablename__ = 'list_style_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    ListView.__table_args__ = [{"schema": "public"}]
    ListView.__declare_last__()
    assert ListView._resolve_schema() == "public"


def test_no_global_listener_flag():
    """Module-level _VIEW_READONLY_LISTENER_INSTALLED removed."""
    from sqlalchemy_utils import view_mixin as vm
    assert not hasattr(vm, '_VIEW_READONLY_LISTENER_INSTALLED')


def test_declare_last_forwarding():
    """ViewMixin coexists with a cooperative mixin that also defines
    ``__declare_last__`` without raising TypeError.

    Regression test: the forwarding loop called
    ``base.__declare_last__(cls)`` passing ``cls`` as a second positional
    argument to a classmethod already bound to ``base``, producing
    ``TypeError: __declare_last__() takes 1 positional argument but 2
    were given``.
    """

    class CooperativeMixin:
        @classmethod
        def __declare_last__(cls):
            pass

    Base = declarative_base()

    class MyView(ViewMixin, CooperativeMixin, Base):
        __tablename__ = 'my_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    MyView.__declare_last__()

    assert MyView.__table__ is not None


def test_view_mixin_aliases():
    """__view_aliases__ is threaded through to _register_view_ddl and the
    backing table's column keys for materialized views."""
    Base = declarative_base()

    source = sa.Table(
        'alias_src', Base.metadata,
        sa.Column('old_col', sa.Integer, primary_key=True),
    )

    class AliasedMV(ViewMixin, Base):
        __tablename__ = 'aliased_mv'
        __view_selectable__ = sa.select(source.c.old_col)
        __view_materialized__ = True
        __view_aliases__ = {'old_col': 'new_col'}
        new_col: Mapped[int] = mapped_column('old_col', sa.Integer, primary_key=True)

    AliasedMV.__declare_last__()

    assert 'new_col' in AliasedMV.__table__.columns
    assert AliasedMV.__table__.columns['new_col'].name == 'old_col'

    records = Base.metadata.info.get('sqlalchemy_utils_views', [])
    vr = [r for r in records if r.name == 'aliased_mv'][0]
    assert vr.materialized is True
    assert vr.aliases == {'old_col': 'new_col'}


def test_view_mixin_aliases_not_set_for_regular_views():
    """__view_aliases__ is ignored for non-materialized views."""
    Base = declarative_base()

    class RegularView(ViewMixin, Base):
        __tablename__ = 'regular_aliased_view'
        __view_selectable__ = sa.select(_src(sa.column('id', sa.Integer)))
        __view_aliases__ = {'id': 'should_be_ignored'}
        id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

    RegularView.__declare_last__()

    assert 'id' in RegularView.__table__.columns
    assert 'should_be_ignored' not in RegularView.__table__.columns

    records = Base.metadata.info.get('sqlalchemy_utils_views', [])
    vr = [r for r in records if r.name == 'regular_aliased_view'][0]
    assert vr.aliases is None


def test_view_mixin_aliases_default_none():
    """ViewMixin.__view_aliases__ defaults to None."""
    assert ViewMixin.__view_aliases__ is None
