import logging

import sqlalchemy as sa

from sqlalchemy_utils.alembic.view_record import ViewRecord
from sqlalchemy_utils.exceptions import ViewReadonlyError
from sqlalchemy_utils.view import create_table_from_selectable, CreateView, DropView

logger = logging.getLogger(__name__)

_VIEW_READONLY_LISTENER_INSTALLED = False


def _view_before_flush(session, flush_context, instances):
    for instance in session.new | session.dirty | session.deleted:
        if isinstance(instance, ViewMixin):
            raise ViewReadonlyError(
                f"Cannot flush changes to view-backed instance "
                f"{instance.__class__.__name__!r}; views are read-only."
            )


class ViewMixin:
    __view_selectable__ = None
    __view_materialized__ = False

    @classmethod
    def __declare_last__(cls):
        selectable = cls.__view_selectable__

        if selectable is None:
            raise TypeError(
                f"{cls.__name__}.__view_selectable__ must be a "
                f"SQLAlchemy selectable, not None"
            )
        if isinstance(selectable, str):
            raise TypeError(
                f"{cls.__name__}.__view_selectable__ must be a SQLAlchemy "
                f"selectable, not a string"
            )

        metadata = cls.metadata

        indexes = []
        table_args = getattr(cls, '__table_args__', None)
        if table_args:
            if isinstance(table_args, (list, tuple)):
                items = table_args
                if isinstance(table_args[-1], dict):
                    items = table_args[:-1]
                for arg in items:
                    if isinstance(arg, sa.Index):
                        indexes.append(arg)

        is_materialized = getattr(cls, '__view_materialized__', False)
        replace = getattr(cls, '__view_replace__', False)
        cascade_on_drop = getattr(cls, '__view_cascade_on_drop__', True)

        declared_col_names = set()
        declared_col_types = {}
        if hasattr(cls, '__table__') and cls.__table__ is not None:
            for col in cls.__table__.columns:
                declared_col_names.add(col.name)
                declared_col_types[col.name] = col.type

        if hasattr(cls, '__table__') and cls.__table__ is not None:
            metadata.remove(cls.__table__)

        # metadata=None so DDL is not emitted by metadata.create_all();
        # view DDL is handled by listeners
        table = create_table_from_selectable(
            name=cls.__tablename__,
            selectable=selectable,
            indexes=indexes if indexes else None,
            metadata=None,
        )
        cls.__table__ = table

        selectable_col_names = {c.name for c in table.columns}
        selectable_col_types = {c.name: c.type for c in table.columns}

        missing_in_selectable = declared_col_names - selectable_col_names
        if missing_in_selectable:
            raise ValueError(
                f"Column(s) {missing_in_selectable!r} declared on "
                f"{cls.__name__} but not found in selectable"
            )

        extra_in_selectable = selectable_col_names - declared_col_names
        if extra_in_selectable:
            logger.warning(
                "Column(s) %r in selectable but not declared on %s",
                extra_in_selectable,
                cls.__name__,
            )

        for col_name, col_type in declared_col_types.items():
            if col_name in selectable_col_types:
                selectable_type = selectable_col_types[col_name]
                if type(col_type) != type(selectable_type):
                    logger.warning(
                        "Type drift on %s.%s: declared %s but selectable has %s",
                        cls.__name__,
                        col_name,
                        col_type,
                        selectable_type,
                    )

        sa.event.listen(
            metadata,
            'after_create',
            CreateView(cls.__tablename__, selectable, materialized=is_materialized),
        )
        sa.event.listen(
            metadata,
            'before_drop',
            DropView(cls.__tablename__, materialized=is_materialized, cascade=cascade_on_drop),
        )

        if is_materialized and indexes:

            @sa.event.listens_for(metadata, 'after_create')
            def create_indexes(target, connection, **kw):
                for idx in table.indexes:
                    idx.create(connection)

        view_records = metadata.info.setdefault('sqlalchemy_utils_views', [])
        view_records.append(ViewRecord(
            name=cls.__tablename__,
            selectable=selectable,
            schema=None,
            materialized=is_materialized,
            replace=replace,
            cascade_on_drop=cascade_on_drop,
        ))

        global _VIEW_READONLY_LISTENER_INSTALLED
        if not _VIEW_READONLY_LISTENER_INSTALLED:
            sa.event.listen(sa.orm.Session, 'before_flush', _view_before_flush)
            _VIEW_READONLY_LISTENER_INSTALLED = True
