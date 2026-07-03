from __future__ import annotations

import logging
from typing import Optional

import sqlalchemy as sa

from sqlalchemy_utils.exceptions import ViewReadonlyError
from sqlalchemy_utils.view import (
    create_table_from_selectable,
    _register_view_ddl,
    refresh_materialized_view,
)

logger = logging.getLogger(__name__)


def _view_before_flush(session, flush_context, instances):
    for instance in session.new | session.dirty | session.deleted:
        if isinstance(instance, ViewMixin):
            raise ViewReadonlyError(
                f"Cannot flush changes to view-backed instance "
                f"{instance.__class__.__name__!r}; views are read-only."
            )


class ViewMixin:
    """Declarative mixin for SQLAlchemy ORM view classes.

    Provides automatic DDL generation (CREATE/DROP VIEW) and read-only
    enforcement for view-backed ORM models.

    **Class attributes:**

    * ``__view_selectable__`` — SQLAlchemy selectable defining the view query.
      Maps to ``definition`` in Alembic migrations.
    * ``__view_materialized__`` — ``True`` for materialized views (default ``False``).
    * ``__view_schema__`` — Schema name; takes precedence over
      ``__table_args__['schema']`` (default ``None``).
    * ``__view_cascade__`` — ``True`` to emit ``DROP ... CASCADE`` (default ``True``).
      This is the mixin equivalent of the ``cascade_on_drop`` parameter accepted by
      :func:`sqlalchemy_utils.view.create_view` and
      :func:`sqlalchemy_utils.view.create_materialized_view`.
    * ``__view_replace__`` — ``True`` to emit ``CREATE OR REPLACE`` (default ``False``).
    * ``__view_aliases__`` — Optional ``{column_name: alias}`` mapping used only
      for materialized views (mirrors the ``aliases`` parameter of
      :func:`sqlalchemy_utils.view.create_materialized_view`); ignored for
      regular views (default ``None``).

    **Methods:**

    * ``refresh(session, concurrently=False)`` — Refresh a materialized view.
      Raises a plain ``sa.exc.InvalidRequestError`` (not
      :exc:`ViewReadonlyError`) for non-materialized views.

    **Implementation note:** ``cls.__table__.metadata`` is a throwaway
    ``MetaData()`` instance so that ``create_all()`` does not emit both
    ``CREATE TABLE`` and ``CREATE VIEW``.  View DDL is handled by event
    listeners on ``cls.metadata`` (the Base's metadata).

    Example::

        import sqlalchemy as sa
        from sqlalchemy.orm import mapped_column, Mapped

        class UserCountView(ViewMixin, Base):
            __tablename__ = 'user_count_view'
            __view_selectable__ = sa.select(
                sa.func.count(User.id).label('count')
            )
            count: Mapped[int] = mapped_column(sa.Integer, primary_key=True)

        class TopUsersMV(ViewMixin, Base):
            __tablename__ = 'top_users_mv'
            __view_selectable__ = sa.select(User).where(User.score > 100)
            __view_materialized__ = True
            id: Mapped[int] = mapped_column(sa.Integer, primary_key=True)
    """
    __view_selectable__ = None
    __view_materialized__ = False
    __view_schema__ = None
    __view_cascade__ = True
    __view_replace__ = False
    __view_aliases__: Optional[dict] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        has_columns = any(
            isinstance(v, sa.Column)
            or (v is not None and type(v).__name__ == "MappedColumn")
            for v in cls.__dict__.values()
        )
        if has_columns:
            has_tablename = any(
                '__tablename__' in base.__dict__ for base in cls.__mro__
                if base is not ViewMixin
            )
            if not has_tablename:
                raise TypeError(
                    f"{cls.__name__}: ViewMixin requires __tablename__ to be "
                    f"set on the class (alongside __view_selectable__)."
                )

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

        # Resolve schema: __view_schema__ takes precedence over __table_args__['schema']
        table_args = getattr(cls, '__table_args__', None)
        view_schema = cls._resolve_schema()
        # Store the resolved schema so refresh() (and other runtime callers)
        # can use it without re-resolving from __table_args__ each time.
        cls._resolved_view_schema = view_schema

        indexes = []
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
        cascade_on_drop = getattr(cls, '__view_cascade__', True)
        # Aliases are only meaningful for materialized views (regular views
        # do not support column aliases via create_table_from_selectable).
        aliases = None
        if is_materialized:
            aliases = getattr(cls, '__view_aliases__', None)

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
            aliases=aliases if is_materialized else None,
            schema=view_schema,
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

        _register_view_ddl(
            metadata=metadata,
            name=cls.__tablename__,
            selectable=selectable,
            materialized=is_materialized,
            replace=replace,
            cascade_on_drop=cascade_on_drop,
            schema=view_schema,
            table=table,
            indexes=indexes if indexes else None,
            aliases=aliases if is_materialized else None,
        )

        if not sa.event.contains(sa.orm.Session, 'before_flush', _view_before_flush):
            sa.event.listen(sa.orm.Session, 'before_flush', _view_before_flush)

    @classmethod
    def _resolve_schema(cls):
        """Resolve the view's schema for DDL operations.

        Priority:
        1. _resolved_view_schema (set by __declare_last__)
        2. __view_schema__ class attribute
        3. __table_args__['schema'] if present
        """
        schema = getattr(cls, '_resolved_view_schema', None)
        if schema is None:
            schema = getattr(cls, '__view_schema__', None)
        if schema is None:
            table_args = getattr(cls, '__table_args__', None)
            if isinstance(table_args, dict):
                schema = table_args.get('schema')
            elif isinstance(table_args, (list, tuple)) and table_args:
                for arg in table_args:
                    if isinstance(arg, dict):
                        schema = arg.get('schema')
                        if schema:
                            break
        return schema

    @classmethod
    def refresh(cls, session, *, concurrently: bool = False):
        """Refresh a materialized view.

        :param session: An SQLAlchemy Session instance.
        :param concurrently: If ``True``, refresh with ``CONCURRENTLY``.

        :raises sa.exc.InvalidRequestError: If the view is not materialized.
            Note this is a plain ``InvalidRequestError`` (not
            :exc:`sqlalchemy_utils.exceptions.ViewReadonlyError`); the latter
            is only raised for write attempts (flushes) on view-backed
            instances.
        """
        if not cls.__view_materialized__:
            raise sa.exc.InvalidRequestError(
                f"Cannot refresh non-materialized view {cls.__tablename__!r}"
            )
        refresh_materialized_view(
            session,
            cls.__tablename__,
            concurrently=concurrently,
            schema=cls._resolve_schema(),
        )
