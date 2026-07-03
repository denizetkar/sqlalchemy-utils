from typing import Optional, Union

import sqlalchemy as sa
from sqlalchemy.ext import compiler
from sqlalchemy.schema import DDLElement, PrimaryKeyConstraint
from sqlalchemy.sql.expression import ClauseElement, Executable

from sqlalchemy_utils.functions import get_columns

from sqlalchemy_utils.view_record import ViewRecord


# ---------------------------------------------------------------------------
# Identifier-quoting helpers (single source of truth; re-exported by
# sqlalchemy_utils.alembic.operations and used by alembic.comparator).
# ---------------------------------------------------------------------------

def _quote_identifier(dialect, name):
    """Quote *name* using the dialect's identifier preparer."""
    return dialect.identifier_preparer.quote(name)


def _quote_qualified_name(dialect, name, schema=None):
    """Return a schema-qualified, properly quoted identifier.

    When *schema* is given the result is ``"schema"."name"`` (both parts
    quoted by the dialect's identifier preparer); otherwise just ``"name"``.
    """
    if schema:
        return f"{_quote_identifier(dialect, schema)}.{_quote_identifier(dialect, name)}"
    return _quote_identifier(dialect, name)


class CreateView(DDLElement):
    """DDL element for CREATE VIEW (or CREATE OR REPLACE VIEW).

    :raises ValueError: if both ``materialized`` and ``replace`` are True.
    """

    def __init__(self, name, selectable, materialized=False, replace=False, *, schema=None):
        if materialized and replace:
            raise ValueError('Cannot use CREATE OR REPLACE with materialized views')
        self.name = name
        self.selectable = selectable
        self.materialized = materialized
        self.replace = replace
        self.schema = schema


@compiler.compiles(CreateView)
def compile_create_materialized_view(element, compiler, **kw):
    """Compile ``CreateView`` to ``CREATE [OR REPLACE] [MATERIALIZED] VIEW``."""
    ip = compiler.dialect.identifier_preparer
    qualified = _quote_qualified_name(compiler.dialect, element.name, element.schema)
    return 'CREATE {}{}VIEW {} AS {}'.format(
        'OR REPLACE ' if element.replace else '',
        'MATERIALIZED ' if element.materialized else '',
        qualified,
        compiler.sql_compiler.process(element.selectable, literal_binds=True),
    )


class DropView(DDLElement):
    """DDL element for DROP VIEW (or DROP MATERIALIZED VIEW)."""

    def __init__(self, name, materialized=False, cascade=True, *, schema=None):
        self.name = name
        self.materialized = materialized
        self.cascade = cascade
        self.schema = schema


@compiler.compiles(DropView)
def compile_drop_materialized_view(element, compiler, **kw):
    """Compile ``DropView`` to ``DROP [MATERIALIZED] VIEW IF EXISTS [...]``."""
    qualified = _quote_qualified_name(compiler.dialect, element.name, element.schema)
    sql = 'DROP {}VIEW IF EXISTS {}'.format(
        'MATERIALIZED ' if element.materialized else '',
        qualified,
    )
    if element.cascade:
        sql += ' CASCADE'
    return sql


def create_table_from_selectable(
    name, selectable, indexes=None, metadata=None, aliases=None, schema=None, **kwargs
):
    """Create a :class:`~sqlalchemy.Table` from a selectable.

    Builds a table whose columns mirror the selectable's output columns.
    If no column has ``primary_key=True``, a :class:`PrimaryKeyConstraint`
    is added over all columns.

    :param name: Table name.
    :param selectable: A SQLAlchemy selectable (``select()``, ``text()``, etc.)
        or a string SQL expression.
    :param indexes: Optional list of :class:`~sqlalchemy.Index` objects.
    :param metadata: :class:`~sqlalchemy.MetaData` to attach the table to.
        If ``None``, the table is not attached to any metadata (used by
        :func:`create_view` and :func:`create_materialized_view`).
    :param aliases: Optional ``{column_name: alias}`` mapping to override
        column keys.
    :param schema: Optional schema name.
    :param kwargs: Additional ``Table`` constructor arguments.
    :returns: The created :class:`~sqlalchemy.Table`.
    """
    if indexes is None:
        indexes = []
    if metadata is None:
        metadata = sa.MetaData()
    if aliases is None:
        aliases = {}
    args = [
        sa.Column(
            c.name, c.type, key=aliases.get(c.name, c.name), primary_key=c.primary_key
        )
        for c in get_columns(selectable)
    ] + indexes
    table = sa.Table(name, metadata, *args, schema=schema, **kwargs)

    if not any([c.primary_key for c in get_columns(selectable)]):
        table.append_constraint(
            PrimaryKeyConstraint(*[c.name for c in get_columns(selectable)])
        )
    return table


def _register_view_ddl(
    metadata,
    name,
    selectable,
    materialized,
    replace,
    cascade_on_drop,
    schema,
    table=None,
    indexes=None,
    aliases=None,
):
    """Register CREATE/DROP DDL listeners and a ViewRecord on *metadata*.

    Shared by :func:`create_view`, :func:`create_materialized_view`, and
    :class:`~sqlalchemy_utils.view_mixin.ViewMixin.__declare_last__` to avoid
    triplicated listener registration and ViewRecord construction.

    When *materialized* is ``True`` and *indexes* (or *table* with indexes) is
    provided, a metadata-scoped ``after_create`` listener is registered that
    creates each index after the view's backing table is created. This is
    required because the backing table is built with ``metadata=None`` (so a
    table-scoped listener would never fire during ``metadata.create_all()``).
    """
    sa.event.listen(
        metadata,
        'after_create',
        CreateView(
            name,
            selectable=selectable,
            materialized=materialized,
            replace=replace,
            schema=schema,
        ),
    )
    sa.event.listen(
        metadata,
        'before_drop',
        DropView(
            name,
            materialized=materialized,
            cascade=cascade_on_drop,
            schema=schema,
        ),
    )

    if materialized and table is not None and table.indexes:

        @sa.event.listens_for(metadata, 'after_create')
        def create_indexes(target, connection, **kw):
            if target is not table:
                return
            for idx in table.indexes:
                idx.create(connection)

    view_records = metadata.info.setdefault('sqlalchemy_utils_views', [])
    view_records.append(ViewRecord(
        name=name,
        selectable=selectable,
        schema=schema,
        materialized=materialized,
        replace=replace,
        cascade_on_drop=cascade_on_drop,
        aliases=aliases,
    ))


def create_materialized_view(
    name: str,
    selectable: Union[str, ClauseElement],
    metadata: sa.MetaData,
    indexes: Optional[list[sa.Index]] = None,
    aliases: Optional[dict[str, str]] = None,
    cascade_on_drop: bool = True,
    *,
    schema: Optional[str] = None,
) -> sa.Table:
    """Create a view on a given metadata

    :param name: The name of the view to create.
    :param selectable: An SQLAlchemy selectable e.g. a select() statement.
    :param metadata:
        An SQLAlchemy Metadata instance that stores the features of the
        database being described.
    :param indexes: An optional list of SQLAlchemy Index instances.
    :param aliases:
        An optional dictionary containing with keys as column names and values
        as column aliases.
    :param cascade_on_drop:
        If ``True`` the view will be dropped with ``CASCADE``,
        deleting all dependent objects as well.
    :param schema:
        Keyword-only. An optional string specifying the schema (database) in
        which the view should be created. When supplied, the view name is
        qualified with the schema in the emitted ``CREATE``/``DROP``/``REFRESH``
        DDL.

    Same as for ``create_view`` except that a ``CREATE MATERIALIZED VIEW``
    statement is emitted instead of a ``CREATE VIEW``.

    .. note::
        The runtime DDL path (this function, ``metadata.create_all()``, and
        the ``CreateView`` DDL element) emits a plain
        ``CREATE MATERIALIZED VIEW``; PostgreSQL defaults to ``WITH DATA``
        so the materialized view is populated immediately. Only the Alembic
        ``op.create_materialized_view`` operation supports an explicit
        ``WITH NO DATA`` clause (via ``with_data=False``, which is also the
        autogenerate default).

    """
    table = create_table_from_selectable(
        name=name,
        selectable=selectable,
        indexes=indexes,
        metadata=None,
        aliases=aliases,
        schema=schema,
    )

    _register_view_ddl(
        metadata=metadata,
        name=name,
        selectable=selectable,
        materialized=True,
        replace=False,
        cascade_on_drop=cascade_on_drop,
        schema=schema,
        table=table,
        indexes=indexes,
        aliases=aliases,
    )
    return table


def create_view(
    name: str,
    selectable: Union[str, ClauseElement],
    metadata: sa.MetaData,
    cascade_on_drop: bool = True,
    replace: bool = False,
    *,
    schema: Optional[str] = None,
) -> sa.Table:
    """Create a view on a given metadata

    :param name: The name of the view to create.
    :param selectable: An SQLAlchemy selectable e.g. a select() statement.
    :param metadata:
        An SQLAlchemy Metadata instance that stores the features of the
        database being described.
    :param cascade_on_drop:
        If ``True`` the view will be dropped with
        ``CASCADE``, deleting all dependent objects as well.
    :param replace:
        If ``True`` the view will be created with ``OR REPLACE``,
        replacing an existing view with the same name.
    :param schema:
        Keyword-only. An optional string specifying the schema (database) in
        which the view should be created. When supplied, the view name is
        qualified with the schema in the emitted ``CREATE``/``DROP`` DDL.

    The process for creating a view is similar to the standard way that a
    table is constructed, except that a selectable is provided instead of a
    set of columns. The view is created once a ``CREATE`` statement is
    executed against the supplied metadata (e.g. ``metadata.create_all(..)``),
    and dropped when a ``DROP`` is executed against the metadata.

    To create a view that performs basic filtering on a table. ::

        metadata = MetaData()
        users = Table('users', metadata,
                Column('id', Integer, primary_key=True),
                Column('name', String),
                Column('fullname', String),
                Column('premium_user', Boolean, default=False),
            )

        premium_members = select(users).where(users.c.premium_user == True)
        create_view('premium_users', premium_members, metadata)

        metadata.create_all(engine) # View is created at this point

    .. note::
        Unlike :func:`create_materialized_view`, this function does not
        accept ``indexes`` or ``aliases`` parameters. Regular PostgreSQL
        views do not support indexes, and column aliases can be defined
        directly in the view's SELECT statement.
    """
    table = create_table_from_selectable(
        name=name, selectable=selectable, metadata=None, schema=schema
    )

    _register_view_ddl(
        metadata=metadata,
        name=name,
        selectable=selectable,
        materialized=False,
        replace=replace,
        cascade_on_drop=cascade_on_drop,
        schema=schema,
        table=table,
        aliases=None,
    )
    return table


class RefreshMaterializedView(Executable, ClauseElement):
    """Executable SQL construct for REFRESH MATERIALIZED VIEW."""

    inherit_cache = True

    def __init__(self, name, concurrently=False, *, schema=None):
        self.name = name
        self.schema = schema
        self.concurrently = concurrently


@compiler.compiles(RefreshMaterializedView)
def compile_refresh_materialized_view(element, compiler, **kw):
    """Compile ``RefreshMaterializedView`` to ``REFRESH MATERIALIZED VIEW``."""
    qualified = _quote_qualified_name(compiler.dialect, element.name, element.schema)
    return 'REFRESH MATERIALIZED VIEW {concurrently}{name}'.format(
        concurrently='CONCURRENTLY ' if element.concurrently else '',
        name=qualified,
    )


def refresh_materialized_view(
    session: sa.orm.Session,
    name: str,
    concurrently: bool = False,
    *,
    schema: Optional[str] = None,
) -> None:
    """Refreshes an already existing materialized view

    :param session: An SQLAlchemy Session instance.
    :param name: The name of the materialized view to refresh.
    :param concurrently:
        Optional flag that causes the ``CONCURRENTLY`` parameter
        to be specified when the materialized view is refreshed.
    :param schema:
        An optional string specifying the schema (database) in which the
        materialized view resides. When supplied, the view name is qualified
        with the schema in the emitted ``REFRESH MATERIALIZED VIEW`` DDL.
    """
    # Since session.execute() bypasses autoflush, we must manually flush in
    # order to include newly-created/modified objects in the refresh.
    session.flush()
    session.execute(RefreshMaterializedView(name, schema=schema, concurrently=concurrently))
