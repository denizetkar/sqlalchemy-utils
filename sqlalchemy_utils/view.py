import sqlalchemy as sa
from sqlalchemy.ext import compiler
from sqlalchemy.schema import DDLElement, PrimaryKeyConstraint
from sqlalchemy.sql.expression import ClauseElement, Executable

from sqlalchemy_utils.functions import get_columns

try:
    from sqlalchemy_utils.alembic.view_record import ViewRecord
except ImportError:
    ViewRecord = None


class CreateView(DDLElement):
    def __init__(self, name, selectable, materialized=False, replace=False, schema=None):
        if materialized and replace:
            raise ValueError('Cannot use CREATE OR REPLACE with materialized views')
        self.name = name
        self.selectable = selectable
        self.materialized = materialized
        self.replace = replace
        self.schema = schema


@compiler.compiles(CreateView)
def compile_create_materialized_view(element, compiler, **kw):
    schema_prefix = f'{compiler.dialect.identifier_preparer.quote(element.schema)}.' if element.schema else ''
    return 'CREATE {}{}VIEW {}{} AS {}'.format(
        'OR REPLACE ' if element.replace else '',
        'MATERIALIZED ' if element.materialized else '',
        schema_prefix,
        compiler.dialect.identifier_preparer.quote(element.name),
        compiler.sql_compiler.process(element.selectable, literal_binds=True),
    )


class DropView(DDLElement):
    def __init__(self, name, materialized=False, cascade=True, schema=None):
        self.name = name
        self.materialized = materialized
        self.cascade = cascade
        self.schema = schema


@compiler.compiles(DropView)
def compile_drop_materialized_view(element, compiler, **kw):
    schema_prefix = f'{compiler.dialect.identifier_preparer.quote(element.schema)}.' if element.schema else ''
    sql = 'DROP {}VIEW IF EXISTS {}{}'.format(
        'MATERIALIZED ' if element.materialized else '',
        schema_prefix,
        compiler.dialect.identifier_preparer.quote(element.name),
    )
    if element.cascade:
        sql += ' CASCADE'
    return sql


def create_table_from_selectable(
    name, selectable, indexes=None, metadata=None, aliases=None, schema=None, **kwargs
):
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


def create_materialized_view(
    name,
    selectable,
    metadata,
    *,
    indexes=None,
    aliases=None,
    cascade_on_drop=True,
    schema=None,
):
    """Create a view on a given metadata

    :param name: The name of the view to create.
    :param selectable: An SQLAlchemy selectable e.g. a select() statement.
    :param metadata:
        An SQLAlchemy Metadata instance that stores the features of the
        database being described.
    :param indexes:
        Keyword-only. An optional list of SQLAlchemy Index instances.
    :param aliases:
        Keyword-only. An optional dictionary containing with keys as column
        names and values as column aliases.
    :param cascade_on_drop:
        Keyword-only. If ``True`` the view will be dropped with ``CASCADE``,
        deleting all dependent objects as well.
    :param schema:
        Keyword-only. An optional string specifying the schema (database) in
        which the view should be created. When supplied, the view name is
        qualified with the schema in the emitted ``CREATE``/``DROP``/``REFRESH``
        DDL.

    Same as for ``create_view`` except that a ``CREATE MATERIALIZED VIEW``
    statement is emitted instead of a ``CREATE VIEW``.

    """
    table = create_table_from_selectable(
        name=name,
        selectable=selectable,
        indexes=indexes,
        metadata=None,
        aliases=aliases,
        schema=schema,
    )

    sa.event.listen(
        metadata,
        'after_create',
        CreateView(name, selectable, materialized=True, schema=schema),
    )

    @sa.event.listens_for(metadata, 'after_create')
    def create_indexes(target, connection, **kw):
        # The table is built with metadata=None (see create_table_from_selectable
        # above), so it is NOT registered on this metadata and a table-scoped
        # listener would never fire during metadata.create_all(). We therefore
        # listen on the metadata and filter to only act for this view's table.
        if target is not table:
            return
        for idx in table.indexes:
            idx.create(connection)

    sa.event.listen(
        metadata,
        'before_drop',
        DropView(
            name, materialized=True, cascade=cascade_on_drop, schema=schema
        ),
    )

    view_records = metadata.info.setdefault('sqlalchemy_utils_views', [])
    view_records.append(ViewRecord(
        name=name,
        selectable=selectable,
        schema=schema,
        materialized=True,
        replace=False,
        cascade_on_drop=cascade_on_drop,
    ))
    return table


def create_view(
    name,
    selectable,
    metadata,
    *,
    cascade_on_drop=True,
    replace=False,
    schema=None,
):
    """Create a view on a given metadata

    :param name: The name of the view to create.
    :param selectable: An SQLAlchemy selectable e.g. a select() statement.
    :param metadata:
        An SQLAlchemy Metadata instance that stores the features of the
        database being described.
    :param cascade_on_drop:
        Keyword-only. If ``True`` the view will be dropped with
        ``CASCADE``, deleting all dependent objects as well.
    :param replace:
        Keyword-only. If ``True`` the view will be created with ``OR REPLACE``,
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

    sa.event.listen(
        metadata,
        'after_create',
        CreateView(name, selectable, replace=replace, schema=schema),
    )

    @sa.event.listens_for(metadata, 'after_create')
    def create_indexes(target, connection, **kw):
        # The table is built with metadata=None (see create_table_from_selectable
        # above), so it is NOT registered on this metadata and a table-scoped
        # listener would never fire during metadata.create_all(). We therefore
        # listen on the metadata and filter to only act for this view's table.
        if target is not table:
            return
        for idx in table.indexes:
            idx.create(connection)

    sa.event.listen(
        metadata, 'before_drop', DropView(name, cascade=cascade_on_drop, schema=schema)
    )

    view_records = metadata.info.setdefault('sqlalchemy_utils_views', [])
    view_records.append(ViewRecord(
        name=name,
        selectable=selectable,
        schema=schema,
        materialized=False,
        replace=replace,
        cascade_on_drop=cascade_on_drop,
    ))
    return table


class RefreshMaterializedView(Executable, ClauseElement):
    inherit_cache = True

    def __init__(self, name, schema=None, concurrently=False):
        self.name = name
        self.schema = schema
        self.concurrently = concurrently


@compiler.compiles(RefreshMaterializedView)
def compile_refresh_materialized_view(element, compiler, **kw):
    schema_prefix = f'{compiler.dialect.identifier_preparer.quote(element.schema)}.' if element.schema else ''
    return 'REFRESH MATERIALIZED VIEW {concurrently}{schema_prefix}{name}'.format(
        concurrently='CONCURRENTLY ' if element.concurrently else '',
        schema_prefix=schema_prefix,
        name=compiler.dialect.identifier_preparer.quote(element.name),
    )


def refresh_materialized_view(session, name, *, schema=None, concurrently=False):
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
