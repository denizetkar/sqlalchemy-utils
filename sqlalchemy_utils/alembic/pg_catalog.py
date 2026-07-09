"""PostgreSQL catalog query module for SQLAlchemy-Utils Alembic integration."""

from __future__ import annotations

import sqlalchemy as sa


def _assert_postgres(connection: sa.engine.Connection) -> None:
    """Fail fast if *connection* is not backed by a PostgreSQL dialect.

    The pg_catalog queries in this module are PostgreSQL-specific.  Calling
    them against any other dialect produces confusing low-level errors
    (e.g. "no such table: pg_views" on SQLite) instead of a clear message.

    :param connection: SQLAlchemy Connection object.
    :raises NotImplementedError: if the connection dialect is not PostgreSQL.
    """
    if connection.dialect.name != 'postgresql':
        raise NotImplementedError(
            f"pg_catalog queries require PostgreSQL; got dialect "
            f"'{connection.dialect.name}'"
        )


def _query_view_catalog(connection: sa.engine.Connection, table: str, name_col: str, schema: str | None = None) -> dict[str, str]:
    """Query a pg_catalog view table for names and definitions.

    :param connection: SQLAlchemy Connection object.
    :param table: Catalog table name (e.g. ``"pg_views"``, ``"pg_matviews"``).
    :param name_col: Column name holding the view name (e.g. ``"viewname"``).
    :param schema: Optional schema name filter. If None, only the
        connection's current default schema is queried (via
        ``current_schema()``).
    :returns: Dictionary mapping view name to definition SQL.
    """
    if not schema:  # catches None and ""
        sql = sa.text(
            f"SELECT {name_col}, definition FROM {table} "
            "WHERE schemaname = current_schema()"
        )
        result = connection.execute(sql)
    else:
        sql = sa.text(
            f"SELECT {name_col}, definition FROM {table} "
            "WHERE schemaname = :schema"
        )
        result = connection.execute(sql, {"schema": schema})
    return {getattr(row, name_col): row.definition for row in result}


def get_database_views(connection: sa.engine.Connection, schema: str | None = None) -> dict[str, str]:
    """Query pg_views catalog for view names and definitions.

    PostgreSQL-specific. Will raise on non-PostgreSQL dialects.

    :param connection: SQLAlchemy Connection object.
    :param schema: Optional schema name filter. If None, only the
        connection's current default schema is queried (via
        ``current_schema()``).
    :returns: Dictionary mapping view_name to definition SQL.
    :raises NotImplementedError: if the connection dialect is not PostgreSQL.

    Example::

        >>> views = get_database_views(connection)  # current schema only
        >>> views = get_database_views(connection, schema="public")
    """
    _assert_postgres(connection)
    return _query_view_catalog(connection, "pg_views", "viewname", schema)


def get_database_materialized_views(connection: sa.engine.Connection, schema: str | None = None) -> dict[str, str]:
    """Query pg_matviews catalog for materialized view names and definitions.

    PostgreSQL-specific. Will raise on non-PostgreSQL dialects.

    :param connection: SQLAlchemy Connection object.
    :param schema: Optional schema name filter. If None, only the
        connection's current default schema is queried (via
        ``current_schema()``).
    :returns: Dictionary mapping matviewname to definition SQL.
    :raises NotImplementedError: if the connection dialect is not PostgreSQL.

    Example::

        >>> mvs = get_database_materialized_views(connection)  # current schema only
        >>> mvs = get_database_materialized_views(connection, schema="public")
    """
    _assert_postgres(connection)
    return _query_view_catalog(connection, "pg_matviews", "matviewname", schema)


def get_dependent_views(connection: sa.engine.Connection, name: str, schema: str | None = None) -> dict[tuple[str, str | None], str]:
    """Query pg_depend for views that depend on the given view.

    PostgreSQL-specific. Will raise on non-PostgreSQL dialects.

    :param connection: SQLAlchemy Connection object.
    :param name: Name of the view to check dependents for.
    :param schema: Optional schema name. If None, returns dependents in
        any schema. When provided, both the dependent and referenced
        views are constrained to this schema.
    :returns: Dictionary mapping ``(dependent_name, dependent_schema)``
        to its definition. Keying by the ``(name, schema)`` tuple avoids
        cross-schema name collisions: two dependent views sharing a name
        in different schemas would otherwise collide and the second would
        overwrite the first. Empty dict if no dependents.
    :raises NotImplementedError: if the connection dialect is not PostgreSQL.
    """
    _assert_postgres(connection)
    schema_clause = (
        " AND v.schemaname = :schema AND refn.nspname = :schema"
        if schema
        else ""
    )
    params: dict[str, str] = {"view_name": name}
    if schema:
        params["schema"] = schema
    # pg_depend joins pg_rewrite/pg_class to resolve dependent and referenced
    # view names; pg_namespace qualifies both sides to prevent cross-schema
    # name collisions. UNION ALL of pg_views and pg_matviews keeps same-named
    # regular + MV rows distinct (plain UNION would dedupe them). When *schema*
    # is given, both the dependent (v.schemaname) and referenced (refn.nspname)
    # schemas are filtered to avoid false positives from other schemas.
    base_select = (
        "SELECT c.relname AS dependent_name, "
        "v.definition AS dependent_definition, "
        "v.schemaname AS dependent_schema "
        "FROM pg_depend dep "
        "JOIN pg_rewrite r ON dep.objid = r.oid "
        "JOIN pg_class c ON r.ev_class = c.oid "
        "JOIN pg_namespace cn ON c.relnamespace = cn.oid "
        "JOIN pg_class ref ON dep.refobjid = ref.oid "
        "JOIN pg_namespace refn ON ref.relnamespace = refn.oid "
        "{view_catalog_join} "
        "WHERE ref.relname = :view_name "
        "AND dep.refobjid != dep.objid "
        "AND c.relname != :view_name "
        "AND cn.nspname = v.schemaname"
        + schema_clause
    )
    views_select = base_select.format(
        view_catalog_join="JOIN pg_views v ON c.relname = v.viewname"
    )
    matviews_select = base_select.format(
        view_catalog_join="JOIN pg_matviews v ON c.relname = v.matviewname"
    )
    sql = sa.text(f"{views_select} UNION ALL {matviews_select}")
    result = connection.execute(sql, params)
    return {
        (row.dependent_name, row.dependent_schema): row.dependent_definition
        for row in result
    }
