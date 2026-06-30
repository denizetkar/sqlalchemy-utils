"""PostgreSQL catalog query module for SQLAlchemy-Utils Alembic integration."""

from __future__ import annotations

import sqlalchemy as sa


def _query_view_catalog(connection, table: str, name_col: str, schema: str | None = None) -> dict[str, str]:
    """Query a pg_catalog view table for names and definitions.

    Args:
        connection: SQLAlchemy Connection object.
        table: Catalog table name (e.g. ``"pg_views"``, ``"pg_matviews"``).
        name_col: Column name holding the view name (e.g. ``"viewname"``).
        schema: Optional schema name filter. If None, all non-system schemas
            are returned (i.e. every schema except ``information_schema`` and
            ``pg_catalog``).

    Returns:
        Dictionary mapping view name to definition SQL.
    """
    if not schema:  # catches None and ""
        sql = sa.text(
            f"SELECT {name_col}, definition FROM {table} "
            "WHERE schemaname NOT IN ('information_schema', 'pg_catalog')"
        )
        result = connection.execute(sql)
    else:
        sql = sa.text(
            f"SELECT {name_col}, definition FROM {table} "
            "WHERE schemaname = :schema"
        )
        result = connection.execute(sql, {"schema": schema})
    return {getattr(row, name_col): row.definition for row in result}


def get_database_views(connection, schema: str | None = None) -> dict[str, str]:
    """Query pg_views catalog for view names and definitions.

    Args:
        connection: SQLAlchemy Connection object.
        schema: Optional schema name filter. If None, all non-system schemas
            are returned (i.e. every schema except ``information_schema`` and
            ``pg_catalog``).

    Returns:
        Dictionary mapping view_name to definition SQL.

    Example:
        >>> views = get_database_views(connection)  # all non-system schemas
        >>> views = get_database_views(connection, schema="public")
    """
    return _query_view_catalog(connection, "pg_views", "viewname", schema)


def get_database_materialized_views(connection, schema: str | None = None) -> dict[str, str]:
    """Query pg_matviews catalog for materialized view names and definitions.

    Args:
        connection: SQLAlchemy Connection object.
        schema: Optional schema name filter. If None, all non-system schemas
            are returned (i.e. every schema except ``information_schema`` and
            ``pg_catalog``).

    Returns:
        Dictionary mapping matviewname to definition SQL.

    Example:
        >>> mvs = get_database_materialized_views(connection)  # all non-system schemas
        >>> mvs = get_database_materialized_views(connection, schema="public")
    """
    return _query_view_catalog(connection, "pg_matviews", "matviewname", schema)


def get_dependent_views(connection, view_name: str, schema: str | None = None) -> dict[str, str]:
    """Query pg_depend for views that depend on the given view.

    Args:
        connection: SQLAlchemy Connection object.
        view_name: Name of the view to check dependents for.
        schema: Optional schema name. If None, searches all non-system schemas.

    Returns:
        Dictionary mapping dependent view name to its definition.
        Empty dict if no dependents.
    """
    schema_clause = " AND v.schemaname = :schema" if schema else ""
    params: dict[str, str] = {"view_name": view_name}
    if schema:
        params["schema"] = schema
    sql = sa.text(
        "SELECT dep.refobjname AS dependent_name, "
        "v.definition AS dependent_definition "
        "FROM pg_depend dep "
        "JOIN pg_rewrite r ON dep.objid = r.oid "
        "JOIN pg_class c ON r.ev_class = c.oid "
        "JOIN pg_views v ON c.relname = v.viewname "
        "WHERE dep.refobjname = :view_name "
        "AND dep.refobjid != dep.objid "
        "AND c.relname != :view_name"
        + schema_clause
    )
    result = connection.execute(sql, params)
    return {row.dependent_name: row.dependent_definition for row in result}
