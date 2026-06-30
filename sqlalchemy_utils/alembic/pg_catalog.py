"""PostgreSQL catalog query module for SQLAlchemy-Utils Alembic integration."""

from __future__ import annotations

import sqlalchemy as sa


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
    if not schema:  # catches None and ""
        sql = sa.text(
            "SELECT viewname, definition FROM pg_views "
            "WHERE schemaname NOT IN ('information_schema', 'pg_catalog')"
        )
        result = connection.execute(sql)
    else:
        sql = sa.text(
            "SELECT viewname, definition FROM pg_views "
            "WHERE schemaname = :schema"
        )
        result = connection.execute(sql, {"schema": schema})
    views = {row.viewname: row.definition for row in result}
    return views


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
    if not schema:  # catches None and ""
        sql = sa.text(
            "SELECT matviewname, definition FROM pg_matviews "
            "WHERE schemaname NOT IN ('information_schema', 'pg_catalog')"
        )
        result = connection.execute(sql)
    else:
        sql = sa.text(
            "SELECT matviewname, definition FROM pg_matviews "
            "WHERE schemaname = :schema"
        )
        result = connection.execute(sql, {"schema": schema})
    mvs = {row.matviewname: row.definition for row in result}
    return mvs


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
    if not schema:
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
        )
        result = connection.execute(sql, {"view_name": view_name})
    else:
        sql = sa.text(
            "SELECT dep.refobjname AS dependent_name, "
            "v.definition AS dependent_definition "
            "FROM pg_depend dep "
            "JOIN pg_rewrite r ON dep.objid = r.oid "
            "JOIN pg_class c ON r.ev_class = c.oid "
            "JOIN pg_views v ON c.relname = v.viewname "
            "WHERE dep.refobjname = :view_name "
            "AND dep.refobjid != dep.objid "
            "AND c.relname != :view_name "
            "AND v.schemaname = :schema"
        )
        result = connection.execute(sql, {"view_name": view_name, "schema": schema})
    return {row.dependent_name: row.dependent_definition for row in result}
