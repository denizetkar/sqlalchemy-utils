"""PostgreSQL catalog query module for SQLAlchemy-Utils Alembic integration."""

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
    if schema is None:
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
    if schema is None:
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
