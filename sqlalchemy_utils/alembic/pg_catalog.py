"""PostgreSQL catalog query module for SQLAlchemy-Utils Alembic integration."""

import sqlalchemy as sa


def get_database_views(connection, schema: str | None = None) -> dict[str, str]:
    """Query pg_views catalog for view names and definitions.

    Args:
        connection: SQLAlchemy Connection object from `conftest.py`.
        schema: Optional schema name filter. If None, defaults to 'public'.

    Returns:
        Dictionary mapping view_name to definition SQL.

    Example:
        >>> views = get_database_views(connection)  # Get all schemas
        >>> views = get_database_views(connection, schema="public")
    """
    sql = sa.text(
        "SELECT viewname, definition FROM pg_views "
        "WHERE schemaname = :schema OR (:schema IS NULL AND schemaname = 'public')"
    )
    result = connection.execute(sql, {"schema": schema})
    views = {row.viewname: row.definition for row in result}
    return views


def get_database_materialized_views(connection, schema: str | None = None) -> dict[str, str]:
    """Query pg_matviews catalog for materialized view names and definitions.

    Args:
        connection: SQLAlchemy Connection object from `conftest.py`.
        schema: Optional schema name filter. If None, defaults to 'public'.

    Returns:
        Dictionary mapping matviewname to definition SQL.

    Example:
        >>> mvs = get_database_materialized_views(connection)
        >>> mvs = get_database_materialized_views(connection, schema="public")
    """
    sql = sa.text(
        "SELECT matviewname, definition FROM pg_matviews "
        "WHERE schemaname = :schema OR (:schema IS NULL AND schemaname = 'public')"
    )
    result = connection.execute(sql, {"schema": schema})
    mvs = {row.matviewname: row.definition for row in result}
    return mvs
