"""PostgreSQL catalog query module for SQLAlchemy-Utils Alembic integration."""

from __future__ import annotations

import sqlalchemy as sa


def _query_view_catalog(connection: sa.engine.Connection, table: str, name_col: str, schema: str | None = None) -> dict[str, str]:
    """Query a pg_catalog view table for names and definitions.

    Args:
        connection: SQLAlchemy Connection object.
        table: Catalog table name (e.g. ``"pg_views"``, ``"pg_matviews"``).
        name_col: Column name holding the view name (e.g. ``"viewname"``).
        schema: Optional schema name filter. If None, only the connection's
            current default schema is queried (via ``current_schema()``).

    Returns:
        Dictionary mapping view name to definition SQL.
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

    Args:
        connection: SQLAlchemy Connection object.
        schema: Optional schema name filter. If None, only the connection's
            current default schema is queried (via ``current_schema()``).

    Returns:
        Dictionary mapping view_name to definition SQL.

    Example:
        >>> views = get_database_views(connection)  # current schema only
        >>> views = get_database_views(connection, schema="public")
    """
    return _query_view_catalog(connection, "pg_views", "viewname", schema)


def get_database_materialized_views(connection: sa.engine.Connection, schema: str | None = None) -> dict[str, str]:
    """Query pg_matviews catalog for materialized view names and definitions.

    PostgreSQL-specific. Will raise on non-PostgreSQL dialects.

    Args:
        connection: SQLAlchemy Connection object.
        schema: Optional schema name filter. If None, only the connection's
            current default schema is queried (via ``current_schema()``).

    Returns:
        Dictionary mapping matviewname to definition SQL.

    Example:
        >>> mvs = get_database_materialized_views(connection)  # current schema only
        >>> mvs = get_database_materialized_views(connection, schema="public")
    """
    return _query_view_catalog(connection, "pg_matviews", "matviewname", schema)


def get_dependent_views(connection: sa.engine.Connection, view_name: str, schema: str | None = None) -> dict[tuple[str, str | None], str]:
    """Query pg_depend for views that depend on the given view.

    PostgreSQL-specific. Will raise on non-PostgreSQL dialects.

    Args:
        connection: SQLAlchemy Connection object.
        view_name: Name of the view to check dependents for.
        schema: Optional schema name. If None, returns dependents in any
            schema. When provided, both the dependent and referenced views
            are constrained to this schema.

    Returns:
        Dictionary mapping ``(dependent_name, dependent_schema)`` to its
        definition. Keying by the ``(name, schema)`` tuple avoids
        cross-schema name collisions: two dependent views sharing a name
        in different schemas would otherwise collide and the second would
        overwrite the first. Empty dict if no dependents.
    """
    schema_clause = (
        " AND v.schemaname = :schema AND refn.nspname = :schema"
        if schema
        else ""
    )
    params: dict[str, str] = {"view_name": view_name}
    if schema:
        params["schema"] = schema
    # ``pg_depend`` columns: classid, objid, objsubid, refclassid, refobjid,
    # refobjsubid, deptype. To resolve the referenced object's name we join
    # ``pg_class ref`` on ``dep.refobjid = ref.oid`` and filter on
    # ``ref.relname``. The dependent view's name comes from ``c.relname``.
    # Join ``pg_namespace`` on both sides to qualify by schema, preventing
    # cross-schema name collisions. A UNION ALL joins regular views
    # (pg_views) and materialized views (pg_matviews) so both dependent
    # kinds are returned; the two subqueries are disjoint by ``relkind``,
    # so UNION ALL avoids silently deduping same-named regular + MV rows
    # that a plain UNION would collapse. ``pg_views``/``pg_matviews`` carry
    # both the dependent view's schema (schemaname) and definition
    # (definition); the join ties the dependent class row to its catalog
    # entry via (schema, name). When ``schema`` is given, both the
    # dependent view's schema (``v.schemaname``) AND the referenced view's
    # namespace (``refn.nspname``) are filtered to avoid false positives
    # from same-named referenced views in other schemas.
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
