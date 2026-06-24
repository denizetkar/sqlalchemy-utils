View Migrations
===============

Quick start
-----------

Activate view autogenerate support in your Alembic ``env.py``. This must be
called **before** ``context.configure()``.

::

    from sqlalchemy_utils.alembic.comparator import register_view_comparator
    register_view_comparator()

Operations reference
---------------------

.. autofunction:: sqlalchemy_utils.alembic.operations.CreateViewOp.create_view
.. autofunction:: sqlalchemy_utils.alembic.operations.DropViewOp.drop_view
.. autofunction:: sqlalchemy_utils.alembic.operations.ReplaceViewOp.replace_view
.. autofunction:: sqlalchemy_utils.alembic.operations.CreateMaterializedViewOp.create_materialized_view
.. autofunction:: sqlalchemy_utils.alembic.operations.DropMaterializedViewOp.drop_materialized_view
.. autofunction:: sqlalchemy_utils.alembic.operations.ReplaceMaterializedViewOp.replace_materialized_view

Autogenerate
------------

Alembic's autogenerate mode monitors database changes. SQLAlchemy-Utils provides a
comparator for view DDL:

* How it works: Each model view is temporarily created inside a savepoint,
  its definition read from PostgreSQL, then the savepoint is rolled back.

* **PostgreSQL only**: View autogenerate comparison queries
  ``pg_views``/``pg_matviews`` and uses savepoints.  On non-PostgreSQL
  dialects the comparator logs a warning and skips view diffing.

* What it detects:

  - New views that need to be created
  - Existing views no longer defined in your models
  - Changed view definitions matching ``CREATE OR REPLACE`` logic

* Known limitations:

  - PostgreSQL does not support ``CREATE OR REPLACE MATERIALIZED VIEW``,
    so ``replace_materialized_view`` issues a ``DROP`` followed by ``CREATE``.
  - ``op.create_materialized_view`` defaults to ``WITH DATA`` (matching
    PostgreSQL's default).  Pass ``with_data=False`` to create an unpopulated
    MV (useful for large datasets during migration).
  - ``cascade_on_drop`` controls (``CASCADE``/``RESTRICT``) are not yet
    configurable per-view in autogenerate.

Full example
------------

.. code-block:: python

    """Migrations config."""
    from logging.config import fileConfig

    from alembic import context
    from sqlalchemy import engine_from_config
    from sqlalchemy import pool
    from sqlalchemy_utils.alembic.comparator import register_view_comparator

    # Override default ConfigOptions to accommodate savepoint-style canonicalization
    # context.config_set_main_option('compare_type', True)
    # context.config_set_main_option('compare_server_default', True)

    # Must call before context.configure()
    register_view_comparator()

    # Import your models for autogenerate to detect
    from your_app.models import User, ItemView
    from your_app.models import Base

    # Interpret the config file for Alembic config.
    config = context.config
    if config.config_file_name is not None:
        fileConfig(config.config_file_name)

    # Add your model's MetaData here for autogenerate to compare against
    # Use the declarative Base's ``MetaData``; view/table schemas are configured
    # per-table via ``__table_args__ = {'schema': 'public'}`` or ``__view_schema__``.
    target_metadata = Base.metadata

    # ... rest of usual alembic setup ...


Dependencies
------------

* Requires PostgreSQL for ``pg_views``/``pg_matviews`` catalog access
* Requires ``alembic`` to be installed, or ``sqlalchemy_utils[alembic]`` extra
