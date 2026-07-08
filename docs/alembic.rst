View Migrations
===============

.. warning::

   If upgrading from a previous version, ``__view_cascade_on_drop__`` has
   been renamed to ``__view_cascade__``. The old name is no longer honored.
   ``CreateViewOp(replace=True)`` is deprecated; use ``op.replace_view()``
   instead.

Quick start
-----------

Activate view autogenerate support in your Alembic ``env.py``. This must be
called **before** ``context.configure()``.

Requires ``pip install sqlalchemy-utils[alembic]``.

.. code-block:: python

    from sqlalchemy_utils import register_view_comparator
    register_view_comparator()

    # Point Alembic at the same MetaData your views were registered on.
    # For ORM projects, this is your declarative Base's MetaData:
    from your_app.models import Base  # your declarative Base
    target_metadata = Base.metadata

Operations reference
---------------------

.. autoclass:: sqlalchemy_utils.alembic.operations.CreateViewOp
   :members: create_view

.. autoclass:: sqlalchemy_utils.alembic.operations.DropViewOp
   :members: drop_view

.. autoclass:: sqlalchemy_utils.alembic.operations.ReplaceViewOp
   :members: replace_view

.. autoclass:: sqlalchemy_utils.alembic.operations.CreateMaterializedViewOp
   :members: create_materialized_view

.. autoclass:: sqlalchemy_utils.alembic.operations.DropMaterializedViewOp
   :members: drop_materialized_view

.. autoclass:: sqlalchemy_utils.alembic.operations.ReplaceMaterializedViewOp
   :members: replace_materialized_view

.. autoclass:: sqlalchemy_utils.alembic.operations.RefreshMaterializedViewOp
   :members: refresh_materialized_view

.. note::

   The ``cascade`` keyword differs between the Alembic and runtime APIs.
   The Alembic ``op.drop_view(...)`` / ``op.drop_materialized_view(...)``
   operations use ``cascade=`` (Alembic convention); the runtime DDL helpers
   :func:`~sqlalchemy_utils.view.create_view` and
   :func:`~sqlalchemy_utils.view.create_materialized_view` use
   ``cascade_on_drop=`` for the same concept.  Writing
   ``op.drop_view(..., cascade_on_drop=True)`` raises ``TypeError`` — use
   ``cascade=True`` with the ``op.*`` helpers.

Autogenerate
------------

Alembic's autogenerate mode monitors database changes. SQLAlchemy-Utils provides a
comparator for view DDL:

* How it works: Each model view is temporarily created inside a savepoint,
  its definition read from PostgreSQL, then the savepoint is rolled back.

* **PostgreSQL only**: View autogenerate comparison queries
  ``pg_views``/``pg_matviews`` and uses savepoints.  On non-PostgreSQL
  dialects the comparator logs a warning and skips view diffing.

  Offline mode (``alembic upgrade --sql``) for *autogenerate* is
  unsupported: the comparator needs a live connection to introspect
  ``pg_catalog`` and create the model view inside a savepoint, so view
  diffing is skipped when running ``alembic revision --autogenerate``
  with ``--sql``.  This limitation applies only to *autogenerate*;
  **applying** already-generated migrations in ``--sql`` mode works
  normally, because the migration operations themselves are plain DDL
  that Alembic renders to a script without needing a database
  connection.

* What it detects:

  - New views that need to be created
  - Existing views no longer defined in your models
  - Changed view definitions (detected via canonicalization)

  ``__view_replace__`` (ViewMixin) controls whether runtime DDL emits
  ``CREATE OR REPLACE VIEW``; autogenerate emits ``ReplaceViewOp`` when
  the definition changes regardless of this flag.

* Known limitations:

  - PostgreSQL does not support ``CREATE OR REPLACE MATERIALIZED VIEW``,
    so ``replace_materialized_view`` issues a ``DROP`` followed by ``CREATE``.
  - Cascade (``CASCADE``/``RESTRICT``) is configurable per-view via
    ``__view_cascade__`` on ``ViewMixin`` subclasses or the ``cascade_on_drop``
    argument to :func:`~sqlalchemy_utils.view.create_view` /
    :func:`~sqlalchemy_utils.view.create_materialized_view`. The autogenerate
    comparator propagates this preference to the generated ``drop_view`` /
    ``drop_materialized_view`` ops.
  - Dependency detection between views uses word-boundary regex matching
    on view names in compiled SQL, not SQL-AST parsing.  View names that
    match common SQL identifiers (e.g. ``id``, ``name``, ``count``,
    ``select``, ``from``, ``user``) may produce false dependency edges;
    prefer distinctive view names.  Full SQL-AST parsing is not yet
    implemented.

.. warning::

   ``with_data`` differs between the three contexts in which a materialized
   view can be created:

   1. **Runtime DDL** — :func:`~sqlalchemy_utils.view.create_materialized_view`
      emits ``CREATE MATERIALIZED VIEW ... WITH DATA`` by default (the
      PostgreSQL default); pass ``with_data=False`` to defer population.
   2. **Manual Alembic op** — ``op.create_materialized_view(...)`` defaults to
      ``with_data=False`` (``WITH NO DATA``), so the MV is created empty.
      Pass ``with_data=True`` explicitly to populate it immediately.
   3. **Autogenerate** — always emits ``with_data=False`` (unpopulated) to
      avoid expensive data population during migrations.  After applying an
      autogenerated ``create_materialized_view`` or
      ``replace_materialized_view`` op, run
      ``op.refresh_materialized_view(...)`` to populate the MV.

env.py snippet (additions to your existing env.py)
--------------------------------------------------

.. code-block:: python

    # Add these lines to your existing env.py
    """Migrations config."""
    from logging.config import fileConfig

    from alembic import context
    from sqlalchemy import engine_from_config
    from sqlalchemy import pool
    from sqlalchemy_utils import register_view_comparator

    # Must call before context.configure()
    register_view_comparator()

    # Import your models for autogenerate to detect
    from your_app.models import Base  # your declarative Base

    # Interpret the config file for Alembic config.
    config = context.config
    if config.config_file_name is not None:
        fileConfig(config.config_file_name)

    # Add your model's MetaData here for autogenerate to compare against
    # Use the declarative Base's ``MetaData``; view/table schemas are configured
    # per-table via ``__table_args__ = {'schema': 'public'}`` or ``__view_schema__``.
    target_metadata = Base.metadata

    # ... rest of usual alembic setup ...

.. note::

   For view-backed models (``ViewMixin``), ``__view_schema__`` takes
   precedence over ``__table_args__['schema']``. The autogenerate
   comparator and migration operations emit the resolved schema for the
   view's ``CREATE``/``DROP``/``REFRESH`` statements.


SA2 Core (non-ORM) pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import sqlalchemy as sa
    from sqlalchemy_utils import create_view, create_materialized_view

    metadata = sa.MetaData()
    create_view("my_view", sa.select(sa.column("id", sa.Integer)), metadata)
    create_materialized_view(
        "my_mv", sa.select(sa.column("id", sa.Integer)), metadata
    )

    # In env.py: target_metadata = metadata


Migrating an existing project
-----------------------------

If your database already contains views created outside SQLAlchemy-Utils (for
example, hand-written ``CREATE VIEW`` scripts run directly against the
database), adopting the autogenerate integration needs a little care.

1. **Before enabling**, add ``create_view(...)`` / ``ViewMixin`` definitions
   for every existing database view to your models or ``MetaData``.  The
   comparator can only diff views it knows about.

2. **First autogenerate** should produce no view operations when your model
   definitions match the database definitions exactly.  Treat a clean
   autogenerate run as the signal that your models and schema are in sync.
   Note that the comparator **canonicalizes** every model view by creating
   it inside a savepoint, reading its definition back from
   ``pg_views``/``pg_matviews``, then rolling back — so the definitions
   are compared in their PostgreSQL-canonical form, not as written in
   your model SQL.

3. **If you cannot model them all at once**, review every generated
   ``drop_view`` op carefully before applying it.  The first run will propose
   dropping each existing view that is not yet represented in your models.

4. **Consider ``--sql`` mode**.  Run the first migration with
   ``alembic upgrade --sql`` so you can inspect the generated SQL before it
   touches the database.

5. **Rename ``__view_cascade_on_drop__`` to ``__view_cascade__``**.  This
   attribute was renamed to align with SQL/Alembic naming.  If your
   ``ViewMixin`` subclasses set ``__view_cascade_on_drop__``, rename it;
   the old name is no longer honored.

6. **Replace ``CreateViewOp(replace=True)`` with ``op.replace_view()``**.
   The ``replace=True`` keyword on ``CreateViewOp`` is deprecated; use
   :meth:`~sqlalchemy_utils.alembic.operations.ReplaceViewOp.replace_view`
   (or ``ReplaceViewOp`` directly) for view-replacement migrations.

.. warning::

   The first autogenerate run against a legacy database will propose dropping
   every existing view that is not declared in your models.  Never apply such
   a migration blindly; inspect the generated ``drop_view`` operations and
   either add the missing view definitions to your models or remove the
   ``drop_view`` ops from the migration script.


Downgrade generation
--------------------

Create-family ops always reverse to a Drop. Drop-family and Replace-family
ops require a stored ``definition``/``old_definition``; without one they
raise ``NotImplementedError``.

.. note::

   ``RefreshMaterializedViewOp.reverse()`` always raises
   ``NotImplementedError`` because ``REFRESH MATERIALIZED VIEW`` is not
   meaningfully reversible (a refresh only repopulates data; there is no SQL
   "un-refresh").  Remove the op from autogenerate's ``downgrade()`` **or**
   replace it with a hand-written ``op.refresh_materialized_view(...)``
   call rather than relying on autogenerate to reverse it.

API reference
-------------

.. autofunction:: sqlalchemy_utils.alembic.comparator.register_view_comparator

.. autoclass:: sqlalchemy_utils.view_record.ViewRecord
   :members:
   :exclude-members: __post_init__,__eq__,__hash__,__repr__

Extending
---------

The integration is built on three standard Alembic extension points.
``comparators.dispatch_for("schema")`` registers a function that Alembic
calls during ``--autogenerate`` to diff a slice of the schema — view
detection is registered this way by :func:`register_view_comparator`.
``Operations.register_operation`` exposes a new ``op.<name>(...)`` helper
and its backing ``MigrateOperation`` subclass (the seven ``*ViewOp``
classes are registered this way).  Finally,
``renderers.dispatch_for(<OpClass>)`` teaches Alembic's offline SQL
renderer how to emit ``CREATE``/``DROP``/``REFRESH`` DDL for each custom
operation.  To add a new view-related operation, subclass
``MigrateOperation``, register it with ``Operations.register_operation``,
implement ``reverse()`` for downgrade generation, and add a renderer via
``renderers.dispatch_for``.


Advanced helpers
----------------

The following helpers are used internally by
:func:`register_view_comparator`.  ``compare_views`` is the Alembic
``"schema"`` comparator entry point registered by
:func:`register_view_comparator`; it is internal and not part of the
public API.  The remaining helpers (``resolve_create_order``,
``resolve_drop_order``, ``get_database_views``,
``get_database_materialized_views``, ``get_dependent_views``) are exposed
in ``sqlalchemy_utils.alembic.__all__`` and are safe for advanced
callers to use directly.

.. autofunction:: sqlalchemy_utils.alembic.depend.resolve_create_order
.. autofunction:: sqlalchemy_utils.alembic.depend.resolve_drop_order

.. autofunction:: sqlalchemy_utils.alembic.pg_catalog.get_database_views
.. autofunction:: sqlalchemy_utils.alembic.pg_catalog.get_database_materialized_views
.. autofunction:: sqlalchemy_utils.alembic.pg_catalog.get_dependent_views
