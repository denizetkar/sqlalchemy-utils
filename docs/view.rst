View utilities
==============


create_view
-----------

.. autofunction:: sqlalchemy_utils.view.create_view


create_materialized_view
------------------------

.. autofunction:: sqlalchemy_utils.view.create_materialized_view


refresh_materialized_view
-------------------------

.. autofunction:: sqlalchemy_utils.view.refresh_materialized_view

.. note::

   For ORM models using :class:`~sqlalchemy_utils.view_mixin.ViewMixin`,
   prefer :meth:`~sqlalchemy_utils.view_mixin.ViewMixin.refresh`
   which resolves the schema automatically.

create_table_from_selectable
----------------------------

.. autofunction:: sqlalchemy_utils.view.create_table_from_selectable

ViewMixin
---------

.. autoclass:: sqlalchemy_utils.view_mixin.ViewMixin
   :members:


ViewReadonlyError
-----------------

.. autoexception:: sqlalchemy_utils.exceptions.ViewReadonlyError

.. note::

   :class:`sqlalchemy_utils.view_record.ViewRecord` (the metadata-attached
   record describing a registered view) is documented in
   :doc:`alembic <alembic>` since it is primarily consumed by the Alembic
   autogenerate integration.

DDL constructs
--------------

These low-level DDL constructs are used internally by
:func:`sqlalchemy_utils.view.create_view` /
:func:`sqlalchemy_utils.view.create_materialized_view`; you typically do not
need to instantiate them directly.

.. autoclass:: sqlalchemy_utils.view.CreateView
   :members:

.. autoclass:: sqlalchemy_utils.view.DropView
   :members:

.. note::

   ``DropView(cascade=...)`` mirrors the Alembic ``op.drop_view`` naming
   convention; the runtime helper :func:`~sqlalchemy_utils.view.create_view`
   uses ``cascade_on_drop=`` for the same concept.

.. autoclass:: sqlalchemy_utils.view.RefreshMaterializedView
   :members:
