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

ViewMixin
---------

.. autoclass:: sqlalchemy_utils.view_mixin.ViewMixin
   :members:


ViewReadonlyError
-----------------

.. autoexception:: sqlalchemy_utils.exceptions.ViewReadonlyError

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

.. autoclass:: sqlalchemy_utils.view.RefreshMaterializedView
   :members:
