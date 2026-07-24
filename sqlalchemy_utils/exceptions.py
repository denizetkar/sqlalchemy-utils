"""
Global SQLAlchemy-Utils exception classes.
"""

import sqlalchemy as sa


class ImproperlyConfigured(Exception):
    """
    SQLAlchemy-Utils is improperly configured; normally due to usage of
    a utility that depends on a missing library.
    """


class ViewReadonlyError(sa.exc.InvalidRequestError):
    """Raised when attempting to modify a view-backed ORM instance. Catchable as ``sa.exc.InvalidRequestError``."""
