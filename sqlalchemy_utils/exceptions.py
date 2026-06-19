"""
Global SQLAlchemy-Utils exception classes.
"""


class ImproperlyConfigured(Exception):
    """
    SQLAlchemy-Utils is improperly configured; normally due to usage of
    a utility that depends on a missing library.
    """


class ViewReadonlyError(Exception):
    """Raised when attempting to modify a view-backed ORM instance."""
