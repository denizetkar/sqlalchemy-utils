"""Backward-compatible re-export shim.

ViewRecord now lives at :mod:`sqlalchemy_utils.view_record` (outside the
``alembic`` package) so that ``view.py`` / ``view_mixin.py`` can import it
without pulling in alembic. This module re-exports it under the old path
for code that imports ``sqlalchemy_utils.alembic.view_record``.
"""
from __future__ import annotations

from sqlalchemy_utils.view_record import ViewRecord

__all__ = ["ViewRecord"]
