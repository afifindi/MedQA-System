"""
Medical QA System - Configuration Package.

Exports Settings and get_settings singleton.
"""

from app.core.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
