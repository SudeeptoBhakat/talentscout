"""
TalentScout — Data Package
Handles persistence and storage of candidate sessions.
"""
from .storage import SessionStorage, BaseSessionStorage

__all__ = ["SessionStorage", "BaseSessionStorage"]
