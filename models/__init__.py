"""
TalentScout — Models Package
Exports the core data models used across the application.
"""
from .candidate import Candidate, CandidateSession

__all__ = ["Candidate", "CandidateSession"]
