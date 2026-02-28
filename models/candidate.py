"""
TalentScout — Candidate Data Model
====================================
Defines the structured data model for a candidate interview session.
Uses Pydantic for automatic validation and serialization.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Question Result (per-answer scoring record)
# ---------------------------------------------------------------------------

class QuestionResult(BaseModel):
    """Stores a single technical question, candidate answer, LLM score, and feedback."""

    question: str = Field(description="The technical question asked.")
    answer: str = Field(description="The candidate's raw answer text.")
    score: int = Field(default=0, ge=0, le=5, description="LLM-assigned score from 0–5.")
    feedback: str = Field(default="", description="Short LLM feedback (1–2 lines).")


# ---------------------------------------------------------------------------
# Core Candidate Profile
# ---------------------------------------------------------------------------

class Candidate(BaseModel):
    """
    Represents all structured information collected from a job candidate
    during the TalentScout interview session.

    Fields are populated progressively as the conversation advances
    through each collection state in the state machine.
    """

    full_name: Optional[str] = Field(default=None, description="Candidate's full legal name.")
    email: Optional[str] = Field(default=None, description="Candidate's contact email address.")
    phone: Optional[str] = Field(default=None, description="Candidate's phone number.")
    years_experience: Optional[int] = Field(
        default=None, ge=0, le=40, description="Years of professional tech experience."
    )
    desired_position: Optional[str] = Field(default=None, description="Job role applying for.")
    location: Optional[str] = Field(default=None, description="Current city/country.")
    tech_stack: List[str] = Field(
        default_factory=list, description="Technologies the candidate is proficient in."
    )

    @field_validator("tech_stack", mode="before")
    @classmethod
    def normalize_tech_stack(cls, v):
        """Ensure all tech stack entries are stripped of whitespace."""
        if isinstance(v, list):
            return [item.strip().title() for item in v if item.strip()]
        return v

    def is_fully_collected(self) -> bool:
        """Returns True only when all 7 required fields have been filled."""
        return all([
            self.full_name,
            self.email,
            self.phone,
            self.years_experience is not None,
            self.desired_position,
            self.location,
            bool(self.tech_stack),
        ])

    def to_summary(self) -> str:
        """Human-readable summary of the candidate profile."""
        stack = ", ".join(self.tech_stack) if self.tech_stack else "Not provided"
        return (
            f"**Name:** {self.full_name or '—'}\n"
            f"**Email:** {self.email or '—'}\n"
            f"**Phone:** {self.phone or '—'}\n"
            f"**Experience:** {self.years_experience if self.years_experience is not None else '—'} years\n"
            f"**Desired Role:** {self.desired_position or '—'}\n"
            f"**Location:** {self.location or '—'}\n"
            f"**Tech Stack:** {stack}"
        )


# ---------------------------------------------------------------------------
# Session Wrapper
# ---------------------------------------------------------------------------

class CandidateSession(BaseModel):
    """
    Wraps a Candidate with session-level metadata: unique ID, timestamps,
    generated questions, and per-question scored results.
    """

    session_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this interview session.",
    )
    candidate: Candidate = Field(
        default_factory=Candidate,
        description="The structured candidate profile being collected.",
    )
    tech_questions: List[str] = Field(
        default_factory=list,
        description="LLM-generated technical questions for this candidate.",
    )
    results: List[QuestionResult] = Field(
        default_factory=list,
        description="Scored results — one entry per answered question.",
    )
    current_q_index: int = Field(
        default=0,
        description="Index of the current question being asked (0-based).",
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="ISO 8601 timestamp when the session was initiated.",
    )
    completed_at: Optional[str] = Field(
        default=None,
        description="ISO 8601 timestamp when the session concluded.",
    )
    is_complete: bool = Field(
        default=False,
        description="Flag set to True when the session ends gracefully.",
    )

    # ------------------------------------------------------------------
    # Scoring properties
    # ------------------------------------------------------------------

    @property
    def total_score(self) -> int:
        """Sum of all question scores."""
        return sum(r.score for r in self.results)

    @property
    def max_score(self) -> int:
        """Maximum achievable score (5 points × number of questions answered)."""
        return len(self.results) * 5

    @property
    def percentage(self) -> float:
        """Score as a percentage, or 0.0 if no questions answered."""
        if self.max_score == 0:
            return 0.0
        return round((self.total_score / self.max_score) * 100, 1)

    @property
    def performance_label(self) -> str:
        """
        Human-readable performance tier based on percentage:
            ≥ 80% → Strong Candidate
            60–79% → Moderate
            < 60% → Needs Improvement
        """
        pct = self.percentage
        if pct >= 80:
            return "Strong Candidate"
        if pct >= 60:
            return "Moderate"
        return "Needs Improvement"

    @property
    def strengths(self) -> List[str]:
        """Questions scored 4–5 (strong answers)."""
        return [
            r.question for r in self.results if r.score >= 4
        ]

    @property
    def areas_to_improve(self) -> List[str]:
        """Questions scored 0–2 (weak answers)."""
        return [
            r.question for r in self.results if r.score <= 2
        ]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def mark_complete(self) -> None:
        """Marks the session as completed and records the end timestamp."""
        self.is_complete = True
        self.completed_at = datetime.utcnow().isoformat()

    def model_dump_safe(self) -> dict:
        """Returns a JSON-serializable dict safe for file/DB persistence."""
        return self.model_dump()
