"""
TalentScout — Input Validators
================================
Pure functions for validating each piece of candidate input collected
during the interview conversation.

Design Principles:
    - Each function accepts a raw string and returns a ``ValidationResult``.
    - No side effects; functions are stateless and independently testable.
    - Regex-based checks are pre-compiled at module level for efficiency.
    - All exit-intent detection is centralised in ``is_exit_intent``.
    - Gibberish/nonsense input is detected and rejected with friendly messages.

Gibberish detection is delegated to ``core.gibberish_detector`` which
uses vowel-ratio, consonant-run, bigram-frequency, and known-technology
matching to catch random keyboard mashing without blocking real inputs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional

from core.gibberish_detector import (
    check_email_gibberish,
    check_location_gibberish,
    check_phone_gibberish,
    check_position_gibberish,
    check_tech_stack,
    gibberish_phrase,
)


# ---------------------------------------------------------------------------
# Pre-compiled Patterns
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)

_PHONE_RE = re.compile(
    r"^\+?[\d\s\-().]{7,20}$"
)

_NAME_RE = re.compile(
    r"^[A-Za-z\u00C0-\u024F\u0900-\u097F]+([\s'\-][A-Za-z\u00C0-\u024F\u0900-\u097F]+)+$"
    # Supports ASCII + Latin Extended + Devanagari (Hindi/Marathi names)
)

#: Keywords that signal the user wants to end the session immediately.
_EXIT_KEYWORDS: frozenset[str] = frozenset({
    "exit", "quit", "bye", "goodbye", "stop", "end",
    "cancel", "close", "leave", "done", "finish",
})

# ---------------------------------------------------------------------------
# Result Type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValidationResult:
    """
    Immutable result returned by every validator.

    Attributes:
        is_valid: True when the input passes all validation rules.
        value:    The cleaned / normalised value, or ``None`` on failure.
        error:    A human-readable error message when ``is_valid`` is False.
    """

    is_valid: bool
    value: Optional[object] = None
    error: Optional[str] = None

    @classmethod
    def ok(cls, value: object) -> "ValidationResult":
        """Factory shortcut for a successful result."""
        return cls(is_valid=True, value=value)

    @classmethod
    def fail(cls, error: str) -> "ValidationResult":
        """Factory shortcut for a failed result."""
        return cls(is_valid=False, error=error)


# ---------------------------------------------------------------------------
# Exit-Intent Detection
# ---------------------------------------------------------------------------

def is_exit_intent(text: str) -> bool:
    """
    Returns True if the candidate's message signals they want to exit.

    Checks whether any token in *text* exactly matches a known exit keyword
    (case-insensitive).

    Examples::

        >>> is_exit_intent("bye")
        True
        >>> is_exit_intent("I want to Quit now")
        True
        >>> is_exit_intent("Python")
        False
    """
    tokens = re.split(r"[\s,!?.]+", text.strip().lower())
    return bool(frozenset(tokens) & _EXIT_KEYWORDS)


# ---------------------------------------------------------------------------
# Individual Field Validators
# ---------------------------------------------------------------------------

def validate_name(raw: str) -> ValidationResult:
    """
    Validates the candidate's full name.

    Rules:
        - At least two words (first + last name minimum).
        - Only alphabetic characters, hyphens, apostrophes, and spaces.
        - Total length between 2 and 80 characters.
        - No gibberish / random keyboard mashing.

    Returns:
        ValidationResult with the title-cased name on success.
    """
    cleaned = raw.strip()

    if not cleaned:
        return ValidationResult.fail(
            "Name cannot be empty. Please enter your full name (e.g. **Jane Doe**)."
        )

    if len(cleaned) > 80:
        return ValidationResult.fail(
            "That name is too long. Please enter a realistic full name."
        )

    if not _NAME_RE.match(cleaned):
        return ValidationResult.fail(
            "Please provide your **full name** with at least a first and last name "
            "(e.g. **'Jane Doe'** or **'Arjun Sharma'**). "
            "Only letters, spaces, hyphens, and apostrophes are allowed."
        )

    # ── Gibberish check — each word must look like a real name ──────────────
    words = cleaned.split()
    for word in words:
        if len(word) >= 3 and _is_gibberish_name_word(word):
            return ValidationResult.fail(
                f"**'{cleaned}'** does not appear to be a real name. "
                "Please enter your actual full name as it appears on official documents."
            )

    return ValidationResult.ok(cleaned.title())


def validate_email(raw: str) -> ValidationResult:
    """
    Validates the candidate's email address.

    Rules:
        - Standard RFC-style format: user@domain.tld
        - Local part and domain must not look like gibberish.

    Returns:
        ValidationResult with the lowercased email on success.
    """
    cleaned = raw.strip().lower()

    if not cleaned:
        return ValidationResult.fail(
            "Email cannot be empty. Please enter your actual email address."
        )

    # ── Format check ─────────────────────────────────────────────────────────
    if not _EMAIL_RE.match(cleaned):
        return ValidationResult.fail(
            "That doesn't look like a valid email address. "
            "Please use the format **user@domain.com** (e.g. john.doe@gmail.com)."
        )

    # ── Gibberish check ───────────────────────────────────────────────────────
    is_junk, reason = check_email_gibberish(cleaned)
    if is_junk:
        return ValidationResult.fail(
            (reason or "That email address doesn't look real.") +
            "\n\nPlease enter your actual email address so we can contact you."
        )

    return ValidationResult.ok(cleaned)


def validate_phone(raw: str) -> ValidationResult:
    """
    Validates the candidate's phone number.

    Rules:
        - International formats accepted (country code optional).
        - Final digit count must be 7–15.
        - Rejects all-same-digit, sequential, and known placeholder numbers.

    Returns:
        ValidationResult with the stripped phone string on success.
    """
    cleaned = raw.strip()

    if not cleaned:
        return ValidationResult.fail(
            "Phone number cannot be empty. Please enter a valid contact number."
        )

    # ── Format check ─────────────────────────────────────────────────────────
    if not _PHONE_RE.match(cleaned):
        return ValidationResult.fail(
            "Invalid phone number format. Please include your country code if applicable.\n"
            "Examples: **+91-9876543210**, **+1-555-123-4567**, **07911 123456**"
        )

    digits_only = re.sub(r"\D", "", cleaned)
    if not (7 <= len(digits_only) <= 15):
        return ValidationResult.fail(
            f"Phone number must have between 7 and 15 digits. "
            f"You provided {len(digits_only)}. Please double-check and re-enter."
        )

    # ── Fake/placeholder number check ─────────────────────────────────────────
    is_fake, reason = check_phone_gibberish(cleaned)
    if is_fake:
        return ValidationResult.fail(
            (reason or "That doesn't look like a real phone number.") +
            "\n\nPlease enter your actual contact number."
        )

    return ValidationResult.ok(cleaned)


def validate_experience(raw: str) -> ValidationResult:
    """
    Validates years of professional experience.

    Accepts whole numbers between 0 and 60 inclusive.
    Also accepts common phrasing like "5 years" or "3+ years".

    Returns:
        ValidationResult with an integer value on success.
    """
    cleaned = raw.strip().lower()

    if not cleaned:
        return ValidationResult.fail(
            "Please enter your years of experience as a number (e.g. **3**, **7**, **12**)."
        )

    # Reject obvious gibberish before attempting numeric extraction
    if not re.search(r"\d", cleaned) and len(cleaned) > 3:
        is_junk, _ = gibberish_phrase(cleaned)
        if is_junk:
            return ValidationResult.fail(
                "Please enter your years of experience as a **number** "
                "(e.g. **0**, **3**, **7**, **12**, **15**)."
            )

    # Extract leading integer — accepts "5 years", "10+", "~8"
    match = re.search(r"(\d+)", cleaned)
    if not match:
        return ValidationResult.fail(
            "I couldn't read a number from your input. "
            "Please enter your years of experience as a number (e.g. **3**, **7**, **12**)."
        )

    years = int(match.group(1))

    if years > 60:
        return ValidationResult.fail(
            f"**{years} years** seems unusually high. "
            "Please double-check and re-enter your actual years of experience."
        )

    return ValidationResult.ok(years)


def validate_position(raw: str) -> ValidationResult:
    """
    Validates the desired job position.

    Rules:
        - Non-empty string between 2 and 100 characters.
        - Must resemble an actual job title (not gibberish).

    Returns:
        ValidationResult with the stripped position string on success.
    """
    cleaned = raw.strip()

    if len(cleaned) < 2:
        return ValidationResult.fail(
            "Please provide a valid job title "
            "(e.g. **Backend Engineer**, **Data Scientist**, **DevOps Engineer**)."
        )

    if len(cleaned) > 100:
        return ValidationResult.fail(
            "Job title is too long. Please keep it under 100 characters."
        )

    # ── Gibberish / nonsense check ────────────────────────────────────────────
    is_junk, reason = check_position_gibberish(cleaned)
    if is_junk:
        return ValidationResult.fail(
            (reason or f"**'{cleaned}'** does not look like a real job title.") +
            "\n\nExamples of valid titles: **Backend Engineer**, **ML Engineer**, "
            "**Full Stack Developer**, **Data Analyst**."
        )

    return ValidationResult.ok(cleaned)


def validate_location(raw: str) -> ValidationResult:
    """
    Validates the candidate's current location.

    Rules:
        - Non-empty string (city, country, or city+country).
        - Between 2 and 100 characters.
        - Must not be gibberish.

    Returns:
        ValidationResult with the title-cased location on success.
    """
    cleaned = raw.strip()

    if len(cleaned) < 2:
        return ValidationResult.fail(
            "Please provide your current location "
            "(e.g. **Bangalore, India** or **New York, USA**)."
        )

    if len(cleaned) > 100:
        return ValidationResult.fail(
            "Location entry is too long. Please keep it concise."
        )

    # ── Gibberish check ───────────────────────────────────────────────────────
    is_junk, reason = check_location_gibberish(cleaned)
    if is_junk:
        return ValidationResult.fail(
            (reason or f"**'{cleaned}'** does not look like a real location.") +
            "\n\nPlease enter your actual city and country "
            "(e.g. **Mumbai, India** or **London, UK**)."
        )

    return ValidationResult.ok(cleaned.title())


def validate_tech_stack(raw: str) -> ValidationResult:
    """
    Validates and parses the candidate's tech stack.

    Rules:
        - Comma-separated list of at least 1 technology.
        - Each entry must be 1–50 characters.
        - Maximum 20 technologies.
        - At least 40 % of entries must resemble known technologies.
          (Lenient: niche tools are allowed, but pure gibberish is not.)

    Returns:
        ValidationResult with a ``List[str]`` of cleaned technology names.
    """
    if not raw.strip():
        return ValidationResult.fail(
            "Please list at least one technology you know "
            "(e.g. **Python, FastAPI, PostgreSQL**)."
        )

    techs: List[str] = [t.strip() for t in raw.split(",") if t.strip()]

    if not techs:
        return ValidationResult.fail(
            "Could not parse any technologies. Please separate them with commas "
            "(e.g. **Python, Django, Docker**)."
        )

    if len(techs) > 20:
        return ValidationResult.fail(
            f"Please list up to 20 technologies. You provided {len(techs)}. "
            "Pick the ones most relevant to your experience."
        )

    for tech in techs:
        if len(tech) > 50:
            return ValidationResult.fail(
                f"**'{tech[:30]}…'** is too long for a single technology name. "
                "Please check your input."
            )

    # ── Gibberish / unknown technology check ─────────────────────────────────
    is_junk, reason = check_tech_stack(techs)
    if is_junk:
        return ValidationResult.fail(
            (reason or "Some entries do not look like real technologies.") +
            "\n\nPlease enter technologies from your actual skill set "
            "(e.g. **Python, React, PostgreSQL, Docker, AWS**)."
        )

    return ValidationResult.ok(techs)


# ---------------------------------------------------------------------------
# Internal helpers (not exported)
# ---------------------------------------------------------------------------

def _is_gibberish_name_word(word: str) -> bool:
    """
    Specialised check for a single name word.
    More lenient than generic gibberish detection to avoid blocking
    legitimate names from other languages/cultures.
    """
    from core.gibberish_detector import _vowel_ratio, _max_consonant_run, _bigram_score

    w = word.lower()

    # Only flag if multiple strong signals fire simultaneously
    vr = _vowel_ratio(w)
    mcr = _max_consonant_run(w)
    bscore = _bigram_score(w)

    # Hard rules for names
    if len(w) >= 5 and vr == 0.0:   # zero vowels in 5+ char word
        return True
    if mcr >= 7:                     # 7+ consecutive consonants
        return True

    # Soft rule: three independent weak signals together
    signals = 0
    if vr < 0.10:       signals += 2
    elif vr < 0.15:     signals += 1
    if mcr >= 5:        signals += 2
    elif mcr >= 4:      signals += 1
    if bscore < 0.08:   signals += 2
    elif bscore < 0.15: signals += 1

    return signals >= 3   # Balanced bar: catches obvious gibberish, tolerates unusual names