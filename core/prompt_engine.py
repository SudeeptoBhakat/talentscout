"""
TalentScout — Prompt Engine
No emojis. All string construction lives here.
"""

from __future__ import annotations
import random
from typing import List, Optional
from uuid import uuid4

BOT_NAME = "TalentScout"
COMPANY = "TalentScout Hiring Platform"

POPULAR_TECH = [
    "Python", "Java", "JavaScript", "React", "Django",
    "Node.js", "AWS", "SQL", "MongoDB", "TensorFlow",
]

COMMON_ROLES = [
    "Backend Developer", "Frontend Developer", "Data Scientist",
    "DevOps Engineer", "ML Engineer",
]


# ---------------------------------------------------------------------------
# Scripted replies
# ---------------------------------------------------------------------------

def reply_greeting() -> str:
    return (
        "Hello, welcome to **TalentScout Hiring Assistant**.\n\n"
        "I will collect some basic details and ask a few short technical questions "
        "based on your tech stack.\n\n"
        "You can type **exit** anytime to stop the process.\n\n"
        "Let's get started!\n\n"
        "**What is your full name?**"
    )


def reply_ask_email(name: str) -> str:
    first = name.split()[0] if name else "there"
    return f"Nice to meet you, **{first}**.\n\nWhat is your **email address**?"


def reply_ask_phone() -> str:
    return (
        "What is your **phone number**?\n\n"
        "_Include country code if applicable, e.g. +91-9876543210_"
    )


def reply_ask_experience() -> str:
    return "How many **years of professional experience** do you have? _(e.g. 2, 5, 10)_"


def reply_ask_position() -> str:
    roles = "\n".join(f"  - {r}" for r in COMMON_ROLES)
    return (
        "What **position** are you applying for?\n\n"
        "Common roles:\n"
        f"{roles}\n\n"
        "_Type your desired role, or any other title._"
    )


def reply_ask_location() -> str:
    return "Where are you **currently located**? _(City, Country — e.g. Mumbai, India)_"


def reply_ask_tech_stack() -> str:
    suggestions = ", ".join(POPULAR_TECH)
    return (
        "Please list your **tech stack** — languages, frameworks, and tools you know.\n\n"
        "Separate with commas, e.g.:\n"
        "> `Python, Django, PostgreSQL, Docker`\n\n"
        "**Popular technologies:** " + suggestions + "\n\n"
        "_(List up to **8** technologies)_"
    )


def reply_validation_error(error_message: str) -> str:
    return (
        f"**I could not accept that input.**\n\n"
        f"{error_message}\n\n"
        "_Please try again, or type **exit** to stop._"
    )


def reply_gibberish_warning(field: str, hint: str) -> str:
    """
    Friendly but firm message shown when random/nonsense input is detected.

    Args:
        field: The field name (e.g. 'name', 'email', 'location').
        hint:  A concrete example of valid input.
    """
    return (
        f"That doesn't look like a real **{field}**.\n\n"
        f"This is a professional screening process and I need accurate information "
        f"to proceed.\n\n"
        f"**Example of a valid {field}:** {hint}\n\n"
        "_Please enter your actual information, or type **exit** to stop._"
    )


def reply_questions_intro(total: int) -> str:
    return (
        f"I will now ask you **{total} short technical questions** "
        f"based on your tech stack.\n\n"
        "Please answer **briefly and clearly**."
    )


def reply_question(n: int, total: int, question: str) -> str:
    return f"**Question {n}/{total}:** {question}"


def reply_answer_ack(n: int, total: int) -> str:
    remaining = total - n
    if remaining > 0:
        return f"Answer noted. Moving to question **{n + 1}/{total}**..."
    return "All questions answered! Calculating your results..."


def reply_wrap_up(name: str) -> str:
    first = name.split()[0] if name else "there"
    return (
        f"Thank you, **{first}**, for completing the screening.\n\n"
        "We will review your responses and contact you shortly.\n\n"
        "**Your result summary is shown below.**"
    )


def reply_exit(name: str | None = None) -> str:
    greeting = f"Goodbye, **{name.split()[0]}**." if name else "Goodbye."
    return (
        f"{greeting}\n\n"
        "Your session has been ended. Feel free to start a new session anytime.\n\n"
        f"Best wishes from the **{COMPANY}** team."
    )


def reply_off_topic() -> str:
    return (
        "I am a hiring assistant and can only help with interview-related questions.\n\n"
        "Please answer the question above, or type **exit** to leave."
    )


# ---------------------------------------------------------------------------
# LLM Prompt Builders
# ---------------------------------------------------------------------------

# Question type templates for variation
_QUESTION_TYPES = [
    "How would you explain {concept} to a junior developer?",
    "What is the difference between {concept} and its common alternative?",
    "Describe a real-world situation where {concept} caused you trouble.",
    "What is the most common mistake developers make with {concept}?",
    "When would you choose NOT to use {concept}?",
    "How does {concept} behave under high load or edge cases?",
    "What best practices do you follow when working with {concept}?",
]

_SENIORITY_ANGLES = {
    "junior":  "conceptual understanding, basic syntax, and common use cases",
    "mid":     "design decisions, trade-offs, and debugging approaches",
    "senior":  "architecture, performance, scalability, and real-world pitfalls",
    "staff":   "system design, team impact, long-term maintainability, and mentoring",
}


def build_tech_questions_prompt(
    tech_stack: List[str],
    desired_position: str,
    years_experience: int,
    num_questions: int = 5,
    previously_asked: Optional[List[str]] = None,
) -> str:
    """
    Builds a dynamic, non-repetitive question generation prompt.

    Args:
        tech_stack:        List of technologies the candidate knows.
        desired_position:  The role they are applying for.
        years_experience:  Years of professional experience.
        num_questions:     Number of questions to generate.
        previously_asked:  Questions asked in earlier sessions — will be
                           explicitly excluded to ensure uniqueness across
                           multiple candidates with the same tech stack.
    """
    stack_display = ", ".join(tech_stack)
    seniority_key, seniority_label = _infer_seniority_full(years_experience)
    focus_angle = _SENIORITY_ANGLES.get(seniority_key, _SENIORITY_ANGLES["mid"])

    # Unique seed per call
    variation_seed = uuid4().hex[:10]

    # Random questioning style and angle for this session
    styles = [
        "practical real-world scenarios",
        "conceptual depth and underlying mechanisms",
        "common mistakes and how to avoid them",
        "performance and scalability considerations",
        "debugging and problem-solving approaches",
        "trade-off analysis between alternatives",
        "best practices and code quality",
        "edge cases and failure modes",
    ]
    selected_styles = random.sample(styles, k=min(3, len(styles)))
    style_instruction = ", ".join(selected_styles)

    # Distribution note
    n = len(tech_stack)
    if n == 1:
        dist_note = f"All {num_questions} questions must be about {tech_stack[0]}. Use variety in angle and difficulty."
    elif n == 2:
        dist_note = f"Split questions roughly evenly: ~{num_questions // 2} for each technology."
    else:
        dist_note = (
            f"Distribute questions across the stack. Each major technology should appear at least once. "
            f"Avoid clustering all questions on a single technology."
        )

    # Build the "avoid" block — this is the key anti-repetition mechanism
    avoid_block = ""
    if previously_asked:
        avoid_lines = "\n".join(f"  - {q}" for q in previously_asked[:30])
        avoid_block = (
            f"\nPREVIOUSLY ASKED QUESTIONS — DO NOT repeat or closely paraphrase any of these:\n"
            f"{avoid_lines}\n\n"
            f"Generate ENTIRELY DIFFERENT questions. If you must cover the same technology, "
            f"use a completely different angle, framing, or concept.\n"
        )

    return (
        f"You are a senior technical interviewer conducting a screening interview.\n\n"
        f"SESSION SEED: {variation_seed}\n"
        f"Questioning Focus: {style_instruction}\n\n"
        f"Generate exactly {num_questions} SHORT-ANSWER interview questions.\n\n"
        f"Candidate Profile:\n"
        f"  Role:       {desired_position}\n"
        f"  Seniority:  {seniority_label}\n"
        f"  Tech Stack: {stack_display}\n"
        f"  Focus on:   {focus_angle}\n"
        f"{avoid_block}"
        f"Rules:\n"
        f"  1. {dist_note}\n"
        f"  2. Each question should require a 2-4 sentence answer (not yes/no).\n"
        f"  3. NEVER use questions that begin with 'What is the difference between' "
        f"     more than once across all questions.\n"
        f"  4. Vary the sentence structure: some should start with 'How', 'When', "
        f"     'Why', 'Describe', 'Explain', 'What would you do if'.\n"
        f"  5. Calibrate difficulty precisely to {seniority_label}.\n"
        f"  6. Do NOT include model answers.\n"
        f"  7. Return ONLY the numbered list (1 to {num_questions}). No intro, no outro.\n"
        f"  8. Make every question feel fresh and non-textbook.\n\n"
        f"Generate questions now:"
    )


def build_evaluation_prompt(question: str, answer: str) -> str:
    """
    Builds the LLM prompt for evaluating a candidate answer.
    Designed to produce reliable, parseable JSON every time.
    """
    stripped_answer = answer.strip() if answer.strip() else "[No answer provided]"
    return (
        "You are a strict but fair technical interviewer evaluating a screening response.\n\n"
        "Your task: score the candidate's answer and give a one-sentence feedback.\n\n"
        f"Question: {question}\n"
        f"Candidate Answer: {stripped_answer}\n\n"
        "Scoring guide:\n"
        "  0 — No answer or completely wrong.\n"
        "  1 — Vague, mostly incorrect, or irrelevant.\n"
        "  2 — Partial understanding, significant gaps.\n"
        "  3 — Adequate answer, covers the basics.\n"
        "  4 — Good answer, demonstrates solid understanding.\n"
        "  5 — Excellent, precise, and complete.\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "  - Respond ONLY with a single valid JSON object. Nothing else.\n"
        "  - Do NOT use markdown code fences or backticks.\n"
        "  - Do NOT add any text before or after the JSON.\n"
        "  - The JSON must have exactly two keys: score (integer 0-5) and feedback (string).\n\n"
        "Example of the ONLY acceptable output format:\n"
        '{"score": 3, "feedback": "Candidate understands the concept but missed key details."}\n\n'
        "Now evaluate and respond with JSON only:"
    )


# ---------------------------------------------------------------------------
# Internal Helpers
# ---------------------------------------------------------------------------

def _infer_seniority_full(years: int):
    """Returns (key, label) tuple for seniority level."""
    if years <= 1:
        return ("junior", "junior (0-1 year)")
    if years <= 3:
        return ("mid", "mid-level (2-3 years)")
    if years <= 6:
        return ("senior", "senior (4-6 years)")
    return ("staff", "staff/principal (7+ years)")


def _infer_seniority(years: int) -> str:
    """Backwards-compatible label-only version."""
    return _infer_seniority_full(years)[1]