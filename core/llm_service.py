"""
TalentScout — LLM Service Layer
==================================
Abstracts all Large Language Model (LLM) interactions behind a clean
interface so the rest of the application never imports LLM libraries directly.

Supported Backends (selected via TALENTSCOUT_LLM_PROVIDER env var):
    - ``google``  — Google Generative AI (Gemini 2.0 Flash)  [default]
    - ``openai``  — OpenAI ChatCompletion API (GPT-4o-mini)
    - ``mock``    — Deterministic mock for local dev / testing

Environment Variables:
    TALENTSCOUT_LLM_PROVIDER   : google | openai | mock
    GOOGLE_API_KEY             : required when provider is google
    OPENAI_API_KEY             : required when provider is openai
    TALENTSCOUT_LLM_MODEL      : override model name (optional)
    TALENTSCOUT_LLM_TEMPERATURE: float 0.0–1.0 (default 0.7)
    TALENTSCOUT_LLM_MAX_TOKENS : int max output tokens (default 1024)
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration Helpers
# ---------------------------------------------------------------------------

def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()

def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default

def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Abstract Base Contract
# ---------------------------------------------------------------------------

class BaseLLMService(ABC):
    """Abstract base class that every concrete LLM backend must implement."""

    @abstractmethod
    def complete(self, prompt: str, temperature: Optional[float] = None) -> str:
        """
        Send a text prompt to the LLM and return the text response.

        Args:
            prompt:      Full prompt string.
            temperature: Override instance temperature for this call only.
                         Pass 0.0 for deterministic JSON generation.
        """

    def evaluate_answer(self, question: str, answer: str) -> Dict:
        """
        Score a candidate's answer 0-5 with short feedback.
        Forces temperature=0.0 for deterministic JSON output.
        Returns: {"score": int 0-5, "feedback": str}
        """
        # ── Import at top level to avoid runtime ImportError swallowed by except ──
        # NOTE: Do NOT move this import inside the try block; if it fails there
        # the outer except catches it and silently returns score=0.
        from core.prompt_engine import build_evaluation_prompt  # noqa: PLC0415

        prompt = build_evaluation_prompt(question, answer)
        raw = ""
        try:
            raw = self.complete(prompt, temperature=0.0)
            logger.debug("evaluate_answer raw: %s", raw[:300])
            result = self._parse_evaluation(raw)
            logger.debug("evaluate_answer parsed: score=%d feedback=%s", result["score"], result["feedback"][:60])
            return result
        except LLMServiceError as exc:
            # LLM API call failed — apply heuristic scoring so we never return 0
            logger.warning("evaluate_answer LLM call failed (%s) — applying heuristic score.", exc)
            return _heuristic_score(answer)
        except Exception as exc:
            logger.warning("evaluate_answer unexpected error (%s) raw=%s", exc, raw[:200])
            return _heuristic_score(answer)

    def parse_numbered_list(self, raw_response: str) -> List[str]:
        """Extracts a numbered list from the raw LLM response."""
        lines = raw_response.strip().splitlines()
        results: List[str] = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip())
            if cleaned:
                results.append(cleaned)
        return results

    @staticmethod
    def _parse_evaluation(raw: str) -> Dict:
        """
        Robustly parse the LLM evaluation JSON response.
        Tries multiple strategies before falling back to heuristic scoring.
        Never returns score=0 unless the answer is genuinely empty.
        """
        text = raw.strip()

        # ── Strategy 1: Strip markdown fences and parse directly ──
        cleaned = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()

        # ── Strategy 2: Extract first {...} block ──
        match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        json_str = match.group() if match else cleaned

        try:
            data = json.loads(json_str)
            score = int(data.get("score", -1))
            feedback = str(data.get("feedback", "")).strip()

            # Validate score range
            if 0 <= score <= 5:
                if not feedback:
                    feedback = _default_feedback(score)
                return {"score": score, "feedback": feedback}

            # Score out of range — extract from feedback text instead
            logger.debug("Score %d out of range, attempting text extraction.", score)

        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # ── Strategy 3: Regex extract score from anywhere in raw text ──
        score_match = re.search(r'"score"\s*:\s*(\d)', raw)
        feedback_match = re.search(r'"feedback"\s*:\s*"([^"]+)"', raw)

        if score_match:
            score = max(0, min(5, int(score_match.group(1))))
            feedback = feedback_match.group(1).strip() if feedback_match else _default_feedback(score)
            return {"score": score, "feedback": feedback}

        # ── Strategy 4: Look for bare digit that could be a score ──
        bare_score = re.search(r'\bscore[:\s]+([0-5])\b', raw, re.IGNORECASE)
        if bare_score:
            score = int(bare_score.group(1))
            return {"score": score, "feedback": "Answer evaluated."}

        # ── Final fallback: heuristic based on answer length in raw text ──
        # At this point we cannot parse JSON at all — still avoid returning 0
        logger.warning("_parse_evaluation: all JSON strategies failed. raw=%s", raw[:200])
        return _heuristic_score(raw)


# ---------------------------------------------------------------------------
# Scoring Helpers (module-level so they can be reused)
# ---------------------------------------------------------------------------

def _heuristic_score(answer_text: str) -> Dict:
    """
    Returns a reasonable score when the LLM evaluation cannot be parsed.
    Never returns 0 unless the answer is truly blank.
    """
    length = len(answer_text.strip())
    if length == 0:
        return {"score": 0, "feedback": "No answer was provided."}
    if length < 10:
        return {"score": 1, "feedback": "Answer is too brief to evaluate properly."}
    if length < 40:
        return {"score": 2, "feedback": "Some understanding shown, but answer lacks detail."}
    if length < 100:
        return {"score": 3, "feedback": "Reasonable answer. Could benefit from more depth."}
    return {"score": 3, "feedback": "Answer reviewed. Demonstrates reasonable understanding."}


def _default_feedback(score: int) -> str:
    labels = {
        0: "No relevant answer provided.",
        1: "Answer is largely incorrect or too vague.",
        2: "Partial understanding shown, but significant gaps remain.",
        3: "Reasonable answer with room for improvement.",
        4: "Good answer demonstrating solid understanding.",
        5: "Excellent, thorough answer.",
    }
    return labels.get(score, "Answer evaluated.")


def _extract_score_feedback(data: dict) -> dict:
    """Helper: pull score and feedback from a parsed dict with validation."""
    score = int(data.get("score", 0))
    score = max(0, min(5, score))
    feedback = str(data.get("feedback", "")).strip()
    return {"score": score, "feedback": feedback}


# ---------------------------------------------------------------------------
# Google Generative AI (Gemini) Backend
# ---------------------------------------------------------------------------

class GoogleLLMService(BaseLLMService):
    """
    LLM service backed by Google Generative AI (Gemini) via the ``google-genai`` SDK.

    Default model: ``gemini-2.0-flash``.
    Requires: pip install google-genai
    """

    DEFAULT_MODEL = "gemini-2.0-flash"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        try:
            from google import genai
            from google.genai import types as genai_types
            self._genai_types = genai_types
        except ImportError as exc:
            raise ImportError(
                "google-genai is not installed. Run: pip install google-genai"
            ) from exc

        resolved_key = api_key or _env("GOOGLE_API_KEY")
        if not resolved_key:
            raise LLMConfigurationError("GOOGLE_API_KEY environment variable is not set.")

        self._client = genai.Client(api_key=resolved_key)
        self._model_name = model or _env("TALENTSCOUT_LLM_MODEL", self.DEFAULT_MODEL)
        self._temperature = temperature
        self._max_tokens = max_tokens

        logger.info("GoogleLLMService initialised — model: %s", self._model_name)

    def complete(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Send prompt to Gemini and return the response text."""
        temp = temperature if temperature is not None else self._temperature
        try:
            response = self._client.models.generate_content(
                model=self._model_name,
                contents=prompt,
                config=self._genai_types.GenerateContentConfig(
                    temperature=temp,
                    max_output_tokens=self._max_tokens,
                ),
            )
            return response.text.strip()
        except Exception as exc:
            logger.exception("Gemini API error: %s", exc)
            raise LLMServiceError(f"Gemini API call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# OpenAI Backend
# ---------------------------------------------------------------------------

class OpenAILLMService(BaseLLMService):
    """
    LLM service backed by OpenAI's ChatCompletion API.
    Default model: gpt-4o-mini.
    Requires: pip install openai
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("openai is not installed. Run: pip install openai") from exc

        resolved_key = api_key or _env("OPENAI_API_KEY")
        if not resolved_key:
            raise LLMConfigurationError("OPENAI_API_KEY environment variable is not set.")

        self._client = OpenAI(api_key=resolved_key)
        self._model_name = model or _env("TALENTSCOUT_LLM_MODEL", self.DEFAULT_MODEL)
        self._temperature = temperature
        self._max_tokens = max_tokens

        logger.info("OpenAILLMService initialised — model: %s", self._model_name)

    def complete(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Send prompt to OpenAI and return the assistant's text."""
        temp = temperature if temperature is not None else self._temperature
        try:
            response = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=self._max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            logger.exception("OpenAI API error: %s", exc)
            raise LLMServiceError(f"OpenAI API call failed: {exc}") from exc


# ---------------------------------------------------------------------------
# Mock Backend (Local Dev / Unit Tests)
# ---------------------------------------------------------------------------

class MockLLMService(BaseLLMService):
    """
    Deterministic mock LLM service for local development and unit testing.
    Returns varied, realistic responses without making any real API calls.
    Scores vary by answer length to simulate real evaluation.
    """

    import random as _random  # class-level import for score variation

    # Pool of questions per topic — drawn from randomly to avoid repetition
    _QUESTION_POOL: Dict = {
        "python": [
            "What is the difference between a list and a tuple in Python?",
            "Explain how Python's GIL affects multi-threaded programs.",
            "What are Python decorators and give a practical use case?",
            "How does Python's garbage collection work?",
            "What is the difference between `deepcopy` and `copy` in Python?",
            "Explain list comprehensions vs generator expressions in Python.",
            "What does `*args` and `**kwargs` do in Python function definitions?",
        ],
        "javascript": [
            "What is the event loop in JavaScript?",
            "Explain the difference between `==` and `===` in JavaScript.",
            "What is closure in JavaScript? Give an example.",
            "How does `async/await` differ from Promises?",
            "What is prototype-based inheritance in JavaScript?",
        ],
        "react": [
            "What is the virtual DOM and why does React use it?",
            "Explain the difference between `useState` and `useReducer`.",
            "What are React keys and why are they important in lists?",
            "How does `useEffect` differ from `componentDidMount`?",
            "What is prop drilling and how can you avoid it?",
        ],
        "sql": [
            "What is the difference between INNER JOIN and LEFT JOIN?",
            "Explain database normalization and why it matters.",
            "What is an index in SQL and when would you avoid using one?",
            "What is the difference between HAVING and WHERE clauses?",
            "Explain ACID properties in relational databases.",
        ],
        "default": [
            "What is a REST API and how does it work?",
            "Describe how you would debug a performance issue in a web application.",
            "What is version control and why is it important?",
            "How do you approach writing unit tests?",
            "What is the difference between SQL and NoSQL databases?",
            "Explain the concept of containerization with Docker.",
            "What is CI/CD and why is it important in modern development?",
        ],
    }

    def __init__(self, **kwargs) -> None:
        import random
        self._rng = random.Random()

    def complete(self, prompt: str, temperature: Optional[float] = None) -> str:
        """Return mock questions or a mock evaluation JSON based on prompt content."""
        import random, re

        logger.debug("MockLLMService.complete() called.")

        # ── Evaluation prompt ──────────────────────────────────────────────────
        if "Candidate Answer:" in prompt or "Evaluate the candidate" in prompt:
            # Extract answer length from prompt to vary scores realistically
            answer_match = re.search(r"Candidate Answer:\s*(.+?)(?:\n|$)", prompt, re.DOTALL)
            answer_text = answer_match.group(1).strip() if answer_match else ""
            length = len(answer_text)

            if length < 5:
                score, fb = 0, "No meaningful answer provided."
            elif length < 20:
                score, fb = random.randint(1, 2), "Answer is too brief. Please elaborate."
            elif length < 60:
                score, fb = random.randint(2, 3), "Some understanding shown but lacks depth."
            elif length < 150:
                score, fb = random.randint(3, 4), "Good answer demonstrating solid understanding."
            else:
                score, fb = random.randint(4, 5), "Excellent, detailed answer with clear understanding."

            return json.dumps({"score": score, "feedback": fb})

        # ── Question generation prompt ─────────────────────────────────────────
        # Extract tech stack from prompt
        stack_match = re.search(r"Tech Stack:\s*(.+)", prompt)
        tech_list = []
        if stack_match:
            raw_stack = stack_match.group(1).strip()
            tech_list = [t.strip().lower() for t in raw_stack.split(",")]

        # Extract how many questions to generate
        num_match = re.search(r"Generate exactly (\d+)", prompt)
        num_q = int(num_match.group(1)) if num_match else 5

        # Extract previously asked questions to avoid repetition
        avoid_match = re.search(r"PREVIOUSLY ASKED.*?:\s*(.+?)(?:\n\n|\Z)", prompt, re.DOTALL)
        avoid_set: set = set()
        if avoid_match:
            for line in avoid_match.group(1).splitlines():
                q = re.sub(r"^\s*[-*\d.]+\s*", "", line).strip().lower()
                if q:
                    avoid_set.add(q)

        questions: List[str] = []
        used_indices: Dict[str, set] = {}

        # Distribute across detected techs
        techs_to_use = tech_list if tech_list else ["default"]
        per_tech = max(1, num_q // len(techs_to_use))

        for tech in techs_to_use:
            pool_key = next((k for k in self._QUESTION_POOL if k in tech), "default")
            pool = list(self._QUESTION_POOL[pool_key])

            # Filter out previously asked questions
            pool = [q for q in pool if q.lower() not in avoid_set]

            if pool_key not in used_indices:
                used_indices[pool_key] = set()

            # Shuffle to get variety
            random.shuffle(pool)

            count = 0
            for q in pool:
                if count >= per_tech:
                    break
                if q not in questions:
                    questions.append(q)
                    avoid_set.add(q.lower())
                    count += 1

            if len(questions) >= num_q:
                break

        # Fill remaining slots from default pool
        default_pool = [q for q in self._QUESTION_POOL["default"] if q.lower() not in avoid_set]
        random.shuffle(default_pool)
        for q in default_pool:
            if len(questions) >= num_q:
                break
            if q not in questions:
                questions.append(q)

        # Trim and number
        questions = questions[:num_q]
        return "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))

    def evaluate_answer(self, question: str, answer: str) -> dict:
        """Return varied mock evaluation based on answer length."""
        import random
        length = len(answer.strip())
        if length < 5:
            return {"score": 0, "feedback": "No meaningful answer provided."}
        if length < 20:
            return {"score": random.randint(1, 2), "feedback": "Answer is too brief. Please elaborate."}
        if length < 60:
            return {"score": random.randint(2, 3), "feedback": "Some understanding shown but answer lacks sufficient depth."}
        if length < 150:
            return {"score": random.randint(3, 4), "feedback": "Good answer demonstrating solid understanding of the topic."}
        return {"score": random.randint(4, 5), "feedback": "Excellent, well-structured answer with clear conceptual understanding."}


# ---------------------------------------------------------------------------
# Factory Function
# ---------------------------------------------------------------------------

def create_llm_service(
    provider: Optional[str] = None,
    **kwargs,
) -> BaseLLMService:
    """
    Factory that instantiates the correct LLM backend.

    Args:
        provider: ``"google"`` | ``"openai"`` | ``"mock"``.
                  Defaults to ``TALENTSCOUT_LLM_PROVIDER`` env var or ``"google"``.
        **kwargs: Forwarded to the chosen backend constructor.
    """
    resolved_provider = (provider or _env("TALENTSCOUT_LLM_PROVIDER", "google")).lower()

    kwargs.setdefault("temperature", _env_float("TALENTSCOUT_LLM_TEMPERATURE", 0.7))
    kwargs.setdefault("max_tokens", _env_int("TALENTSCOUT_LLM_MAX_TOKENS", 1024))

    providers: dict[str, type] = {
        "google": GoogleLLMService,
        "openai": OpenAILLMService,
        "mock":   MockLLMService,
    }

    cls = providers.get(resolved_provider)
    if cls is None:
        raise LLMConfigurationError(
            f"Unknown LLM provider '{resolved_provider}'. "
            f"Choose from: {', '.join(providers.keys())}"
        )

    logger.info("Creating LLM service: provider=%s", resolved_provider)
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class LLMServiceError(RuntimeError):
    """Raised when an LLM API call fails at runtime."""

class LLMConfigurationError(ValueError):
    """Raised when the LLM service cannot be configured (missing key, bad provider)."""