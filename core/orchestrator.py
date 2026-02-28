"""
TalentScout — Conversation Orchestrator
=========================================
The central controller that drives the hiring-assistant conversation.

Key upgrade: Technical question phase now asks questions ONE AT A TIME,
evaluates each answer with the LLM (score 0–5 + feedback), and stores
structured results in the CandidateSession.

Anti-repetition: A global registry of previously-asked questions is
maintained so that if two candidates with the same tech stack go through
the chatbot, they receive different questions each time.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Set

from core.state_manager import ConversationState, StateManager
from core.prompt_engine import (
    reply_greeting,
    reply_ask_email,
    reply_ask_phone,
    reply_ask_experience,
    reply_ask_position,
    reply_ask_location,
    reply_ask_tech_stack,
    reply_validation_error,
    reply_gibberish_warning,
    reply_questions_intro,
    reply_question,
    reply_answer_ack,
    reply_wrap_up,
    reply_exit,
    build_tech_questions_prompt,
)
from core import validators as val
from core.llm_service import BaseLLMService, create_llm_service, LLMServiceError
from models.candidate import Candidate, CandidateSession, QuestionResult
from data.storage import SessionStorage

logger = logging.getLogger(__name__)

# Number of questions to generate: clamp to [4, 6] per the spec
_MIN_QUESTIONS = 4
_MAX_QUESTIONS = 6

# ---------------------------------------------------------------------------
# Global asked-questions registry
# ---------------------------------------------------------------------------
# This set lives for the lifetime of the process. Every question that has
# been asked to ANY candidate is stored here so the prompt engine can
# explicitly exclude it, guaranteeing variation across sessions.
# In production you would persist this in Redis / a DB column.

_GLOBAL_ASKED_QUESTIONS: Set[str] = set()


class TalentScoutOrchestrator:
    """
    Drives the entire TalentScout conversation for a single candidate session.

    Each call to ``handle_message`` processes one user turn and returns the
    bot's next response. One instance per candidate session.

    State machine flow:
        GREETING → COLLECT_NAME → COLLECT_EMAIL → COLLECT_PHONE
        → COLLECT_EXPERIENCE → COLLECT_POSITION → COLLECT_LOCATION
        → COLLECT_TECH_STACK → TECH_QUESTIONS (loop N times) → WRAP_UP → ENDED

    In TECH_QUESTIONS state:
        - Questions are asked one at a time.
        - Each answer is evaluated by the LLM (score 0–5 + feedback).
        - Results are stored in session.results list.
        - After all questions answered → WRAP_UP.
    """

    def __init__(
        self,
        llm_service: Optional[BaseLLMService] = None,
        storage: Optional[SessionStorage] = None,
        num_questions: int = 5,
    ) -> None:
        self._llm: BaseLLMService = llm_service or create_llm_service()
        self._storage: SessionStorage = storage or SessionStorage()
        self._num_questions: int = max(_MIN_QUESTIONS, min(_MAX_QUESTIONS, num_questions))

        self._sm = StateManager()
        self._session = CandidateSession()

        logger.info("Orchestrator initialised — session_id=%s", self._session.session_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_ended(self) -> bool:
        return self._sm.is_ended

    @property
    def session(self) -> CandidateSession:
        return self._session

    def get_greeting(self) -> str:
        text = reply_greeting()
        self._sm.transition(ConversationState.COLLECT_NAME)
        return text

    def handle_message(self, user_input: str) -> str:
        if self._sm.is_ended:
            return reply_exit(self._session.candidate.full_name)

        text = user_input.strip()

        if val.is_exit_intent(text):
            self._end_session(graceful=False)
            return reply_exit(self._session.candidate.full_name)

        state = self._sm.current_state
        handler = self._STATE_HANDLERS.get(state)

        if handler is None:
            logger.error("No handler for state %s", state)
            return "An internal error occurred. Please refresh and restart."

        return handler(self, text)

    # ------------------------------------------------------------------
    # Collection State Handlers
    # ------------------------------------------------------------------

    def _handle_collect_name(self, text: str) -> str:
        result = val.validate_name(text)
        if not result.is_valid:
            return reply_validation_error(result.error)
        self._session.candidate.full_name = result.value
        self._sm.transition(ConversationState.COLLECT_EMAIL)
        return reply_ask_email(result.value)

    def _handle_collect_email(self, text: str) -> str:
        result = val.validate_email(text)
        if not result.is_valid:
            return reply_validation_error(result.error)
        self._session.candidate.email = result.value
        self._sm.transition(ConversationState.COLLECT_PHONE)
        return reply_ask_phone()

    def _handle_collect_phone(self, text: str) -> str:
        result = val.validate_phone(text)
        if not result.is_valid:
            return reply_validation_error(result.error)
        self._session.candidate.phone = result.value
        self._sm.transition(ConversationState.COLLECT_EXPERIENCE)
        return reply_ask_experience()

    def _handle_collect_experience(self, text: str) -> str:
        result = val.validate_experience(text)
        if not result.is_valid:
            return reply_validation_error(result.error)
        self._session.candidate.years_experience = result.value
        self._sm.transition(ConversationState.COLLECT_POSITION)
        return reply_ask_position()

    def _handle_collect_position(self, text: str) -> str:
        result = val.validate_position(text)
        if not result.is_valid:
            return reply_validation_error(result.error)
        self._session.candidate.desired_position = result.value
        self._sm.transition(ConversationState.COLLECT_LOCATION)
        return reply_ask_location()

    def _handle_collect_location(self, text: str) -> str:
        result = val.validate_location(text)
        if not result.is_valid:
            return reply_validation_error(result.error)
        self._session.candidate.location = result.value
        self._sm.transition(ConversationState.COLLECT_TECH_STACK)
        return reply_ask_tech_stack()

    def _handle_collect_tech_stack(self, text: str) -> str:
        result = val.validate_tech_stack(text)
        if not result.is_valid:
            return reply_validation_error(result.error)

        techs = result.value[:8]
        self._session.candidate.tech_stack = techs

        num_q = self._compute_question_count(len(techs))
        questions = self._generate_tech_questions(num_q)
        self._session.tech_questions = questions
        self._session.current_q_index = 0

        # Register these questions in the global registry immediately
        # so that even if this session is abandoned, the questions
        # are marked as used for future sessions.
        _GLOBAL_ASKED_QUESTIONS.update(questions)

        self._sm.transition(ConversationState.TECH_QUESTIONS)
        logger.debug("Tech stack collected, %d questions generated → TECH_QUESTIONS", len(questions))

        intro = reply_questions_intro(len(questions))
        q1 = reply_question(1, len(questions), questions[0])
        return intro + "\n\n" + q1

    # ------------------------------------------------------------------
    # Technical Question Phase Handler (one question at a time)
    # ------------------------------------------------------------------

    def _handle_tech_questions(self, text: str) -> str:
        questions = self._session.tech_questions
        idx = self._session.current_q_index

        if idx >= len(questions):
            return self._handle_wrap_up(text)

        current_q = questions[idx]

        # Evaluate the answer
        try:
            eval_result = self._llm.evaluate_answer(current_q, text)
        except Exception as exc:
            logger.warning("evaluate_answer raised unexpectedly: %s", exc)
            # Fall back to heuristic so we never crash the conversation
            from core.llm_service import _heuristic_score
            eval_result = _heuristic_score(text)

        qr = QuestionResult(
            question=current_q,
            answer=text,
            score=eval_result.get("score", 1),
            feedback=eval_result.get("feedback", "Answer recorded."),
        )
        self._session.results.append(qr)
        self._session.current_q_index = idx + 1

        logger.debug(
            "Q%d answered | score=%d | feedback=%s",
            idx + 1, qr.score, qr.feedback[:60],
        )

        next_idx = self._session.current_q_index
        total = len(questions)

        if next_idx < total:
            next_q = reply_question(next_idx + 1, total, questions[next_idx])
            return f"Got it!\n\n{next_q}"

        self._sm.transition(ConversationState.WRAP_UP)
        return self._handle_wrap_up(text)

    def _handle_wrap_up(self, _text: str) -> str:
        name = self._session.candidate.full_name or ""
        self._end_session(graceful=True)
        return reply_wrap_up(name)

    # ------------------------------------------------------------------
    # Dispatch Table
    # ------------------------------------------------------------------

    _STATE_HANDLERS: dict = {}

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _compute_question_count(self, num_techs: int) -> int:
        if num_techs <= 2:
            desired = 4
        elif num_techs <= 4:
            desired = 5
        else:
            desired = 6
        return max(_MIN_QUESTIONS, min(_MAX_QUESTIONS, desired))

    def _generate_tech_questions(self, num_q: int) -> List[str]:
        """
        Calls the LLM to generate technical questions, passing the global
        registry of previously-asked questions so the model avoids repeating them.
        Falls back to varied hardcoded questions on LLM failure.
        """
        candidate = self._session.candidate

        # Pass the full global registry of previously asked questions
        previously_asked = list(_GLOBAL_ASKED_QUESTIONS)

        prompt = build_tech_questions_prompt(
            tech_stack=candidate.tech_stack,
            desired_position=candidate.desired_position or "Software Engineer",
            years_experience=candidate.years_experience or 0,
            num_questions=num_q,
            previously_asked=previously_asked,
        )
        try:
            raw = self._llm.complete(prompt)
            questions = self._llm.parse_numbered_list(raw)

            # Remove any question that is still too similar to a previously asked one
            questions = self._deduplicate(questions, _GLOBAL_ASKED_QUESTIONS, threshold=0.75)

            # Pad or trim to exact count
            if len(questions) < num_q:
                fallback = self._fallback_questions(candidate.tech_stack, _GLOBAL_ASKED_QUESTIONS)
                questions += fallback[:num_q - len(questions)]

            return questions[:num_q]
        except LLMServiceError as exc:
            logger.warning("Question generation failed: %s — using fallback.", exc)
            return self._fallback_questions(candidate.tech_stack, _GLOBAL_ASKED_QUESTIONS)[:num_q]

    @staticmethod
    def _deduplicate(
        questions: List[str],
        seen: Set[str],
        threshold: float = 0.75,
    ) -> List[str]:
        """
        Remove questions that are too similar to those already seen.
        Uses simple word-overlap ratio as a lightweight similarity check.
        """
        def _overlap(a: str, b: str) -> float:
            wa = set(a.lower().split())
            wb = set(b.lower().split())
            if not wa or not wb:
                return 0.0
            return len(wa & wb) / min(len(wa), len(wb))

        filtered = []
        for q in questions:
            too_similar = any(_overlap(q, s) >= threshold for s in seen)
            if not too_similar:
                filtered.append(q)
        return filtered

    @staticmethod
    def _fallback_questions(
        tech_stack: List[str],
        seen: Optional[Set[str]] = None,
    ) -> List[str]:
        """
        Varied fallback questions grouped by common technology.
        Questions are shuffled and filtered against the seen set.
        """
        seen = seen or set()

        # Tech-specific pools
        pools: dict[str, List[str]] = {
            "python": [
                "How does Python's `with` statement relate to context managers?",
                "Explain the difference between `@staticmethod` and `@classmethod` in Python.",
                "What is the purpose of `__slots__` in a Python class?",
                "How would you handle memory-intensive data processing in Python?",
                "What makes a Python function a generator, and when would you use one?",
            ],
            "javascript": [
                "What is the difference between `var`, `let`, and `const` in JavaScript?",
                "Explain how the JavaScript event loop handles asynchronous tasks.",
                "What is a JavaScript Promise and how does it differ from a callback?",
                "How does `this` binding work in arrow functions vs regular functions?",
            ],
            "react": [
                "What problem does React's `useCallback` hook solve?",
                "How would you prevent unnecessary re-renders in a React component?",
                "What is the difference between controlled and uncontrolled components?",
                "Explain the concept of lifting state up in React.",
            ],
            "sql": [
                "What is query optimization and how does EXPLAIN help?",
                "Explain the difference between a clustered and non-clustered index.",
                "When would you use a stored procedure vs application-level logic?",
                "What are window functions in SQL and give a practical example?",
            ],
            "docker": [
                "What is the difference between a Docker image and a container?",
                "How do Docker volumes differ from bind mounts?",
                "What is a multi-stage Docker build and why is it useful?",
            ],
            "aws": [
                "What is the difference between S3 and EBS storage in AWS?",
                "How does auto-scaling work in AWS and when would you use it?",
                "What is the difference between security groups and NACLs in VPC?",
            ],
        }

        # General fallback pool (varied, non-textbook)
        general_pool = [
            "How do you approach debugging a production issue you cannot reproduce locally?",
            "What strategies do you use to review code written by others effectively?",
            "How would you design a system that needs to handle millions of requests per day?",
            "What is the trade-off between code readability and performance?",
            "Describe how you would migrate a monolithic application to microservices.",
            "How do you ensure backward compatibility when changing a public API?",
            "What is your approach to writing documentation for a complex module?",
            "How would you handle a situation where tests pass locally but fail in CI?",
        ]

        primary = (tech_stack[0] if tech_stack else "").lower()
        questions: List[str] = []

        # Try tech-specific pool first
        for key, pool in pools.items():
            if key in primary or any(key in t.lower() for t in tech_stack):
                shuffled = random.sample(pool, len(pool))
                for q in shuffled:
                    if q.lower() not in {s.lower() for s in seen}:
                        questions.append(q)

        # Fill from general pool
        general_shuffled = random.sample(general_pool, len(general_pool))
        for q in general_shuffled:
            if q.lower() not in {s.lower() for s in seen}:
                questions.append(q)

        return questions

    def _end_session(self, graceful: bool) -> None:
        self._sm.force_end()
        if graceful:
            self._session.mark_complete()
        try:
            self._storage.save(self._session)
            logger.info("Session %s saved (graceful=%s)", self._session.session_id, graceful)
        except Exception as exc:
            logger.warning("Failed to persist session: %s", exc)


# ---------------------------------------------------------------------------
# Bind dispatch table after class body
# ---------------------------------------------------------------------------

TalentScoutOrchestrator._STATE_HANDLERS = {
    ConversationState.COLLECT_NAME:       TalentScoutOrchestrator._handle_collect_name,
    ConversationState.COLLECT_EMAIL:      TalentScoutOrchestrator._handle_collect_email,
    ConversationState.COLLECT_PHONE:      TalentScoutOrchestrator._handle_collect_phone,
    ConversationState.COLLECT_EXPERIENCE: TalentScoutOrchestrator._handle_collect_experience,
    ConversationState.COLLECT_POSITION:   TalentScoutOrchestrator._handle_collect_position,
    ConversationState.COLLECT_LOCATION:   TalentScoutOrchestrator._handle_collect_location,
    ConversationState.COLLECT_TECH_STACK: TalentScoutOrchestrator._handle_collect_tech_stack,
    ConversationState.TECH_QUESTIONS:     TalentScoutOrchestrator._handle_tech_questions,
    ConversationState.WRAP_UP:            TalentScoutOrchestrator._handle_wrap_up,
    ConversationState.ENDED:              lambda self, _: reply_exit(self._session.candidate.full_name),
}