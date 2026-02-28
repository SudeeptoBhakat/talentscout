"""
TalentScout — Enum-Based State Machine
=========================================
Defines all conversation states and provides a guarded transition manager.

State Flow:
    GREETING
        └─► COLLECT_NAME
                └─► COLLECT_EMAIL
                        └─► COLLECT_PHONE
                                └─► COLLECT_EXPERIENCE
                                        └─► COLLECT_POSITION
                                                └─► COLLECT_LOCATION
                                                        └─► COLLECT_TECH_STACK
                                                                └─► TECH_QUESTIONS
                                                                        └─► WRAP_UP
                                                                                └─► ENDED
    Any state ──(exit intent)──► ENDED
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Dict, Set


# ---------------------------------------------------------------------------
# State Enumeration
# ---------------------------------------------------------------------------

class ConversationState(Enum):
    """
    Exhaustive set of states the TalentScout hiring assistant can be in.
    Each state corresponds to a distinct phase of the interview conversation.
    """

    GREETING         = auto()   # Initial greeting hasn't been sent yet
    COLLECT_NAME     = auto()   # Waiting for candidate's full name
    COLLECT_EMAIL    = auto()   # Waiting for a valid email address
    COLLECT_PHONE    = auto()   # Waiting for a valid phone number
    COLLECT_EXPERIENCE = auto() # Waiting for years of experience (numeric)
    COLLECT_POSITION = auto()   # Waiting for desired job position
    COLLECT_LOCATION = auto()   # Waiting for current location
    COLLECT_TECH_STACK = auto() # Waiting for tech stack (comma-separated)
    TECH_QUESTIONS   = auto()   # Generating + asking technical questions
    WRAP_UP          = auto()   # Concluding the interview, saving session
    ENDED            = auto()   # Conversation has terminated (graceful or exit)


# ---------------------------------------------------------------------------
# Allowed State Transitions (Adjacency Map)
# ---------------------------------------------------------------------------

#: Maps each state to the set of states it is permitted to transition into.
ALLOWED_TRANSITIONS: Dict[ConversationState, Set[ConversationState]] = {
    ConversationState.GREETING:            {ConversationState.COLLECT_NAME,     ConversationState.ENDED},
    ConversationState.COLLECT_NAME:        {ConversationState.COLLECT_EMAIL,    ConversationState.ENDED},
    ConversationState.COLLECT_EMAIL:       {ConversationState.COLLECT_PHONE,    ConversationState.ENDED},
    ConversationState.COLLECT_PHONE:       {ConversationState.COLLECT_EXPERIENCE, ConversationState.ENDED},
    ConversationState.COLLECT_EXPERIENCE:  {ConversationState.COLLECT_POSITION, ConversationState.ENDED},
    ConversationState.COLLECT_POSITION:    {ConversationState.COLLECT_LOCATION, ConversationState.ENDED},
    ConversationState.COLLECT_LOCATION:    {ConversationState.COLLECT_TECH_STACK, ConversationState.ENDED},
    ConversationState.COLLECT_TECH_STACK:  {ConversationState.TECH_QUESTIONS,   ConversationState.ENDED},
    ConversationState.TECH_QUESTIONS:      {ConversationState.WRAP_UP,          ConversationState.ENDED},
    ConversationState.WRAP_UP:             {ConversationState.ENDED},
    ConversationState.ENDED:               set(),  # Terminal state — no further transitions
}


# ---------------------------------------------------------------------------
# State Manager
# ---------------------------------------------------------------------------

class StateManager:
    """
    Manages the current conversation state and enforces legal transitions.

    Raises:
        InvalidTransitionError: When an illegal state transition is attempted.

    Usage::

        sm = StateManager()
        sm.transition(ConversationState.COLLECT_NAME)
        print(sm.current_state)  # ConversationState.COLLECT_NAME
    """

    def __init__(self) -> None:
        self._state: ConversationState = ConversationState.GREETING

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_state(self) -> ConversationState:
        """The current active conversation state (read-only)."""
        return self._state

    @property
    def is_ended(self) -> bool:
        """Returns True if the conversation has reached the ENDED terminal state."""
        return self._state == ConversationState.ENDED

    @property
    def is_collecting(self) -> bool:
        """Returns True if the bot is currently in a data-collection state."""
        collection_states = {
            ConversationState.COLLECT_NAME,
            ConversationState.COLLECT_EMAIL,
            ConversationState.COLLECT_PHONE,
            ConversationState.COLLECT_EXPERIENCE,
            ConversationState.COLLECT_POSITION,
            ConversationState.COLLECT_LOCATION,
            ConversationState.COLLECT_TECH_STACK,
        }
        return self._state in collection_states

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def transition(self, next_state: ConversationState) -> None:
        """
        Transition to the given state if it is a legal move from the current state.

        Args:
            next_state: The target state to transition into.

        Raises:
            InvalidTransitionError: If the transition is not permitted.
        """
        allowed = ALLOWED_TRANSITIONS.get(self._state, set())
        if next_state not in allowed:
            raise InvalidTransitionError(
                from_state=self._state,
                to_state=next_state,
                allowed=allowed,
            )
        self._state = next_state

    def force_end(self) -> None:
        """
        Immediately transitions to ENDED, regardless of current state.
        Used when exit intent is detected or an unrecoverable error occurs.
        """
        self._state = ConversationState.ENDED

    def __repr__(self) -> str:
        return f"StateManager(current={self._state.name})"


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class InvalidTransitionError(Exception):
    """
    Raised when an illegal state transition is attempted in the StateManager.

    Attributes:
        from_state: The state the machine was in before the transition attempt.
        to_state:   The target state that was requested.
        allowed:    The set of legally reachable states from `from_state`.
    """

    def __init__(
        self,
        from_state: ConversationState,
        to_state: ConversationState,
        allowed: Set[ConversationState],
    ) -> None:
        self.from_state = from_state
        self.to_state = to_state
        self.allowed = allowed
        allowed_names = ", ".join(s.name for s in allowed) or "none"
        super().__init__(
            f"Cannot transition from {from_state.name} → {to_state.name}. "
            f"Allowed: [{allowed_names}]"
        )
