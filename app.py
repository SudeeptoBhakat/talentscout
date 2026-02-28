"""
TalentScout — Streamlit Application Entry Point
Run with: streamlit run app.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path as _Path

import streamlit as st
from dotenv import load_dotenv

# Bootstrap
_env_file = _Path(".env") if _Path(".env").exists() else _Path(".env.example")
load_dotenv(dotenv_path=_env_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

from core.orchestrator import TalentScoutOrchestrator  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="TalentScout - AI Hiring Assistant",
    page_icon="T",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .stChatMessage [data-testid="stChatMessageContent"] { border-radius: 10px; }
    div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.header("Configuration")
        st.caption("Select your LLM provider and enter the API key.")

        provider = st.selectbox(
            "LLM Provider",
            options=["google", "openai", "mock"],
            index=0,
            help="'mock' works offline without any API key.",
        )

        api_key = ""
        if provider == "google":
            api_key = st.text_input(
                "Google API Key",
                value=os.getenv("GOOGLE_API_KEY", ""),
                type="password",
                placeholder="AIza...",
                help="Get a free key at https://aistudio.google.com/app/apikey",
            )
        elif provider == "openai":
            api_key = st.text_input(
                "OpenAI API Key",
                value=os.getenv("OPENAI_API_KEY", ""),
                type="password",
                placeholder="sk-...",
            )
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key

        st.divider()
        st.caption("Tip: choose 'mock' to test without an API key.")

    return api_key, provider


# ---------------------------------------------------------------------------
# Session init
# ---------------------------------------------------------------------------

def _init_session(api_key: str = "", provider: str = "google") -> bool:
    os.environ["TALENTSCOUT_LLM_PROVIDER"] = provider
    if api_key and provider == "google":
        os.environ["GOOGLE_API_KEY"] = api_key
    elif api_key and provider == "openai":
        os.environ["OPENAI_API_KEY"] = api_key

    if provider == "google" and not os.getenv("GOOGLE_API_KEY"):
        return False
    if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
        return False

    if "orchestrator" in st.session_state:
        if st.session_state.get("_ts_provider") != provider:
            del st.session_state["orchestrator"]
            st.session_state.pop("messages", None)

    st.session_state["_ts_provider"] = provider

    if "orchestrator" not in st.session_state:
        from core.llm_service import LLMConfigurationError
        try:
            num_q = int(os.getenv("TALENTSCOUT_NUM_QUESTIONS", "5"))
            orc = TalentScoutOrchestrator(num_questions=num_q)
            st.session_state.orchestrator = orc
        except LLMConfigurationError as exc:
            st.error(f"Configuration error: {exc}")
            return False
        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            logger.exception("Orchestrator init failed: %s", exc)
            return False

    if "messages" not in st.session_state:
        orc: TalentScoutOrchestrator = st.session_state.orchestrator
        greeting = orc.get_greeting()
        st.session_state.messages = [{"role": "assistant", "content": greeting}]

    return True


# ---------------------------------------------------------------------------
# Chat history
# ---------------------------------------------------------------------------

def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


# ---------------------------------------------------------------------------
# Handle user input
# ---------------------------------------------------------------------------

def _handle_user_input(user_text: str) -> None:
    orc: TalentScoutOrchestrator = st.session_state.orchestrator

    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Processing..."):
            bot_reply = orc.handle_message(user_text)
        st.markdown(bot_reply)

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})


# ---------------------------------------------------------------------------
# Result summary panel
# ---------------------------------------------------------------------------

def _render_result_panel(orc: TalentScoutOrchestrator) -> None:
    """Professional result summary panel — no emojis."""
    session = orc.session
    results = session.results

    if not results:
        return

    st.divider()
    st.subheader("Interview Result Summary")

    # Score metrics row
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Score", f"{session.total_score} / {session.max_score}")
    col2.metric("Percentage", f"{session.percentage}%")
    col3.metric("Performance", session.performance_label)

    st.divider()

    # Per-question breakdown
    st.markdown("#### Detailed Breakdown")
    table_data = {
        "Q":        [f"Q{i + 1}" for i in range(len(results))],
        "Question": [r.question for r in results],
        "Score":    [f"{r.score} / 5" for r in results],
        "Feedback": [r.feedback or "-" for r in results],
    }
    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Q":        st.column_config.TextColumn(width="small"),
            "Question": st.column_config.TextColumn(width="large"),
            "Score":    st.column_config.TextColumn(width="small"),
            "Feedback": st.column_config.TextColumn(width="large"),
        },
    )

    st.divider()

    # Strengths & areas to improve
    col_s, col_i = st.columns(2)
    with col_s:
        st.markdown("#### Strengths")
        strengths = session.strengths
        if strengths:
            for s in strengths:
                st.success(s[:90] + ("..." if len(s) > 90 else ""))
        else:
            st.info("No questions scored 4 or above.")

    with col_i:
        st.markdown("#### Areas to Improve")
        improvements = session.areas_to_improve
        if improvements:
            for item in improvements:
                st.warning(item[:90] + ("..." if len(item) > 90 else ""))
        else:
            st.success("No low-scoring areas detected.")

    st.divider()

    with st.expander("Candidate Profile", expanded=False):
        st.markdown(session.candidate.to_summary())

    st.caption(
        f"Session: {session.session_id}   |   "
        f"Completed: {session.completed_at or '-'}"
    )


# ---------------------------------------------------------------------------
# Restart button
# ---------------------------------------------------------------------------

def _render_restart_button() -> None:
    """Replaces the chat input after session ends."""
    st.divider()
    if st.button("Restart Interview", type="primary", use_container_width=True):
        for key in ["orchestrator", "messages", "_ts_provider"]:
            st.session_state.pop(key, None)
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("TalentScout")
    st.caption("AI-Powered Technical Screening Assistant")
    st.divider()

    api_key, provider = _render_sidebar()

    if not _init_session(api_key=api_key, provider=provider):
        st.info(
            "Enter your API key in the sidebar to begin, "
            "or choose 'mock' to test without a key."
        )
        return

    _render_history()

    orc: TalentScoutOrchestrator = st.session_state.orchestrator

    if orc.is_ended:
        st.info("Session complete. Start a new interview using the button below.")
        _render_result_panel(orc)
        _render_restart_button()
        return

    if user_input := st.chat_input("Type your response here..."):
        _handle_user_input(user_input)
        if orc.is_ended:
            _render_result_panel(orc)
            _render_restart_button()


if __name__ == "__main__":
    main()
