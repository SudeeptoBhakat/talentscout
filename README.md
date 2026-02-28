# TalentScout — AI-Powered Hiring Assistant

## Project Overview
TalentScout is a production-ready AI screening system. It acts as the first line of technical interviews, collecting structured candidate details (name, email, phone, experience, position, location, and tech stack) before dynamically generating a short, customized technical assessment based exclusively on the candidate's declared technologies.

Each answer is evaluated in real-time by an LLM, scored from 0-5, and provided with concise feedback. At the conclusion of the session, the system displays a comprehensive hiring summary including a performance label (Strong Candidate, Moderate, Needs Improvement), a breakdown of strengths and areas to improve, and a coplete question-by-question scoring table.

## Installation Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd talentscout
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment setup:**
   Copy `.env.example` to `.env` and assign your API keys if you plan to use them without UI setup:
   ```bash
   cp .env.example .env
   ```

## Usage Guide

To launch the Streamlit interface:
```bash
streamlit run app.py
```

- **Configuration:** By default, the app sidebar opens to let you choose an LLM provider:
  - **mock:** Run completely offline with predefined tests
  - **google:** Uses Gemini 2.0 Flash (requires Google AI API Key)
  - **openai:** Uses GPT-4o-mini (requires OpenAI API Key)
- **Interaction:** Follow the chatbot's prompts. Type your answers to the background questions, then answer the generated technical questions.
- **Commands:** Type `exit` at any time to immediately safely close the session and skip right to evaluation.
- **Results:** Upon finishing, the chat box becomes disabled. You will be presented with an evaluation UI and a "Restart Interview" button to test again.

## Technical Details
- **Frontend / Framework:** Streamlit. Used for both input data handling and final evaluation display.
- **Architecture:** The application leverages an **Enum-based State Machine** (`ConversationState`, `StateManager`) to orchestrate a robust dialogue flow. User messages are routed securely dependent on the state (greeting → data variables → tech questions → wrap up).
- **Validation Engine:** Strict regex-based parsing, accompanied by a custom offline **Gibberish Detection Engine**, blocks nonsense keyboard mashing and validates fields like email shapes, generic job titles, phone numbers, and actual known tools/frameworks.
- **LLM Abstraction Layer:** Interacts dynamically via `BaseLLMService`. Provider logic resides cleanly in subclasses (`GoogleLLMService`, `OpenAILLMService`, `MockLLMService`), keeping UI rendering disjointed from complex tool selection layers.

## Prompt Design Explanation

Our prompts rely heavily on zero-shot strict-output instructions. The goal is programmatic integration without messy string parsing.
1. **Dynamic Questioning (`build_tech_questions_prompt`)**: Distributes questions fairly across a provided tech array (so 1 Python and 1 Java equals two varying queries). A seniority inferencer tailors difficulty (Junior vs Senior keywords). Output is strictly instructed to be a numbered list to make split-parsing easy.
2. **Deterministic Evaluation (`build_evaluation_prompt`)**: Demands the LLM output *ONLY* a JSON object: `{"score": X, "feedback": "Y"}`. The temperature defaults to `0.0` to force consistent object generation and rule out flowery prose. Fallback parsers execute regular expressions (`r"\{.*\}"`) on the resulting payload to isolate the JSON object if the LLM wrongly pads it in Markdown code blocks.

## Challenges & Solutions
- **Handling API Configuration Errors:** If no `GOOGLE_API_KEY` was found, the app initially crashed completely preventing users from starting. **Solution:** Extracted LLM initialization out of module scope and directly into the Streamlit session state initialization function, falling back gracefully to a sidebar prompt.
- **LLM Evaluation Data Hallucinations**: Prompt generation returned JSON wrapped in Markdown (```json ... ```) breaking standard `json.loads`. **Solution:** Modified the `_parse_evaluation` system with 4 successive strategies. It tries standard parsing, then strips code fences, then hunts the string via greedy regex, then finally operates a fallback line-search pattern.
- **Circular Imports**: Building out a separated `prompt_engine`, `orchestrator`, and `llm_service` yielded structural import chains inside `core/__init__.py`. **Solution:** Deliberately stripped eager imports of the controller classes from `__init__.py`, loading them directly at caller scopes (i.e., `import core.orchestrator` inside `app.py`).
- **Fake User Inputs ("adffsfds")**: We wanted robustness without wasting API overhead verifying basic fields. **Solution:** Invented a lightweight Gibberish Detector tracking consonant clusters, vowel ratios, and frequency mapping alongside a known framework dictionary, validating inputs locally.

## Video Link
*(Please insert the link to your walkthrough or demonstration video here)*

---
*Built as a state-of-the-art AI recruitment screener.*
