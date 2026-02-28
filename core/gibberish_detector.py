"""
TalentScout — Gibberish Detection Engine
==========================================
Lightweight, zero-dependency module that detects random keyboard mashing
and nonsense input across every form field in the hiring assistant.

Techniques used (no ML, no external libs):
  1. Vowel-ratio analysis  — real words have 20–55 % vowels
  2. Consonant-run length  — 5+ consecutive consonants ≈ gibberish
  3. English bigram model  — common letter pairs vs rare combos
  4. Repeated-char penalty — "aaaaaaa" / "qqqqqq"
  5. Known-tech fuzzy match — for tech stack entries
  6. Known-domain whitelist — for email TLD / domain sanity

All public functions return (is_gibberish: bool, reason: str | None).
"""

from __future__ import annotations

import re
import unicodedata
from typing import FrozenSet, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VOWELS: FrozenSet[str] = frozenset("aeiouAEIOU")

# Strictly high-frequency English bigrams only — derived from corpus frequency studies.
# Deliberately excludes rare structural pairs (sn, nf, kd, ls, etc.) so that
# gibberish words with scattered vowels still receive low bigram scores.
_COMMON_BIGRAMS: FrozenSet[str] = frozenset({
    # Vowel-pair and vowel-led (highest frequency in English text)
    "an", "ar", "al", "as", "at", "ai", "au", "ay", "am", "ap",
    "ad", "ab", "ac", "er", "en", "es", "ed", "ea", "el", "et",
    "em", "in", "it", "is", "io", "ic", "ir", "il", "im", "id",
    "on", "or", "ou", "of", "ow", "ot", "un", "us", "ur", "up",
    # High-frequency consonant-to-vowel pairs
    "th", "he", "re", "te", "ti", "to", "ha", "se", "ve",
    "ro", "ra", "ri", "hi", "ne", "me", "de", "co", "ta",
    "si", "na", "li", "la", "sa", "ca", "ma", "ce", "pa", "be",
    "lo", "le", "no", "wa", "we", "wi", "ge", "pe", "po",
    "bo", "ba", "bu", "fa", "fe", "fi", "fo", "ga", "go",
    "ho", "hu", "ja", "jo", "ju", "ke", "ki",
    "mo", "mu", "mi", "ni", "no", "nu", "pi", "pu",
    "ru", "su", "so", "tu", "vi", "yo", "ze",
    # Genuinely high-frequency consonant clusters in real English words
    "st", "nt", "nd", "ng", "ll", "ss", "sh", "ch",
    "wh", "tr", "pr", "br", "cl", "cr", "dr", "fl",
    "fr", "gr", "pl", "sc", "sp", "ph", "qu", "ck",
    "ly", "ry", "ny", "my", "nk", "nc", "mp", "mb",
    "rn", "rd", "rs", "rt", "ld", "lt",
})

# ---------------------------------------------------------------------------
# Comprehensive known-tech catalogue (≈ 230 entries, all lowercase)
# Used for fuzzy matching in validate_tech_stack.
# ---------------------------------------------------------------------------

KNOWN_TECHNOLOGIES: FrozenSet[str] = frozenset({
    # Languages
    "python", "java", "javascript", "typescript", "c", "c++", "c#", "go",
    "golang", "rust", "ruby", "php", "swift", "kotlin", "scala", "r",
    "matlab", "perl", "lua", "haskell", "elixir", "erlang", "clojure",
    "dart", "groovy", "f#", "vb.net", "cobol", "fortran", "assembly",
    "bash", "shell", "powershell", "julia", "nim", "zig", "crystal",
    "objective-c", "objective c",
    # Web / Frontend
    "react", "angular", "vue", "vue.js", "next.js", "nuxt", "nuxt.js",
    "svelte", "sveltekit", "jquery", "bootstrap", "tailwind", "tailwindcss",
    "sass", "scss", "css", "html", "html5", "css3", "webpack", "vite",
    "rollup", "parcel", "babel", "eslint", "prettier", "storybook",
    "gatsby", "remix", "astro", "htmx", "alpinejs",
    # Backend / Frameworks
    "django", "flask", "fastapi", "tornado", "aiohttp", "starlette",
    "express", "node.js", "nodejs", "koa", "nestjs", "hapi",
    "spring", "spring boot", "springboot", "quarkus", "micronaut",
    "rails", "ruby on rails", "sinatra", "laravel", "symfony", "codeigniter",
    "asp.net", "asp.net core", ".net", "dotnet", "blazor", "gin", "fiber",
    "echo", "chi", "actix", "rocket", "axum", "phoenix", "ktor",
    # Databases
    "sql", "mysql", "postgresql", "postgres", "sqlite", "mariadb",
    "mongodb", "mongo", "redis", "cassandra", "dynamodb", "firestore",
    "couchdb", "couchbase", "neo4j", "influxdb", "timescaledb",
    "elasticsearch", "opensearch", "supabase", "planetscale", "neon",
    "cockroachdb", "mssql", "sql server", "oracle", "db2",
    # Cloud / Infrastructure
    "aws", "amazon web services", "gcp", "google cloud", "azure",
    "microsoft azure", "heroku", "vercel", "netlify", "cloudflare",
    "digitalocean", "linode", "vultr", "railway", "render", "fly.io",
    # DevOps / Containers
    "docker", "kubernetes", "k8s", "helm", "terraform", "ansible",
    "puppet", "chef", "vagrant", "jenkins", "github actions", "gitlab ci",
    "circleci", "travis ci", "argo cd", "flux", "istio", "envoy",
    "nginx", "apache", "caddy", "haproxy", "traefik",
    # Data / ML / AI
    "tensorflow", "pytorch", "keras", "scikit-learn", "sklearn",
    "pandas", "numpy", "scipy", "matplotlib", "seaborn", "plotly",
    "xgboost", "lightgbm", "catboost", "huggingface", "transformers",
    "langchain", "openai", "llamaindex", "llama", "spark", "pyspark",
    "hadoop", "hive", "kafka", "airflow", "dbt", "dask", "ray",
    "mlflow", "kubeflow", "sagemaker", "vertex ai",
    # Mobile
    "android", "ios", "flutter", "react native", "xamarin", "ionic",
    "capacitor", "cordova",
    # Testing
    "jest", "pytest", "junit", "mocha", "chai", "cypress", "playwright",
    "selenium", "testng", "rspec", "minitest", "vitest",
    # Messaging / Streaming
    "rabbitmq", "kafka", "nats", "activemq", "sqs", "sns", "pubsub",
    "celery", "sidekiq", "bull",
    # Version Control / Tools
    "git", "github", "gitlab", "bitbucket", "jira", "confluence",
    "linux", "ubuntu", "debian", "centos", "fedora", "macos",
    # API / Protocols
    "rest", "graphql", "grpc", "websocket", "webrtc", "mqtt", "soap",
    "openapi", "swagger",
    # Security
    "oauth", "oauth2", "jwt", "ldap", "saml", "ssl", "tls",
})

# ---------------------------------------------------------------------------
# Common legitimate job-title words (to avoid false-positives on positions)
# ---------------------------------------------------------------------------

_JOB_WORDS: FrozenSet[str] = frozenset({
    "engineer", "developer", "architect", "analyst", "scientist", "manager",
    "lead", "senior", "junior", "principal", "staff", "head", "director",
    "vp", "cto", "ceo", "ciso", "intern", "associate", "consultant",
    "specialist", "administrator", "administrator", "coordinator", "officer",
    "devops", "backend", "frontend", "fullstack", "full", "stack", "mobile",
    "cloud", "data", "ml", "ai", "security", "qa", "test", "sre", "platform",
    "software", "hardware", "product", "project", "scrum", "agile", "tech",
    "technical", "information", "systems", "it", "web", "ui", "ux",
})

# Known common city/country fragments (partial list to anchor location checks)
_LOCATION_ANCHORS: FrozenSet[str] = frozenset({
    "india", "usa", "uk", "us", "canada", "australia", "germany", "france",
    "spain", "italy", "japan", "china", "brazil", "mexico", "russia",
    "singapore", "dubai", "uae", "london", "paris", "berlin", "tokyo",
    "beijing", "shanghai", "sydney", "melbourne", "toronto", "vancouver",
    "york", "angeles", "francisco", "delhi", "mumbai", "bangalore",
    "bengaluru", "hyderabad", "chennai", "pune", "kolkata", "ahmedabad",
    "miami", "chicago", "seattle", "boston", "austin", "denver", "phoenix",
    "amsterdam", "stockholm", "oslo", "helsinki", "zurich", "geneva",
    "remote", "hybrid", "anywhere",
})


# ---------------------------------------------------------------------------
# Core word-level analysis
# ---------------------------------------------------------------------------

def _vowel_ratio(word: str) -> float:
    """Fraction of alphabetic characters that are vowels (0.0–1.0)."""
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return 0.5  # non-alpha tokens are neutral
    return sum(1 for c in letters if c in _VOWELS) / len(letters)


def _max_consonant_run(word: str) -> int:
    """Maximum number of consecutive consonant characters."""
    max_run = run = 0
    for c in word.lower():
        if c.isalpha() and c not in _VOWELS:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _bigram_score(word: str) -> float:
    """
    Fraction of character bigrams that appear in common English.
    Score close to 0.0 → very few common bigrams → likely gibberish.
    """
    w = word.lower()
    if len(w) < 2:
        return 1.0
    bigrams = [w[i:i + 2] for i in range(len(w) - 1) if w[i].isalpha() and w[i + 1].isalpha()]
    if not bigrams:
        return 1.0
    return sum(1 for b in bigrams if b in _COMMON_BIGRAMS) / len(bigrams)


def _repeated_char_ratio(word: str) -> float:
    """
    Fraction of characters that are part of a run of 3+ identical chars.
    'aaabbb' → 1.0; 'hello' → 0.0
    """
    if not word:
        return 0.0
    count = 0
    i = 0
    while i < len(word):
        run = 1
        while i + run < len(word) and word[i + run] == word[i]:
            run += 1
        if run >= 3:
            count += run
        i += run
    return count / len(word)


def _sequential_digits(s: str) -> bool:
    """Detect monotonically sequential digit strings like 12345678 or 98765432."""
    digits = re.sub(r"\D", "", s)
    if len(digits) < 6:
        return False
    diffs = [int(digits[i + 1]) - int(digits[i]) for i in range(len(digits) - 1)]
    return len(set(diffs)) == 1  # all differences equal → pure sequence


def is_gibberish_word(word: str, strict: bool = False) -> bool:
    """
    Returns True if a single word looks like random keyboard mashing.

    Args:
        word:   A single token (no spaces).
        strict: Tighten thresholds; use for fields like name/location.
    """
    w = word.strip()
    if len(w) < 3:
        return False  # too short to judge reliably

    # Normalise accented characters to ASCII for analysis
    w_ascii = unicodedata.normalize("NFKD", w).encode("ascii", "ignore").decode()
    if not w_ascii:
        return False

    vr = _vowel_ratio(w_ascii)
    mcr = _max_consonant_run(w_ascii)
    bscore = _bigram_score(w_ascii)
    rep = _repeated_char_ratio(w_ascii.lower())

    # Hard rules — always gibberish regardless of strict flag
    if rep >= 0.5:                              # "aaaabbbb"
        return True
    if len(w_ascii) >= 4 and vr == 0.0:        # zero vowels in a 4+ char word
        return True
    if mcr >= 6:                                # "qwrtplk"
        return True

    # Soft rules — combination of signals
    signals = 0
    if vr < 0.12:          signals += 2   # extremely low vowel ratio
    elif vr < 0.22:        signals += 1   # catches 0.17-0.22 ratio words
    if mcr >= 5:           signals += 2
    elif mcr >= 4:         signals += 1
    if bscore < 0.15:      signals += 2   # almost no common bigrams
    elif bscore < 0.30:    signals += 1   # widened to catch more marginal cases

    # Extra signal: both first AND last alphabetic bigram are uncommon
    # Catches words like "lkesnf" (starts lk-, ends -nf) that otherwise
    # score OK because of accidentally common middle bigrams (ke, es).
    alpha_only = "".join(c for c in w_ascii if c.isalpha())
    if len(alpha_only) >= 4:
        first_bg = alpha_only[:2]
        last_bg  = alpha_only[-2:]
        if first_bg not in _COMMON_BIGRAMS and last_bg not in _COMMON_BIGRAMS:
            signals += 1

    threshold = 2 if strict else 3
    return signals >= threshold


def gibberish_phrase(text: str, min_gibberish_words: int = 1) -> Tuple[bool, Optional[str]]:
    """
    Analyses a phrase (multiple words) for gibberish content.

    Returns:
        (True, reason_string) if the phrase is detected as gibberish.
        (False, None)         if it looks legitimate.
    """
    words = re.split(r"[\s\-_]+", text.strip())
    alpha_words = [w for w in words if re.search(r"[a-zA-Z]", w) and len(w) >= 3]

    if not alpha_words:
        return False, None

    bad_words = [w for w in alpha_words if is_gibberish_word(w, strict=True)]

    if len(bad_words) >= min_gibberish_words and len(bad_words) >= len(alpha_words) * 0.5:
        examples = ", ".join(f"'{w}'" for w in bad_words[:3])
        return True, f"'{text}' does not appear to be a real value (detected gibberish: {examples})"

    return False, None


# ---------------------------------------------------------------------------
# Email-specific checks
# ---------------------------------------------------------------------------

def check_email_gibberish(email: str) -> Tuple[bool, Optional[str]]:
    """
    Checks whether the local-part or domain of an email contains gibberish.

    Legitimate emails like "jdoe@company.com" pass.
    Gibberish like "askdh@poaewnf.com" fails because 'askdh' has no vowels
    and 'poaewnf' has an unusual consonant cluster.
    """
    email = email.strip().lower()
    parts = email.split("@")
    if len(parts) != 2:
        return False, None  # format already caught by regex validator

    local, domain = parts[0], parts[1]
    domain_base = domain.split(".")[0]  # strip TLD

    # Check local part
    if len(local) >= 3 and is_gibberish_word(local, strict=False):
        return True, (
            f"The email address '{email}' looks like it may have been typed randomly. "
            "Please enter your real email address."
        )

    # Check domain base (but be lenient — many legitimate short domains exist)
    if len(domain_base) >= 5 and is_gibberish_word(domain_base, strict=False):
        return True, (
            f"The email domain '{domain}' does not look like a real email provider. "
            "Please use your actual email address."
        )

    return False, None


# ---------------------------------------------------------------------------
# Phone-specific checks
# ---------------------------------------------------------------------------

def check_phone_gibberish(phone: str) -> Tuple[bool, Optional[str]]:
    """
    Detects obviously fake phone numbers:
      - All same digit: 9999999999
      - Sequential run:  1234567890
      - Placeholder:     0000000000 / 1111111111
    """
    digits = re.sub(r"\D", "", phone)
    if not digits:
        return False, None

    # All same digit
    if len(set(digits)) == 1:
        return True, (
            f"'{phone}' does not appear to be a real phone number "
            "(all identical digits). Please enter your actual number."
        )

    # Sequential digits (ascending or descending)
    if len(digits) >= 7 and _sequential_digits(digits):
        return True, (
            f"'{phone}' looks like a sequential placeholder. "
            "Please enter your real phone number."
        )

    # Known placeholder patterns
    placeholders = {"1234567890", "0987654321", "1111111111", "0000000000",
                    "9999999999", "1234567", "7654321", "123456789", "987654321"}
    if digits in placeholders or digits[:10] in placeholders:
        return True, (
            f"'{phone}' appears to be a placeholder number. "
            "Please enter your actual phone number."
        )

    return False, None


# ---------------------------------------------------------------------------
# Tech-stack fuzzy matching
# ---------------------------------------------------------------------------

def _edit_distance(a: str, b: str) -> int:
    """Simple Levenshtein distance (O(m*n))."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _tech_similarity(entry: str) -> float:
    """
    Returns a 0.0–1.0 similarity score for how closely `entry` resembles
    any known technology. 1.0 = exact match.
    """
    e = entry.lower().strip()
    if e in KNOWN_TECHNOLOGIES:
        return 1.0

    best = 0.0
    for tech in KNOWN_TECHNOLOGIES:
        # Substring bonus
        if e in tech or tech in e:
            length_ratio = min(len(e), len(tech)) / max(len(e), len(tech))
            best = max(best, 0.7 * length_ratio + 0.3)
            continue
        # Edit-distance similarity
        max_len = max(len(e), len(tech))
        if max_len == 0:
            continue
        dist = _edit_distance(e, tech)
        sim = 1.0 - (dist / max_len)
        if sim > best:
            best = sim
    return best


def check_tech_entry(entry: str) -> Tuple[bool, Optional[str]]:
    """
    Returns (is_invalid, reason) for a single tech stack entry.

    An entry is invalid if:
      - It is pure gibberish (no resemblance to any known technology), AND
      - It looks like random keyboard mashing.

    We are lenient: niche/new technologies that score moderately are allowed.
    """
    e = entry.strip()
    if not e:
        return False, None

    sim = _tech_similarity(e)

    # Accepted: close enough to a real technology
    if sim >= 0.55:
        return False, None

    # Check for gibberish pattern as a secondary gate
    if is_gibberish_word(e, strict=True):
        return True, (
            f"'{e}' does not appear to be a recognized technology. "
            "Please check the spelling or use a technology from your actual skill set."
        )

    # Low similarity — also check individual words in multi-word entries
    words_in_entry = e.split()
    if len(words_in_entry) > 1:
        gibberish_words = [w for w in words_in_entry if len(w) >= 3 and is_gibberish_word(w, strict=True)]
        if len(gibberish_words) >= len(words_in_entry) * 0.6:
            return True, (
                f"'{entry.strip()}' does not look like a real technology. "
                "Please check the spelling or list technologies you actually use."
            )

    # Low similarity but not obviously gibberish — might be a niche/new tool
    return False, None


def check_tech_stack(techs: List[str]) -> Tuple[bool, Optional[str]]:
    """
    Validates an entire list of tech stack entries.

    Returns failure only if the MAJORITY of entries are unrecognized gibberish,
    to avoid blocking legitimate niche technologies.
    """
    if not techs:
        return False, None

    bad: List[str] = []
    for tech in techs:
        is_bad, _ = check_tech_entry(tech)
        if is_bad:
            bad.append(tech)

    # Block only when more than half are invalid
    if len(bad) > 0 and len(bad) >= len(techs) * 0.6:
        bad_display = ", ".join(f"'{b}'" for b in bad[:4])
        return True, (
            f"The following entries do not appear to be real technologies: {bad_display}. "
            "Please list actual programming languages, frameworks, or tools you use."
        )

    return False, None


# ---------------------------------------------------------------------------
# Position / role check
# ---------------------------------------------------------------------------

def check_position_gibberish(position: str) -> Tuple[bool, Optional[str]]:
    """
    Checks whether a job title looks legitimate.

    Strategy: at least one word must be either a known job word OR a real
    English-looking word (passes the anti-gibberish test).
    """
    words = re.split(r"[\s/\-,]+", position.strip().lower())
    alpha_words = [w for w in words if re.search(r"[a-zA-Z]", w)]

    if not alpha_words:
        return True, "Please enter a valid job title (e.g. **Backend Engineer**, **Data Scientist**)."

    # If any word is a known job word → definitely legitimate
    if any(w in _JOB_WORDS for w in alpha_words):
        return False, None

    # Otherwise require at least one non-gibberish word
    non_gibberish = [w for w in alpha_words if len(w) >= 3 and not is_gibberish_word(w, strict=True)]
    if non_gibberish:
        return False, None

    return True, (
        f"'{position}' does not look like a valid job title. "
        "Please enter your actual desired position (e.g. **Backend Engineer**, **ML Engineer**)."
    )


# ---------------------------------------------------------------------------
# Location check
# ---------------------------------------------------------------------------

def check_location_gibberish(location: str) -> Tuple[bool, Optional[str]]:
    """
    Checks whether a location string looks plausible.

    Strategy:
      - If any word matches a known location anchor → accept.
      - Otherwise check all words for gibberish; reject if all are gibberish.
    """
    loc_lower = location.strip().lower()
    words = re.split(r"[\s,./\-]+", loc_lower)
    alpha_words = [w for w in words if re.search(r"[a-zA-Z]", w) and len(w) >= 3]

    if not alpha_words:
        return False, None  # purely numeric → let other validators handle

    # Known anchor match (fast path)
    if any(w in _LOCATION_ANCHORS for w in alpha_words):
        return False, None

    # Check if all content words are gibberish
    gibberish_words = [w for w in alpha_words if is_gibberish_word(w, strict=True)]

    if gibberish_words and len(gibberish_words) >= len(alpha_words) * 0.6:
        examples = ", ".join(f"'{w}'" for w in gibberish_words[:2])
        return True, (
            f"'{location}' does not appear to be a real location (gibberish detected: {examples}). "
            "Please enter your actual city and country (e.g. **Mumbai, India**)."
        )

    # Secondary check: consonant-heavy words (low bigram score AND consonant run >= 3)
    suspicious = [
        w for w in alpha_words
        if _bigram_score(w) < 0.30 and _max_consonant_run(w) >= 3
    ]
    if suspicious and len(suspicious) >= len(alpha_words) * 0.5:
        examples = ", ".join(f"'{w}'" for w in suspicious[:2])
        return True, (
            f"'{location}' does not look like a real city or country ({examples} not recognised). "
            "Please enter your actual location (e.g. **Mumbai, India** or **London, UK**)."
        )

    return False, None