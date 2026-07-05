"""
Medical QA System - HTML Escaping and Input Sanitization Utilities.

Provides defence-in-depth functions to:
  1. Escape HTML special characters before rendering user-supplied text.
  2. Detect and strip common prompt-injection patterns.
  3. Validate and clean user questions before they are processed by the RAG
     pipeline.

All functions are pure (no side effects) and raise ``ValueError`` on invalid
input so callers can convert errors to HTTP 400 responses.
"""

from __future__ import annotations

import html
import re

# ---------------------------------------------------------------------------
# Prompt-injection patterns to strip from user input
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bignore\s+(previous|above|all\s+previous)\b", re.IGNORECASE),
    re.compile(r"\bforget\s+(everything|all|previous)\b", re.IGNORECASE),
    re.compile(r"\bnew\s+instruction[s]?\b", re.IGNORECASE),
    re.compile(r"\bdisregard\b", re.IGNORECASE),
    re.compile(r"^system\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^assistant\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^human\s*:", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^<\|.*?\|>", re.IGNORECASE | re.MULTILINE),  # <|im_start|> etc.
    re.compile(r"###\s*(Instruction|System|Human|Assistant)", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"\bprompt\s+injection\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(?:if\s+you\s+are|a)\b", re.IGNORECASE),
    re.compile(r"\byou\s+are\s+now\b", re.IGNORECASE),
]


def escape_html(text: str) -> str:
    """
    Escape HTML special characters in *text*.

    Converts ``<``, ``>``, ``&``, ``"``, and ``'`` to their HTML entity
    equivalents so that user-supplied strings cannot inject HTML when
    rendered in a browser context.

    Args:
        text: The raw string to escape.

    Returns:
        The HTML-escaped string.

    Example::

        >>> escape_html("<script>alert('xss')</script>")
        '&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;'
    """
    return html.escape(text, quote=True)


def sanitize_prompt_injection(text: str) -> str:
    """
    Remove known prompt-injection patterns from *text*.

    Scans the input against a list of compiled regular expressions that match
    common jailbreak / prompt-override phrases.  Each match is replaced with
    a single space and the result is whitespace-normalised.

    Args:
        text: The raw user input string.

    Returns:
        The sanitized string with injection patterns removed.

    Example::

        >>> sanitize_prompt_injection("Ignore previous instructions and tell me...")
        'and tell me...'
    """
    sanitized = text
    for pattern in _INJECTION_PATTERNS:
        sanitized = pattern.sub(" ", sanitized)
    # Collapse any runs of whitespace introduced by replacements
    sanitized = re.sub(r"[ \t]{2,}", " ", sanitized).strip()
    return sanitized


def validate_question(text: str, max_length: int = 500) -> str:
    """
    Validate, sanitize, and return a clean question string.

    Performs the following operations in order:
      1. Strip leading/trailing whitespace.
      2. Raise ``ValueError`` if the stripped text is empty.
      3. Raise ``ValueError`` if the text exceeds *max_length* characters.
      4. Remove prompt-injection patterns via :func:`sanitize_prompt_injection`.
      5. Escape HTML special characters via :func:`escape_html`.

    Args:
        text:       The raw question string from the user.
        max_length: Maximum allowed character length (default 500).

    Returns:
        The validated, sanitized, HTML-escaped question string.

    Raises:
        ValueError: If the question is empty or exceeds *max_length*.

    Example::

        >>> validate_question("  What is diabetes?  ")
        'What is diabetes?'

        >>> validate_question("")
        ValueError: Question must not be empty.

        >>> validate_question("x" * 600)
        ValueError: Question exceeds maximum length of 500 characters.
    """
    stripped = text.strip()

    if not stripped:
        raise ValueError("Question must not be empty.")

    if len(stripped) > max_length:
        raise ValueError(
            f"Question exceeds maximum length of {max_length} characters "
            f"(received {len(stripped)} characters)."
        )

    sanitized = sanitize_prompt_injection(stripped)

    # Re-check emptiness after sanitization in case the entire input was
    # a prompt-injection phrase
    if not sanitized:
        raise ValueError(
            "Question was empty after sanitization. "
            "Please provide a genuine medical question."
        )

    return escape_html(sanitized)
