"""
Call x.ai (Grok) to map a natural-language query to semantic layer tags.
Returns {"tags": ["Tag1", ...], "explanation": "..."} for use by run_query.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

XAI_API_URL = "https://api.x.ai/v1/chat/completions"
MODEL = "grok-3-fast"  # override with XAI_MODEL; valid: grok-3, grok-3-fast, grok-3-mini, grok-3-mini-fast

# Valid tags from semantic layer (must match src.semantic.list_tags())
VALID_TAGS = frozenset(
    {
        "High Energy",
        "Low Energy",
        "Positive",
        "Negative",
        "Calm Focus",
        "Dramatic",
        "Neutral",
    }
)

SYSTEM_PROMPT = """You are a music analyst. We have a catalog of songs with predicted emotion: arousal (energy, 1-9) and valence (mood, 1-9). Songs are tagged as: High Energy, Low Energy, Positive, Negative, Calm Focus, Dramatic, Neutral.

Given the user's request, respond with ONLY a single JSON object (no markdown, no extra text):
{"tags": ["Tag1", "Tag2"], "explanation": "One short sentence describing what you're recommending."}

Use only these exact tag names: High Energy, Low Energy, Positive, Negative, Calm Focus, Dramatic, Neutral. Pick one or more tags that best match the request. "explanation" should be one sentence for the user."""


def _call_grok(user_query: str, api_key: str) -> dict[str, Any]:
    """POST to x.ai chat/completions; return parsed JSON from the reply."""
    model = os.environ.get("XAI_MODEL") or MODEL
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        "temperature": 0.3,
        "max_tokens": 200,
    }
    req = urllib.request.Request(
        XAI_API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "music-intelligence-pipeline/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        err_body = err.read().decode("utf-8", errors="replace")
        raise ValueError(f"x.ai HTTP {err.code}: {err_body[:500]}")
    if not data.get("choices") or not data["choices"][0].get("message", {}).get(
        "content"
    ):
        raise ValueError("Empty or invalid response from x.ai")
    return data


def _parse_llm_response(content: str) -> dict[str, Any]:
    """Extract JSON from LLM content; validate tags; return {"tags": [...], "explanation": "..."}."""
    # Strip markdown code block if present
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```\w*\n?", "", content)
        content = re.sub(r"\n?```\s*$", "", content)
    raw = json.loads(content)
    tags = raw.get("tags") or []
    if not isinstance(tags, list):
        tags = [t.strip() for t in str(tags).split(",")]
    tags = [t for t in tags if t in VALID_TAGS]
    if not tags:
        tags = ["Neutral"]
    explanation = (raw.get("explanation") or "Songs matching your request.").strip()
    return {"tags": tags, "explanation": explanation}


# Fallback when XAI_API_KEY is not set (e.g. CI): simple keyword -> tags
_FALLBACK_RULES = [
    (r"\bcalm\b|studying|focus", ["Calm Focus"]),
    (r"\benergetic\b|high energy|upbeat", ["High Energy", "Positive"]),
    (r"\bdramatic\b|tense|dark", ["Dramatic"]),
    (r"\bpositive\b|happy|uplifting", ["Positive"]),
    (r"\bnegative\b|sad|melancholic", ["Negative"]),
    (r"\blow energy\b|chill|subdued", ["Low Energy"]),
    (r"\bneutral\b|middle", ["Neutral"]),
]


def query_to_intent_fallback(user_query: str) -> dict[str, Any]:
    """Rule-based intent when LLM is not available. Returns {"tags": [...], "explanation": "..."}."""
    q = user_query.strip().lower()
    for pattern, tags in _FALLBACK_RULES:
        if re.search(pattern, q, re.IGNORECASE):
            return {
                "tags": tags,
                "explanation": f"Songs matching: {', '.join(tags)} (rule-based).",
            }
    return {
        "tags": ["Neutral"],
        "explanation": "Songs from the middle of the emotion space (no key match).",
    }


def query_to_intent(user_query: str) -> dict[str, Any]:
    """
    Map user query to semantic tags using Grok (x.ai) when XAI_API_KEY is set.
    Otherwise use rule-based fallback so the script runs without a key (e.g. CI).
    Returns {"tags": ["Tag1", ...], "explanation": "..."}.
    """
    api_key = os.environ.get("XAI_API_KEY")
    if not api_key or not api_key.strip():
        return query_to_intent_fallback(user_query)
    try:
        response = _call_grok(user_query.strip(), api_key.strip())
        content = response["choices"][0]["message"]["content"]
        return _parse_llm_response(content)
    except (ValueError, urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
        import sys

        print(f"Grok fallback (x.ai error): {e}", file=sys.stderr)
        return query_to_intent_fallback(user_query)
