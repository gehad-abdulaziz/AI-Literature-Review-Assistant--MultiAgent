"""
One shared chat model instance, reused by every node (Step 2 / Step 4 spirit:
keep it simple and observable). Structured-output helpers live here too so
every node asks for JSON the same way (Step 18 / Step 22 need this).
"""
import json
import re

from langchain_groq import ChatGroq

from config import GROQ_API_KEY, MODEL_NAME

llm = ChatGroq(
    model=MODEL_NAME,
    api_key=GROQ_API_KEY,
    temperature=0,
    max_tokens=8000,
    # gpt-oss models on Groq are "reasoning" models — they spend tokens
    # thinking before producing the final answer. Without this, a low
    # max_tokens can get eaten entirely by reasoning, leaving an empty
    # final response. "low" keeps more budget for the actual output.
    reasoning_effort="low",
)


class LLMCallError(Exception):
    """
    Raised when the underlying API call itself fails (network error, rate
    limit, auth, timeout, etc.) — distinct from `call_llm_json`'s ValueError,
    which means the model responded but the response wasn't valid JSON.
    Callers that want to degrade gracefully instead of crashing the whole
    graph run should catch this (and, where relevant, ValueError) rather
    than letting it propagate unhandled.
    """


def call_llm(system: str, user: str) -> str:
    """Single-turn call. Returns raw text content. Raises LLMCallError if the
    API call itself fails (never raises the underlying SDK exception type
    directly, so callers only need to catch one thing)."""
    try:
        resp = llm.invoke([("system", system), ("human", user)])
    except Exception as e:
        raise LLMCallError(f"LLM API call failed: {e}") from e
    return resp.content if isinstance(resp.content, str) else str(resp.content)


def sanitize_raw_json(text: str) -> str:
    """Cleans common unicode anomalies, smart quotes, and non-standard hyphens."""
    # Replace non-breaking hyphens (U+2011) and soft hyphens with standard ASCII hyphen (-)
    text = text.replace("\u2011", "-").replace("\u00ad", "")
    # Replace non-breaking spaces (\xa0) with standard space
    text = text.replace("\xa0", " ")
    # Normalize smart quotes to standard quotes
    text = text.replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")
    return text


def call_llm_json(system: str, user: str) -> dict | list:
    """
    Calls the LLM with an instruction to return ONLY JSON, then parses it.
    Strips markdown code fences defensively and applies a multi-stage fallback
    strategy. Can raise LLMCallError (API call failed — see call_llm) or
    ValueError (API call succeeded but the response wasn't parseable JSON
    even after repair attempts). Callers should generally catch both.
    """
    strict_system = (
        system
        + "\n\nRespond with ONLY valid JSON. No preamble, no markdown code "
        "fences, no explanation before or after the JSON. Do not use LaTeX "
        "or backslash escapes (e.g. \\(n\\), \\alpha) anywhere in the JSON "
        "string values — write math and symbols in plain text instead "
        "(e.g. 'n', 'alpha'), since stray backslashes break JSON parsing."
    )
    raw = call_llm(strict_system, user)

    # Strip markdown code fences
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    cleaned = sanitize_raw_json(cleaned)

    # Attempt 1: Direct JSON parse with strict=False (allows raw newlines inside strings)
    try:
        return json.loads(cleaned, strict=False)
    except json.JSONDecodeError:
        pass

    # Attempt 2: Extract outermost JSON object or array in case extra text was appended
    json_match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0), strict=False)
        except json.JSONDecodeError:
            pass

    # Attempt 3: Repair invalid backslash escape sequences
    repaired = re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", cleaned)
    try:
        return json.loads(repaired, strict=False)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON. Raw output:\n{raw}") from e