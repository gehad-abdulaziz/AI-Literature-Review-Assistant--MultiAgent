"""
Central configuration. Loads keys from environment variables.
Never hard-code secrets here (Phase 0 / Step 3).
"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is not set. Add it to your .env file."
    )

# Model used for every LLM node. Swap freely.
# This is scoped to what your Groq account actually has access to
# (confirmed via GET /openai/v1/models) — Llama models were not in that
# list, so we use a GPT-OSS model instead, which was.
MODEL_NAME = os.environ.get("MODEL_NAME", "openai/gpt-oss-20b")

# arXiv is keyless, so no key needed. Semantic Scholar key is optional.
SEMANTIC_SCHOLAR_API_KEY = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")  # optional


# Tunables mentioned throughout the roadmap
MAX_SEARCH_RETRIES = int(os.environ.get("MAX_SEARCH_RETRIES", "2"))
MIN_RELEVANT_PAPERS = int(os.environ.get("MIN_RELEVANT_PAPERS", "3"))
MAX_PAPERS_PER_ROUND = int(os.environ.get("MAX_PAPERS_PER_ROUND", "8"))