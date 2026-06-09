"""Shared LLM factory for all agents.

Supports both OpenAI directly and OpenRouter (OpenAI-compatible API).
"""

import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """Return a ChatOpenAI client — uses OpenAI directly if OPENAI_API_KEY is set."""
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            api_key=openai_key,
            max_tokens=512,
            temperature=0.3,
            max_retries=3,
        )
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL", "google/gemma-4-31b-it:free"),
        openai_api_key=os.getenv("OPENROUTER_API_KEY"),
        openai_api_base="https://openrouter.ai/api/v1",
        max_tokens=512,
        temperature=0.3,
        max_retries=3,
    )
