# CS4241 Manual RAG Chatbot — LLM Client (PART D)
# Student: Yaw Acheampong Ahenkora Gyamera | Index: 10022200141
#
# Thin wrapper around the official OpenAI SDK.
# Uses gpt-4o-mini by default (cheap, fast, good instruction-following).
# No LangChain / LlamaIndex — raw SDK calls only.

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import OpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from src.logger import log_stage

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not OPENAI_API_KEY:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and add your key."
            )
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


def call_llm(
    prompt: str,
    model: str = OPENAI_MODEL,
    temperature: float = LLM_TEMPERATURE,
    max_tokens: int = LLM_MAX_TOKENS,
    system_msg: str | None = None,
) -> str:
    """
    Send a prompt to the LLM and return the response text.

    Parameters
    ----------
    prompt      : the full prompt string (already includes context + question)
    model       : OpenAI model name
    temperature : sampling temperature (0.2 for reproducibility)
    max_tokens  : max response length
    system_msg  : optional system message (overrides default if provided)

    Returns
    -------
    str — the assistant's response text
    """
    client = get_client()

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})

    messages.append({"role": "user", "content": prompt})

    log_stage("LLM_CALL", {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "prompt_chars": len(prompt),
    })

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    answer = response.choices[0].message.content or ""
    usage = response.usage

    log_stage("LLM_RESPONSE", {
        "model": model,
        "prompt_tokens": usage.prompt_tokens if usage else None,
        "completion_tokens": usage.completion_tokens if usage else None,
        "answer_preview": answer[:150],
    })

    return answer


def call_llm_no_context(query: str, model: str = OPENAI_MODEL) -> str:
    """
    Call the LLM with NO retrieved context — pure LLM baseline for PART E comparison.
    """
    prompt = (
        f"Answer the following question about Ghana to the best of your knowledge. "
        f"Be factual and concise.\n\nQuestion: {query}\n\nAnswer:"
    )
    log_stage("LLM_NO_CONTEXT_CALL", {"query": query[:80]})
    return call_llm(prompt, model=model)
