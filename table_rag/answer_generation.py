"""Generate structured, source-grounded answers with a local Ollama model."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel, Field

from table_rag.context_builder import ContextPackage


DEFAULT_ANSWER_MODEL = "qwen3.5:9b"
DEFAULT_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
DEFAULT_NUM_CONTEXT = 32_768
DEFAULT_MAX_OUTPUT_TOKENS = 8_192
DEFAULT_REQUEST_TIMEOUT_SECONDS = 600
PROMPT_VERSION = "answer_generation_v1"
DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[1] / "prompts" / f"{PROMPT_VERSION}.md"
)
_TABLE_CITATION = re.compile(r"\[(tbl_\d{3})\]")
_TABLE_ID = re.compile(r"tbl_\d{3}")


class AnswerPayload(BaseModel):
    """Schema enforced by Ollama structured output and Pydantic validation."""

    status: Literal["answered", "insufficient_evidence"]
    answer: str = Field(min_length=1)
    cited_table_ids: list[str]
    calculation_or_evidence: str = Field(min_length=1)


@dataclass(frozen=True)
class GeneratedAnswer:
    """Validated model output with its configuration and provenance."""

    status: str
    text: str
    model: str
    cited_table_ids: tuple[str, ...]
    calculation_or_evidence: str
    response_id: str | None
    prompt_version: str


def load_answer_instructions(path: str | Path = DEFAULT_PROMPT_PATH) -> str:
    """Load the versioned prompt and reject an empty prompt file."""
    instructions = Path(path).read_text(encoding="utf-8").strip()
    if not instructions:
        raise ValueError(f"Answer-generation prompt is empty: {path}")
    return instructions


def validate_answer_payload(
    payload: AnswerPayload, allowed_table_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Validate declared citations and the inline [tbl_###] citations."""
    citations = tuple(payload.cited_table_ids)
    if len(citations) != len(set(citations)):
        raise ValueError("The answer contains duplicate declared table citations")
    malformed = [table_id for table_id in citations if not _TABLE_ID.fullmatch(table_id)]
    if malformed:
        raise ValueError(f"The answer contains malformed table IDs: {malformed}")
    invalid = sorted(set(citations) - set(allowed_table_ids))
    if invalid:
        raise ValueError(f"The answer cited tables outside its context: {invalid}")

    inline = tuple(dict.fromkeys(_TABLE_CITATION.findall(payload.answer)))
    if set(inline) != set(citations):
        raise ValueError(
            "Inline [tbl_###] citations must match cited_table_ids exactly"
        )
    if payload.status == "answered" and not citations:
        raise ValueError("An answered response must cite at least one supplied table")
    return citations


def _ollama_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    response_schema: dict[str, Any],
    num_context: int,
    max_output_tokens: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Call the local Ollama HTTP API without another Python dependency."""
    request_body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "format": response_schema,
        "think": False,
        "options": {
            "temperature": 0,
            "num_ctx": num_context,
            "num_predict": max_output_tokens,
        },
    }
    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(
            "Cannot reach Ollama at "
            f"{base_url}. Install/start Ollama and pull {model!r}."
        ) from exc


def generate_answer(
    context: ContextPackage,
    *,
    model: str = DEFAULT_ANSWER_MODEL,
    base_url: str = DEFAULT_OLLAMA_BASE_URL,
    num_context: int = DEFAULT_NUM_CONTEXT,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout_seconds: int = DEFAULT_REQUEST_TIMEOUT_SECONDS,
    prompt_path: str | Path = DEFAULT_PROMPT_PATH,
    client: Any | None = None,
) -> GeneratedAnswer:
    """Call a local Ollama model with a schema and validate its citations.

    A compatible client can be injected for deterministic tests. No API key or
    paid cloud service is required for the default local runtime.
    """
    if not isinstance(context, ContextPackage) or not context.tables:
        raise ValueError("A non-empty ContextPackage is required")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be non-empty text")
    if not isinstance(base_url, str) or not base_url.strip():
        raise ValueError("base_url must be non-empty text")
    for name, value in (
        ("num_context", num_context),
        ("max_output_tokens", max_output_tokens),
        ("timeout_seconds", timeout_seconds),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")

    model = model.strip()
    base_url = base_url.strip()
    instructions = load_answer_instructions(prompt_path)
    schema = AnswerPayload.model_json_schema()
    schema_text = json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    messages = [
        {
            "role": "system",
            "content": (
                f"{instructions}\n\n"
                "Return JSON matching this exact schema:\n"
                f"{schema_text}"
            ),
        },
        {"role": "user", "content": context.context_text},
    ]

    if client is None:
        response = _ollama_chat(
            base_url=base_url,
            model=model,
            messages=messages,
            response_schema=schema,
            num_context=num_context,
            max_output_tokens=max_output_tokens,
            timeout_seconds=timeout_seconds,
        )
    else:
        response = client.chat(
            model=model,
            messages=messages,
            stream=False,
            format=schema,
            think=False,
            options={
                "temperature": 0,
                "num_ctx": num_context,
                "num_predict": max_output_tokens,
            },
        )

    if isinstance(response, dict) and response.get("done_reason") == "length":
        raise ValueError("Model output was truncated. Increase --max-output-tokens.")

    try:
        if isinstance(response, dict):
            response_content = response["message"]["content"]
            response_id = response.get("created_at")
        else:
            response_content = response.message.content
            response_id = getattr(response, "created_at", None)
        payload = AnswerPayload.model_validate_json(response_content)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "The local answer model did not return the required structured output"
        ) from exc

    # Render declared citations when the model omits inline markers entirely.
    # Existing inline markers still must match; unknown IDs are always rejected.
    if payload.cited_table_ids and not _TABLE_CITATION.search(payload.answer):
        markers = " ".join(f"[{table_id}]" for table_id in payload.cited_table_ids)
        payload = payload.model_copy(update={"answer": f"{payload.answer.rstrip()} {markers}"})
    citations = validate_answer_payload(payload, context.table_ids)
    return GeneratedAnswer(
        status=payload.status,
        text=payload.answer.strip(),
        model=model,
        cited_table_ids=citations,
        calculation_or_evidence=payload.calculation_or_evidence.strip(),
        response_id=str(response_id) if response_id is not None else None,
        prompt_version=PROMPT_VERSION,
    )
