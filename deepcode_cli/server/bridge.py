"""
Bridge between the DeepSeekAPI (deepseek4free) and the OpenAI-compatible schema.

Responsibilities:
  - Convert an OpenAI `messages` array into a single DeepSeek prompt string.
  - Map OpenAI model names to DeepSeek model types.
  - Drive the DeepSeekAPI.chat_completion() generator and yield OpenAI-formatted
    SSE chunks (or accumulate into a full response for non-streaming requests).
  - Emit <think>...</think> tags around reasoning chunks so clients that
    understand the tag (OpenWebUI, Continue, etc.) can render them.
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional

# ---------------------------------------------------------------------------
# DeepSeek imports — resolve deepseek4free from the CLI package location
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_DSK_PATH = os.path.join(_HERE, "..", "deepseek4free")
if _DSK_PATH not in sys.path:
    sys.path.insert(0, _DSK_PATH)

from dsk.api import DeepSeekAPI, MODEL_FLASH, MODEL_PRO  # noqa: E402

from .models import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    ChatChunkChoice,
    ChatChoice,
    ChatChoiceMessage,
    ChatMessage,
    DeltaContent,
    UsageInfo,
)

# ---------------------------------------------------------------------------
# Model name → DeepSeek model type
# ---------------------------------------------------------------------------

_FLASH_ALIASES = {
    "deepseek-v4-flash",
    "deepseek-chat",
    "deepseek-v3",
    "deepseek",
    "default",
    "flash",
}

_PRO_ALIASES = {
    "deepseek-v4-pro",
    "deepseek-r1",
    "deepseek-r1-pro",
    "expert",
    "pro",
    "o1",        # common alias used in some UIs
    "o1-mini",
}


def model_name_to_type(model: str) -> str:
    """Return MODEL_FLASH or MODEL_PRO based on the requested model name."""
    lower = model.lower().strip()
    if lower in _PRO_ALIASES:
        return MODEL_PRO
    # Default to flash for anything unknown
    return MODEL_FLASH


# ---------------------------------------------------------------------------
# messages[] → single prompt string
# ---------------------------------------------------------------------------

_ROLE_LABELS: Dict[str, str] = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
    "function": "Function",
}


def _extract_text(content) -> str:
    """Extract plain text from content that may be a string or an array of parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # OpenAI vision format: list of {"type": "text", "text": "..."} | {"type": "image_url", ...}
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
            elif isinstance(part, str):
                parts.append(part)
        return "".join(parts)
    return str(content)


def messages_to_prompt(messages: List[ChatMessage]) -> str:
    """
    Flatten the OpenAI messages array into a single prompt string.

    Format:
        System: <system content>

        User: <user content>

        Assistant: <assistant content>

        User: <last user message>

    If there is only one message it is returned as-is (no role prefix),
    which keeps single-turn requests clean.
    """
    if not messages:
        return ""

    if len(messages) == 1:
        return _extract_text(messages[0].content)

    parts: List[str] = []
    for msg in messages:
        label = _ROLE_LABELS.get(msg.role, msg.role.capitalize())
        content = _extract_text(msg.content).strip()
        if content:
            parts.append(f"{label}: {content}")

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Streaming bridge
# ---------------------------------------------------------------------------

async def stream_completion(
    api: DeepSeekAPI,
    chat_session_id: str,
    prompt: str,
    parent_message_id: Optional[str],
    model_name: str,
    thinking_enabled: bool,
    search_enabled: bool,
) -> AsyncGenerator[str, None]:
    """
    Async generator that drives the synchronous DeepSeekAPI.chat_completion()
    iterator in a thread executor and yields OpenAI-compatible SSE strings.

    Thinking chunks  → delta.reasoning_content  (separate field, NOT in content)
    Text chunks      → delta.content
    """
    import asyncio

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_type = model_name_to_type(model_name)

    def _sse(chunk: ChatCompletionChunk) -> str:
        return f"data: {chunk.model_dump_json()}\n\n"

    def _make_chunk(
        content: Optional[str] = None,
        reasoning_content: Optional[str] = None,
        role: Optional[str] = None,
        finish_reason: Optional[str] = None,
    ) -> str:
        delta = DeltaContent(
            role=role,
            content=content,
            reasoning_content=reasoning_content,
        )
        chunk = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model_name,
            choices=[ChatChunkChoice(index=0, delta=delta, finish_reason=finish_reason)],
        )
        return _sse(chunk)

    loop = asyncio.get_event_loop()

    def _run_sync():
        chunks = []
        gen = api.chat_completion(
            chat_session_id,
            prompt,
            parent_message_id=parent_message_id,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
            model_type=model_type,
        )
        for chunk in gen:
            chunks.append(chunk)
        return chunks

    # First chunk carries the role
    yield _make_chunk(role="assistant", content="")

    try:
        raw_chunks = await loop.run_in_executor(None, _run_sync)
    except Exception as e:
        error_payload = (
            'data: {"error": {"message": '
            + '"' + str(e).replace('"', "'") + '", "type": "api_error"'
            + "}}\n\n"
        )
        yield error_payload
        yield "data: [DONE]\n\n"
        return

    for raw in raw_chunks:
        ctype   = raw.get("type")
        content = raw.get("content", "")
        if not content:
            continue

        if ctype == "thinking":
            # Emit as reasoning_content delta — kept separate from content
            yield _make_chunk(reasoning_content=content)
        elif ctype == "text":
            # Emit as normal content delta
            yield _make_chunk(content=content)
        # "status" chunks carry message_id — not forwarded

    # Stop chunk
    yield _make_chunk(finish_reason="stop")
    yield "data: [DONE]\n\n"


# ---------------------------------------------------------------------------
# Non-streaming bridge
# ---------------------------------------------------------------------------

def collect_completion(
    api: DeepSeekAPI,
    chat_session_id: str,
    prompt: str,
    parent_message_id: Optional[str],
    model_name: str,
    thinking_enabled: bool,
    search_enabled: bool,
) -> ChatCompletionResponse:
    """
    Synchronous wrapper — collects the full stream and returns a single
    ChatCompletionResponse object.
    """
    model_type = model_name_to_type(model_name)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    thinking_parts: List[str] = []
    text_parts: List[str] = []

    gen = api.chat_completion(
        chat_session_id,
        prompt,
        parent_message_id=parent_message_id,
        thinking_enabled=thinking_enabled,
        search_enabled=search_enabled,
        model_type=model_type,
    )

    for chunk in gen:
        ctype = chunk.get("type")
        content = chunk.get("content", "")
        if ctype == "thinking" and content:
            thinking_parts.append(content)
        elif ctype == "text" and content:
            text_parts.append(content)

    full_text = "".join(text_parts)
    full_thinking = "".join(thinking_parts) or None

    # Rough token estimate (1 token ≈ 4 chars)
    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = max(1, (len(full_text) + len(full_thinking or "")) // 4)

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[
            ChatChoice(
                index=0,
                message=ChatChoiceMessage(
                    role="assistant",
                    content=full_text,
                    reasoning_content=full_thinking,
                ),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
