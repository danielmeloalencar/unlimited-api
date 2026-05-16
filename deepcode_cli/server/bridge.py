"""
Bridge between the DeepSeekAPI (deepseek4free) and the OpenAI-compatible schema.

Responsibilities:
  - Convert an OpenAI `messages` array into a single DeepSeek prompt string,
    injecting tool definitions and handling tool call / tool result messages.
  - Map OpenAI model names to DeepSeek model types.
  - Drive the DeepSeekAPI.chat_completion() generator and yield OpenAI-formatted
    SSE chunks (or accumulate into a full response for non-streaming requests).
  - Parse <tool_call>…</tool_call> blocks emitted by DeepSeek and convert them
    back to the OpenAI tool_calls format expected by clients.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

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
    ToolCall,
    ToolCallDelta,
    ToolFunction,
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
# messages[] → single prompt string  (with optional tool injection)
# ---------------------------------------------------------------------------

_ROLE_LABELS: Dict[str, str] = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool",
    "function": "Function",
}

_TOOL_START = "<tool_call>"
_TOOL_END   = "</tool_call>"


def _extract_text(content: Any) -> str:
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


def messages_to_prompt(
    messages: List[ChatMessage],
    tools: Optional[List[Any]] = None,
    tool_choice: Optional[Any] = None,
) -> str:
    """
    Flatten the OpenAI messages array into a single prompt string understood by
    DeepSeek, injecting tool definitions and handling multi-turn tool calls.
    """
    if not messages:
        return ""

    system_parts: List[str] = []
    conv_parts: List[str] = []

    # Collect system messages into a dedicated block
    for msg in messages:
        if msg.role == "system":
            text = _extract_text(msg.content).strip()
            if text:
                system_parts.append(text)

    # Build system prompt + optional tool instructions
    system_prompt = "\n\n".join(system_parts)

    if tools:
        formatted_tools = []
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "function":
                fn = t["function"]
                formatted_tools.append({
                    "name": fn.get("name", ""),
                    "description": fn.get("description", ""),
                    "parameters": fn.get("parameters", {}),
                })
            else:
                formatted_tools.append(t)

        tools_json = json.dumps(formatted_tools, ensure_ascii=False, indent=2)
        tool_instructions = (
            "\n\n# TOOLS AVAILABLE\n"
            "You have access to the following tools:\n"
            f"{tools_json}\n\n"
            "To use a tool you MUST output a JSON object wrapped EXACTLY in these tags:\n"
            f"{_TOOL_START}\n"
            '{"name": "tool_name", "arguments": {"param_name": "value"}}\n'
            f"{_TOOL_END}\n\n"
            "RULES:\n"
            "1. You can call multiple tools by outputting multiple "
            f"{_TOOL_START} blocks consecutively.\n"
            "2. Do NOT output any other text after your tool_call blocks. "
            "Wait for the user to provide the tool response.\n"
            "3. The JSON must be valid and accurately follow the tool's parameters.\n"
        )
        system_prompt = (system_prompt + tool_instructions) if system_prompt else tool_instructions

        # Force a specific tool if tool_choice specifies one
        if (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") == "function"
            and tool_choice.get("function")
        ):
            forced = tool_choice["function"]["name"]
            system_prompt += f'\nCRITICAL: You MUST call the tool "{forced}" in this response.\n'

    # Check if the last message is a tool result — if so, add a nudge
    last_msg = messages[-1] if messages else None
    last_is_tool_result = last_msg is not None and last_msg.role in ("tool", "function")
    if last_is_tool_result:
        system_prompt += (
            "\nIMPORTANT: You just received tool results. "
            "Now provide the final answer to the user in plain text. "
            f"Only emit {_TOOL_START} again if another tool is absolutely required.\n"
        )

    # Build conversation turns (skip system messages — already handled above)
    for msg in messages:
        if msg.role == "system":
            continue

        if msg.role == "assistant":
            content = _extract_text(msg.content).strip()
            # Re-serialise any tool_calls back to <tool_call> tags so DeepSeek
            # understands the history during multi-turn exchanges.
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    args = tc.function.arguments if tc.function else "{}"
                    if isinstance(args, str):
                        try:
                            args_obj = json.loads(args)
                        except Exception:
                            args_obj = {}
                    else:
                        args_obj = args
                    tag = (
                        f"\n{_TOOL_START}"
                        f'{{"name": "{tc.function.name}", "arguments": {json.dumps(args_obj)}}}'
                        f"{_TOOL_END}"
                    )
                    content += tag
            if content:
                conv_parts.append(f"Assistant: {content.strip()}")

        elif msg.role in ("tool", "function"):
            tool_name = msg.name or "tool"
            result = _extract_text(msg.content).strip()
            conv_parts.append(f"Tool Response ({tool_name}): {result}")

        elif msg.role == "user":
            text = _extract_text(msg.content).strip()
            if text:
                conv_parts.append(f"User: {text}")

    conversation = "\n\n".join(conv_parts)
    if system_prompt:
        return f"{system_prompt}\n\n{conversation}" if conversation else system_prompt
    return conversation


# ---------------------------------------------------------------------------
# Tool call parsing  (<tool_call>…</tool_call> → OpenAI ToolCall objects)
# ---------------------------------------------------------------------------

def _parse_tool_calls(text: str) -> tuple[List[ToolCall], str]:
    """
    Extract all <tool_call>…</tool_call> blocks from *text*.

    Returns:
        (tool_calls, remaining_text)  where remaining_text is *text* with all
        tool_call blocks (and any whitespace-only text around them) removed.
    """
    tool_calls: List[ToolCall] = []
    remaining = text

    pattern = re.compile(
        re.escape(_TOOL_START) + r"(.*?)" + re.escape(_TOOL_END),
        re.DOTALL,
    )

    for match in pattern.finditer(text):
        raw_json = match.group(1).strip()
        try:
            obj = json.loads(raw_json)
        except json.JSONDecodeError:
            # Try a lenient parse — strip trailing commas, etc.
            cleaned = re.sub(r",\s*([}\]])", r"\1", raw_json)
            try:
                obj = json.loads(cleaned)
            except Exception:
                continue  # skip malformed block

        tool_name = obj.get("name", "")
        args = obj.get("arguments", {})
        if not isinstance(args, dict):
            # Some models wrap args in a string
            try:
                args = json.loads(args)
            except Exception:
                args = {}

        tool_calls.append(
            ToolCall(
                id=f"call_{uuid.uuid4().hex[:24]}",
                type="function",
                function=ToolFunction(
                    name=tool_name,
                    arguments=json.dumps(args, ensure_ascii=False),
                ),
            )
        )

    # Remove all matched blocks from the text
    remaining = pattern.sub("", text).strip()
    return tool_calls, remaining


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
    tools: Optional[List[Any]] = None,
) -> AsyncGenerator[str, None]:
    """
    Async generator that drives the synchronous DeepSeekAPI.chat_completion()
    iterator in a thread executor and yields OpenAI-compatible SSE strings.

    Thinking chunks  → delta.reasoning_content
    Text chunks      → delta.content  (with <tool_call> blocks intercepted and
                       emitted as delta.tool_calls in OpenAI format)
    """
    import asyncio

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    model_type = model_name_to_type(model_name)
    has_tools = bool(tools)

    def _sse(chunk: ChatCompletionChunk) -> str:
        return f"data: {chunk.model_dump_json()}\n\n"

    def _make_chunk(
        content: Optional[str] = None,
        reasoning_content: Optional[str] = None,
        role: Optional[str] = None,
        finish_reason: Optional[str] = None,
        tool_calls: Optional[List[ToolCallDelta]] = None,
    ) -> str:
        delta = DeltaContent(
            role=role,
            content=content,
            reasoning_content=reasoning_content,
            tool_calls=tool_calls,
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

    # Collect all text first so we can detect tool calls across chunk boundaries
    reasoning_parts: List[str] = []
    text_parts: List[str] = []

    for raw in raw_chunks:
        ctype   = raw.get("type")
        content = raw.get("content", "")
        if not content:
            continue
        if ctype == "thinking" and thinking_enabled:
            reasoning_parts.append(content)
        elif ctype == "text":
            text_parts.append(content)

    full_text = "".join(text_parts)

    # Emit reasoning first
    if reasoning_parts:
        yield _make_chunk(reasoning_content="".join(reasoning_parts))

    if has_tools:
        # Parse and emit tool calls or plain text
        tool_calls, remaining_text = _parse_tool_calls(full_text)

        if tool_calls:
            # Emit any text before the first tool call
            if remaining_text:
                yield _make_chunk(content=remaining_text)

            # Emit each tool call as a delta
            for idx, tc in enumerate(tool_calls):
                yield _make_chunk(
                    tool_calls=[
                        ToolCallDelta(
                            index=idx,
                            id=tc.id,
                            type="function",
                            function={"name": tc.function.name, "arguments": tc.function.arguments},
                        )
                    ]
                )

            yield _make_chunk(finish_reason="tool_calls")
        else:
            # No tool calls — emit text normally
            if full_text:
                yield _make_chunk(content=full_text)
            yield _make_chunk(finish_reason="stop")
    else:
        # No tools in this request — stream text as-is
        if full_text:
            yield _make_chunk(content=full_text)
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
    tools: Optional[List[Any]] = None,
) -> ChatCompletionResponse:
    """
    Synchronous wrapper — collects the full stream and returns a single
    ChatCompletionResponse object (with tool_calls when present).
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
        if ctype == "thinking" and content and thinking_enabled:
            thinking_parts.append(content)
        elif ctype == "text" and content:
            text_parts.append(content)

    full_text = "".join(text_parts)
    full_thinking = "".join(thinking_parts) or None

    # Rough token estimate (1 token ≈ 4 chars)
    prompt_tokens = max(1, len(prompt) // 4)
    completion_tokens = max(1, (len(full_text) + len(full_thinking or "")) // 4)

    # Parse tool calls when tools were provided
    tool_calls: Optional[List[ToolCall]] = None
    response_content: Optional[str] = full_text
    finish_reason = "stop"

    if tools:
        parsed_calls, remaining_text = _parse_tool_calls(full_text)
        if parsed_calls:
            tool_calls = parsed_calls
            response_content = remaining_text or None
            finish_reason = "tool_calls"

    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=model_name,
        choices=[
            ChatChoice(
                index=0,
                message=ChatChoiceMessage(
                    role="assistant",
                    content=response_content,
                    reasoning_content=full_thinking,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
            )
        ],
        usage=UsageInfo(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
