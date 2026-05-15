"""
Pydantic schemas compatible with the OpenAI Chat Completions API.
References:
  https://platform.openai.com/docs/api-reference/chat/create
  https://platform.openai.com/docs/api-reference/chat/streaming
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str  # accept any role string (system, user, assistant, tool, developer, etc.)
    content: Optional[Union[str, List[Any]]] = None  # content can be str or array (vision)
    name: Optional[str] = None

    class Config:
        extra = "allow"


class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-v4-flash"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[Union[str, List[str]]] = None
    user: Optional[str] = None
    # DeepCode extensions (passed via extra fields or dedicated fields)
    thinking: Optional[bool] = Field(default=None, description="Enable DeepSeek thinking mode")
    search: Optional[bool] = Field(default=None, description="Enable DeepSeek web search")

    class Config:
        extra = "allow"


# ---------------------------------------------------------------------------
# Non-streaming response
# ---------------------------------------------------------------------------

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatChoiceMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None  # DeepSeek-style thinking field


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatChoiceMessage
    finish_reason: Optional[str] = "stop"
    logprobs: None = None

    model_config = {"populate_by_name": True}


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatChoice]
    usage: UsageInfo = Field(default_factory=UsageInfo)
    system_fingerprint: Optional[str] = None

    def model_dump(self, **kwargs) -> Dict[str, Any]:  # type: ignore[override]
        d = super().model_dump(exclude_none=True, **kwargs)
        return d


# ---------------------------------------------------------------------------
# Streaming response (SSE chunks)
# ---------------------------------------------------------------------------

class DeltaContent(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None  # DeepSeek-style thinking field

    def model_dump_json(self, **kwargs) -> str:  # type: ignore[override]
        # Omit None fields to keep chunks lean
        import json
        d: Dict[str, Any] = {}
        if self.role is not None:
            d["role"] = self.role
        if self.content is not None:
            d["content"] = self.content
        if self.reasoning_content is not None:
            d["reasoning_content"] = self.reasoning_content
        return json.dumps(d)


class ChatChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: Optional[str] = None
    logprobs: None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatChunkChoice]
    system_fingerprint: Optional[str] = None

    def model_dump_json(self, **kwargs) -> str:  # type: ignore[override]
        import json
        choices_out = []
        for c in self.choices:
            choice: Dict[str, Any] = {
                "index": c.index,
                "delta": json.loads(c.delta.model_dump_json()),
                "finish_reason": c.finish_reason,
                "logprobs": None,
            }
            choices_out.append(choice)
        obj: Dict[str, Any] = {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": choices_out,
        }
        return json.dumps(obj)


# ---------------------------------------------------------------------------
# Models list
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = 1700000000
    owned_by: str = "deepseek"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelInfo]
