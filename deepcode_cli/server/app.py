"""
DeepCode — OpenAI-compatible API server.

Usage:
    deepcode-server [--host HOST] [--port PORT] [--token TOKEN]
    python -m deepcode_cli.server.app

Endpoints:
    GET  /health                  — liveness probe
    GET  /v1/models               — list available models
    POST /v1/chat/completions     — chat completion (stream & non-stream)

Authentication:
    The server resolves the DeepSeek auth token in this order per-request:

      1. CLI argument --token  (set once at startup, used for every request)
      2. Environment variable  DEEPSEEK_AUTH_TOKEN
      3. DeepCode config file  (~/.deepcode/config.json)
      4. Authorization header  Bearer <TOKEN>  sent by the client

    When the token arrives via the Authorization header and no token is saved
    yet, it is automatically persisted to the config file so future requests
    (and the interactive CLI) can reuse it without re-sending the header.

    The server starts even with no token configured — clients just need to
    include the header on every request until a token is saved.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Resolve package paths before any local import
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_CLI_ROOT = os.path.join(_HERE, "..")
_DSK_PATH = os.path.join(_CLI_ROOT, "deepseek4free")
for _p in (_DSK_PATH, _CLI_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from dsk.api import DeepSeekAPI

from deepcode_cli.src.config_manager import ConfigManager
from .bridge import collect_completion, messages_to_prompt, stream_completion
from .models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ModelInfo,
    ModelList,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("deepcode.server")

# ---------------------------------------------------------------------------
# Available models advertised to clients
# ---------------------------------------------------------------------------
AVAILABLE_MODELS = [
    ModelInfo(id="deepseek-v4-flash", owned_by="deepseek"),
    ModelInfo(id="deepseek-v4-pro",   owned_by="deepseek"),
    ModelInfo(id="deepseek-chat",     owned_by="deepseek"),
    ModelInfo(id="deepseek-r1",       owned_by="deepseek"),
]

# ---------------------------------------------------------------------------
# Token / API instance state
# ---------------------------------------------------------------------------

# Optional token pinned at startup via --token or env var.
# When set, it takes priority over any Authorization header.
_pinned_token: Optional[str] = None

# Per-token DeepSeekAPI instance cache (avoids recreating for every request)
_api_cache: Dict[str, DeepSeekAPI] = {}

_config: ConfigManager = ConfigManager()


def _get_or_create_api(token: str) -> DeepSeekAPI:
    """Return a cached DeepSeekAPI for the given token, creating one if needed."""
    if token not in _api_cache:
        _api_cache[token] = DeepSeekAPI(token)
    return _api_cache[token]


def _resolve_token_for_request(request: Request) -> str:
    """
    Resolve the DeepSeek auth token for an incoming request.

    Priority:
      1. Pinned token (--token CLI arg or DEEPSEEK_AUTH_TOKEN env var at startup)
      2. Token saved in config file
      3. Authorization: Bearer <token> header from the client

    If a token arrives via the header and none is saved yet, it is persisted
    to the config file automatically.

    Raises HTTPException(401) if no token can be found from any source.
    """
    # 1. Pinned at startup
    if _pinned_token:
        return _pinned_token

    # 2. Config file
    saved = _config.get("deepseek_auth_token", "").strip()
    if saved:
        return saved

    # 3. Authorization header
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        header_token = auth_header[7:].strip()
        if header_token:
            # Persist so the next request (and the CLI) can reuse it
            _config.set("deepseek_auth_token", header_token)
            log.info("Token received via Authorization header and saved to config.")
            return header_token

    raise HTTPException(
        status_code=401,
        detail=(
            "DeepSeek auth token not configured. "
            "Pass it via:  Authorization: Bearer <YOUR_DEEPSEEK_TOKEN>  "
            "or start the server with --token <TOKEN>. "
            "To get your token: open chat.deepseek.com DevTools Console and run: "
            "JSON.parse(localStorage.getItem('userToken')).value"
        ),
    )


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("DeepCode API server ready.")
    yield
    log.info("DeepCode API server shutting down.")


app = FastAPI(
    title="DeepCode OpenAI-compatible API",
    version="1.0.0",
    description="OpenAI-compatible proxy backed by DeepSeek via deepseek4free.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Log the raw body and validation errors to help diagnose 422s from clients."""
    try:
        body = await request.body()
        log.warning("422 Validation error — raw body: %s", body.decode("utf-8", errors="replace")[:2000])
    except Exception:
        pass
    log.warning("422 Validation errors: %s", exc.errors())
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body)},
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "time": int(time.time())}


@app.get("/v1/models")
async def list_models():
    return ModelList(data=AVAILABLE_MODELS)


# Also expose without /v1 prefix for compatibility with some clients
@app.get("/models")
async def list_models_no_prefix():
    return ModelList(data=AVAILABLE_MODELS)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    token = _resolve_token_for_request(request)
    api   = _get_or_create_api(token)

    # --- Validate we have messages ---
    if not body.messages:
        raise HTTPException(status_code=400, detail="messages array cannot be empty")

    # --- Convert messages to prompt ---
    prompt = messages_to_prompt(body.messages)
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="All messages have empty content")

    # --- Resolve thinking / search flags ---
    # Client can pass {"thinking": true} or {"search": true} in the request body
    # as DeepCode-specific extensions.
    thinking_enabled: bool = True
    search_enabled: bool = False

    if body.thinking is not None:
        thinking_enabled = body.thinking
    if body.search is not None:
        search_enabled = body.search

    # Also honour X-Thinking / X-Search headers
    if request.headers.get("x-thinking", "").lower() in ("false", "0", "no"):
        thinking_enabled = False
    if request.headers.get("x-thinking", "").lower() in ("true", "1", "yes"):
        thinking_enabled = True
    if request.headers.get("x-search", "").lower() in ("true", "1", "yes"):
        search_enabled = True

    # --- Create a fresh DeepSeek chat session for this request ---
    try:
        log.info(
            "Request  model=%s  stream=%s  thinking=%s  search=%s",
            body.model, body.stream, thinking_enabled, search_enabled,
        )
        chat_session_id: str = await asyncio.get_event_loop().run_in_executor(
            None, api.create_chat_session
        )
    except Exception as exc:
        log.error("Failed to create DeepSeek session: %s", exc)
        raise HTTPException(status_code=502, detail=f"DeepSeek session error: {exc}")

    parent_message_id: Optional[str] = None

    # --- Streaming response ---
    if body.stream:
        async def _event_stream():
            try:
                async for chunk in stream_completion(
                    api=api,
                    chat_session_id=chat_session_id,
                    prompt=prompt,
                    parent_message_id=parent_message_id,
                    model_name=body.model,
                    thinking_enabled=thinking_enabled,
                    search_enabled=search_enabled,
                ):
                    yield chunk
            finally:
                # Best-effort session cleanup
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None, api.delete_chat_session, chat_session_id
                    )
                except Exception:
                    pass

        return StreamingResponse(
            _event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    # --- Non-streaming response ---
    try:
        loop = asyncio.get_event_loop()
        response: ChatCompletionResponse = await loop.run_in_executor(
            None,
            lambda: collect_completion(
                api=api,
                chat_session_id=chat_session_id,
                prompt=prompt,
                parent_message_id=parent_message_id,
                model_name=body.model,
                thinking_enabled=thinking_enabled,
                search_enabled=search_enabled,
            ),
        )
    except Exception as exc:
        log.error("Completion error: %s", exc)
        raise HTTPException(status_code=502, detail=f"DeepSeek completion error: {exc}")
    finally:
        # Clean up session regardless of outcome
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, api.delete_chat_session, chat_session_id
            )
        except Exception:
            pass

    return JSONResponse(content=response.model_dump())


# Alias without /v1 prefix
@app.post("/chat/completions")
async def chat_completions_no_prefix(request: Request, body: ChatCompletionRequest):
    return await chat_completions(request, body)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    """CLI entry point: deepcode-server"""
    import uvicorn

    parser = argparse.ArgumentParser(
        prog="deepcode-server",
        description="DeepCode — OpenAI-compatible API server backed by DeepSeek",
    )
    parser.add_argument("--host",  default="0.0.0.0",    help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port",  default=8000, type=int, help="Bind port (default: 8000)")
    parser.add_argument("--token", default="",           help="DeepSeek auth token (pins for all requests)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Uvicorn log level",
    )
    args = parser.parse_args()

    global _pinned_token

    # Priority 1: explicit --token arg
    if args.token.strip():
        _pinned_token = args.token.strip()
        _config.set("deepseek_auth_token", _pinned_token)
        log.info("Token pinned from --token argument and saved to config.")

    # Priority 2: env var
    elif os.environ.get("DEEPSEEK_AUTH_TOKEN", "").strip():
        _pinned_token = os.environ["DEEPSEEK_AUTH_TOKEN"].strip()
        log.info("Token pinned from DEEPSEEK_AUTH_TOKEN environment variable.")

    # Priority 3: saved config
    elif _config.get("deepseek_auth_token", "").strip():
        _pinned_token = _config.get("deepseek_auth_token").strip()
        log.info("Token loaded from config file.")

    # No token — server starts anyway, waits for Authorization header
    else:
        log.warning(
            "No DeepSeek token configured. "
            "Clients must send:  Authorization: Bearer <YOUR_DEEPSEEK_TOKEN>\n"
            "  The token will be saved automatically on first use.\n"
            "  Get your token at chat.deepseek.com → DevTools Console:\n"
            "    JSON.parse(localStorage.getItem('userToken')).value"
        )

    print(
        f"\n  DeepCode OpenAI-compatible API\n"
        f"  ─────────────────────────────\n"
        f"  Listening on  http://{args.host}:{args.port}\n"
        f"  Docs          http://localhost:{args.port}/docs\n"
        f"  Models        http://localhost:{args.port}/v1/models\n\n"
        f"  OpenAI base_url → http://localhost:{args.port}/v1\n"
        + (f"  Token         → configured ✓\n" if _pinned_token else
           f"  Token         → not set (send via Authorization: Bearer <TOKEN>)\n")
    )

    uvicorn.run(
        "deepcode_cli.server.app:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        reload=args.reload,
    )


if __name__ == "__main__":
    run()
