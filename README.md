# DeepCode

**DeepCode** is an autonomous AI coding agent powered by DeepSeek-V4. It operates directly inside your coding workspace, capable of reading, writing, editing files, running commands, and completing full coding tasks. It features a beautiful terminal UI inspired by Claude Code.

It also ships with a built-in **OpenAI-compatible API server** that exposes DeepSeek through a standard REST interface — drop-in compatible with any tool that supports the OpenAI API (Cursor, Continue, OpenWebUI, LiteLLM, etc.).

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Getting your DeepSeek token](#getting-your-deepseek-token)
- [CLI Usage](#cli-usage)
- [OpenAI-Compatible API Server](#openai-compatible-api-server)
  - [Starting the server](#starting-the-server)
  - [Authentication](#authentication)
  - [Endpoints](#endpoints)
  - [Models](#models)
  - [Request reference](#request-reference)
  - [Response reference](#response-reference)
  - [Thinking (reasoning)](#thinking-reasoning)
  - [Web search](#web-search)
  - [Streaming](#streaming)
  - [Examples](#examples)
  - [Connecting clients](#connecting-clients)
  - [Docker & Coolify deploy](#docker--coolify-deploy)
- [Credits](#credits)
- [License](#license)

---

## Features

- **Autonomous Coding** — give it a task and it iteratively reads, writes and runs commands until done.
- **DeepSeek-V4 Flash & Pro** — switch between instant and expert reasoning models.
- **Extended Thinking** — expose DeepSeek's internal reasoning as `reasoning_content`, fully compatible with OpenAI's reasoning format.
- **Web Search** — built-in DeepSeek web search, toggleable per request.
- **OpenAI-compatible API** — full `/v1/chat/completions` support (streaming + non-streaming).
- **Beautiful Terminal UI** — real-time streaming, Markdown rendering, Rich-powered layout.
- **Cross-Platform** — Windows, macOS, Linux.

---

## Installation

```bash
git clone <your-repo-url>
cd deepcode
pip install -e .
```

This installs two commands:

| Command | Description |
|---|---|
| `deepcode` | Interactive CLI agent |
| `deepcode-server` | OpenAI-compatible API server |

---

## Getting your DeepSeek token

1. Go to [chat.deepseek.com](https://chat.deepseek.com) and log in.
2. Open Developer Tools (F12) → **Console** tab.
3. Run:
   ```javascript
   JSON.parse(localStorage.getItem("userToken")).value
   ```
4. Copy the returned string (no quotes).

The token is saved automatically to `~/.deepcode/config.json` (or `%APPDATA%\DeepCode\config.json` on Windows) on first use — you only need to provide it once.

---

## CLI Usage

```bash
# Start in current directory
deepcode

# Start in a specific directory
deepcode -d /path/to/project

# One-shot non-interactive mode
deepcode -p "fix the bug in main.py"

# Allow access outside workspace
deepcode -f

# Skip confirmations for risky commands
deepcode -a

# Control thinking mode
deepcode --thinking
deepcode --no-thinking
```

### Slash commands

| Command | Description |
|---|---|
| `/help` | Show all commands |
| `/clear` | Clear screen, start new conversation |
| `/think` | Toggle extended thinking on/off |
| `/search` | Toggle web search on/off |
| `/model flash\|pro` | Switch model |
| `/workspace [path]` | View or change working directory |
| `/session` | Show session info |
| `/session del` | Reset current session |
| `/config` | Open config folder |
| `/exit` | Exit DeepCode |

---

## OpenAI-Compatible API Server

### Starting the server

```bash
# Uses token from config or DEEPSEEK_AUTH_TOKEN env var
deepcode-server

# Pin a token explicitly (also saves it to config)
deepcode-server --token YOUR_DEEPSEEK_TOKEN

# Custom host and port
deepcode-server --host 127.0.0.1 --port 9000

# All options
deepcode-server --help
```

```
options:
  --host        Bind host (default: 0.0.0.0)
  --port        Bind port (default: 8000)
  --token       DeepSeek auth token
  --reload      Enable auto-reload (dev mode)
  --log-level   debug | info | warning | error
```

---

### Authentication

The server resolves the DeepSeek auth token in this order per request:

| Priority | Source |
|---|---|
| 1 | `--token` CLI argument (pinned at startup) |
| 2 | `DEEPSEEK_AUTH_TOKEN` environment variable |
| 3 | `~/.deepcode/config.json` saved token |
| 4 | `Authorization: Bearer <TOKEN>` request header |

When the token arrives via the `Authorization` header and no token is saved yet, it is **automatically persisted** to the config file — future requests work without the header.

Clients connecting to the server may pass any string as `api_key` (it is accepted but ignored — the server uses its own DeepSeek token).

If no token is found from any source, the server returns `401` with instructions.

---

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe |
| `GET` | `/v1/models` | List available models |
| `GET` | `/models` | Same (no `/v1` prefix) |
| `POST` | `/v1/chat/completions` | Chat completion |
| `POST` | `/chat/completions` | Same (no `/v1` prefix) |
| `GET` | `/docs` | Swagger UI |

---

### Models

Pass the model name in the `"model"` field of the request body.

| `"model"` value | Backend | Description |
|---|---|---|
| `deepseek-v4-flash` | `default` | Fast, efficient — good for most tasks |
| `deepseek-chat` | `default` | Alias for flash |
| `default` / `flash` | `default` | Aliases for flash |
| `deepseek-v4-pro` | `expert` | Slower, more powerful reasoning |
| `deepseek-r1` | `expert` | Alias for pro |
| `expert` / `pro` | `expert` | Aliases for pro |
| any other string | `default` | Falls back to flash |

---

### Request reference

`POST /v1/chat/completions`

```json
{
  "model": "deepseek-v4-flash",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the capital of France?"}
  ],
  "stream": false,

  "thinking": true,
  "search": false,

  "temperature": 0.7,
  "max_tokens": 1024,
  "top_p": 1.0,
  "stop": null,
  "user": "user-123"
}
```

#### Standard OpenAI fields

| Field | Type | Default | Description |
|---|---|---|---|
| `model` | string | `deepseek-v4-flash` | Model name (see [Models](#models)) |
| `messages` | array | required | Array of `{role, content}` objects |
| `stream` | bool | `false` | Enable SSE streaming |
| `temperature` | float | — | Accepted but not forwarded to DeepSeek |
| `max_tokens` | int | — | Accepted but not forwarded to DeepSeek |
| `top_p` | float | — | Accepted but not forwarded to DeepSeek |
| `stop` | string\|array | — | Accepted but not forwarded to DeepSeek |
| `user` | string | — | Accepted but not forwarded to DeepSeek |

#### DeepCode extensions

| Field | Type | Default | Description |
|---|---|---|---|
| `thinking` | bool | `true` | Enable DeepSeek extended thinking. When `true`, reasoning is returned in `reasoning_content`. |
| `search` | bool | `false` | Enable DeepSeek web search for up-to-date information. |

These can also be set via request headers:

| Header | Values | Description |
|---|---|---|
| `X-Thinking` | `true` / `false` | Override `thinking` (takes priority over body) |
| `X-Search` | `true` / `false` | Override `search` (takes priority over body) |

#### Message roles

All standard OpenAI roles are accepted: `system`, `user`, `assistant`, `tool`, `function`, `developer`. The entire conversation history is formatted and sent as a single prompt to DeepSeek.

#### Content formats

Both string and array content are supported:

```json
{"role": "user", "content": "Hello"}

{"role": "user", "content": [{"type": "text", "text": "Hello"}]}
```

---

### Response reference

#### Non-streaming response

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "deepseek-v4-pro",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The capital of France is Paris.",
        "reasoning_content": "The user is asking a geography question..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 8,
    "total_tokens": 28
  }
}
```

- `reasoning_content` — only present when `thinking: true`. Contains the full internal reasoning text. Omitted entirely when `null`.

#### Streaming response (SSE)

Each chunk follows the OpenAI streaming format. There are two types of content deltas:

**Thinking chunk** (`reasoning_content` field):
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-r1","choices":[{"index":0,"delta":{"reasoning_content":"Let me think..."},"finish_reason":null,"logprobs":null}]}
```

**Text chunk** (`content` field):
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-r1","choices":[{"index":0,"delta":{"content":"Paris."},"finish_reason":null,"logprobs":null}]}
```

**Stop chunk:**
```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1700000000,"model":"deepseek-r1","choices":[{"index":0,"delta":{},"finish_reason":"stop","logprobs":null}]}

data: [DONE]
```

`reasoning_content` and `content` are always in separate chunks — never mixed. When `thinking: false`, no `reasoning_content` chunks are emitted.

---

### Thinking (reasoning)

DeepSeek's internal reasoning is exposed via the `reasoning_content` field, following the same convention as the official DeepSeek API and compatible with OpenWebUI, Continue, and other clients that render reasoning.

| `thinking` | Non-streaming | Streaming |
|---|---|---|
| `true` (default) | `message.reasoning_content` populated | Chunks with `delta.reasoning_content` emitted before `delta.content` |
| `false` | `reasoning_content` absent | No `reasoning_content` chunks emitted |

---

### Web search

When `search: true`, DeepSeek performs a live web search to ground the response in up-to-date information. Useful for news, current events, prices, etc.

```json
{
  "model": "deepseek-v4-flash",
  "messages": [{"role": "user", "content": "What is the current Bitcoin price?"}],
  "search": true,
  "thinking": false
}
```

---

### Streaming

Enable streaming by setting `"stream": true`. The response is a standard SSE stream terminated by `data: [DONE]`.

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Count to 5"}],"stream":true,"thinking":false}'
```

---

### Examples

#### Simple question (non-streaming)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "stream": false,
    "thinking": false
  }'
```

#### With system prompt and conversation history

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "system",    "content": "You are a concise assistant. Reply in Portuguese."},
      {"role": "user",      "content": "What is the capital of Brazil?"},
      {"role": "assistant", "content": "Brasilia."},
      {"role": "user",      "content": "And of Argentina?"}
    ],
    "stream": false,
    "thinking": false
  }'
```

#### Pro model with thinking enabled

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "Solve: if 3x + 7 = 22, what is x?"}],
    "stream": false,
    "thinking": true
  }'
```

Response will include:
```json
"message": {
  "role": "assistant",
  "content": "x = 5",
  "reasoning_content": "3x + 7 = 22 → 3x = 15 → x = 5"
}
```

#### Web search for current information

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "What are the top tech news today?"}],
    "stream": false,
    "thinking": false,
    "search": true
  }'
```

#### Streaming with thinking (pro model)

```bash
curl -N -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "Write a haiku about the ocean."}],
    "stream": true,
    "thinking": true
  }'
```

#### Token via Authorization header (first-time setup)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_DEEPSEEK_TOKEN" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

The token is saved automatically — the header is not required on subsequent requests.

#### Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","time":1700000000}
```

#### List models

```bash
curl http://localhost:8000/v1/models
```

```json
{
  "object": "list",
  "data": [
    {"id": "deepseek-v4-flash", "object": "model", "created": 1700000000, "owned_by": "deepseek"},
    {"id": "deepseek-v4-pro",   "object": "model", "created": 1700000000, "owned_by": "deepseek"},
    {"id": "deepseek-chat",     "object": "model", "created": 1700000000, "owned_by": "deepseek"},
    {"id": "deepseek-r1",       "object": "model", "created": 1700000000, "owned_by": "deepseek"}
  ]
}
```

---

### Connecting clients

Set the `base_url` to your server and any non-empty string as `api_key`:

#### Python (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="any-string",
)

response = client.chat.completions.create(
    model="deepseek-v4-flash",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={"thinking": False, "search": False},
)
print(response.choices[0].message.content)
```

#### Streaming with reasoning

```python
stream = client.chat.completions.create(
    model="deepseek-r1",
    messages=[{"role": "user", "content": "Explain quantum entanglement."}],
    stream=True,
    extra_body={"thinking": True},
)
for chunk in stream:
    delta = chunk.choices[0].delta
    if hasattr(delta, "reasoning_content") and delta.reasoning_content:
        print("[think]", delta.reasoning_content, end="", flush=True)
    elif delta.content:
        print(delta.content, end="", flush=True)
```

#### Cursor / VS Code (Continue)

```json
{
  "baseUrl": "http://localhost:8000/v1",
  "apiKey": "any-string",
  "model": "deepseek-r1"
}
```

#### OpenWebUI

Set provider to **OpenAI**, base URL to `http://localhost:8000/v1`, API key to any non-empty string.

---

### Docker & Coolify deploy

Build and run locally:

```bash
docker build -t deepcode-server .
docker run -p 8000:8000 -e DEEPSEEK_AUTH_TOKEN=your_token deepcode-server
```

#### Coolify configuration

| Setting | Value |
|---|---|
| Build Pack | `Dockerfile` |
| Port | `8000` |
| Health Check Path | `/health` |

**Environment variables:**

| Variable | Required | Description |
|---|---|---|
| `DEEPSEEK_AUTH_TOKEN` | No* | DeepSeek auth token. If omitted, clients must send `Authorization: Bearer <TOKEN>` on the first request — it will be saved automatically. |
| `PORT` | No | Override listen port (default: `8000`) |
| `HOST` | No | Override bind host (default: `0.0.0.0`) |

**Persistent volume (recommended):**

Mount `/app/data` to a persistent volume so the saved token and sessions survive redeploys:

```
/app/data  →  deepcode_data (volume)
```

After deploy, your `base_url` is:

```
https://your-coolify-domain.com/v1
```

---

## Credits

Special thanks to [xtekky/deepseek4free](https://github.com/xtekky/deepseek4free) for the original `deepseek4free` library. It has been modified and bundled into this project to provide the autonomous agent and API server capabilities.

---

## License

MIT License. See `LICENSE` for more information.
