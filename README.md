# copilot-openrouter-to-ollama-proxy

A fork of [gagin/openrouter-to-ollama-proxy](https://github.com/gagin/openrouter-to-ollama-proxy), modified to work with GitHub Copilot in addition to Ollama-compatible clients.

For full setup details, local model bypass, embedding support, and Open-WebUI integration, see the original repo.

## Usage

Set your OpenRouter API key and start the proxy:

```bash
export OPENROUTER_API_KEY=your-api-key-here
docker-compose up -d
```

The proxy will be available at `http://localhost:11434`.

## Changes in `proxy.js`

The following modifications were made to support GitHub Copilot:

- **Added `/v1/chat/completions` endpoint** — Copilot uses the OpenAI-compatible endpoint directly; the original proxy only exposed `/api/chat` (Ollama format).
- **Strict OpenAI SSE streaming** — When the request comes in on `/v1/chat/completions`, the proxy forwards a proper `text/event-stream` response (`data: {...}\n\n` lines) instead of Ollama NDJSON.
- **Response shape cleaners** (`cleanDelta`, `cleanStreamingChunk`, `cleanMessage`) — Strip non-standard fields from OpenRouter's response so Copilot receives a spec-compliant OpenAI payload.
- **Reasoning injection** — If the model returns a `reasoning` field (e.g. thinking models), its content is appended to `content` so Copilot can display it.
