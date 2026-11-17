# AI Finance Assistant: frontend

React + TypeScript (Vite) frontend for the AI Finance Assistant. It keeps the UI thin: no finance logic lives here. All answers come from the FastAPI + LangGraph backend.

Tabs: **Chat** (done), **Portfolio**, **Market**, **Goals** (placeholders, built next).

## Run it

```bash
npm install
npm run dev        # http://localhost:5173
```

The dev server proxies `/api/*` to FastAPI at `http://localhost:8000`, so no CORS setup is needed. To point it elsewhere, set `API_PROXY_TARGET` in a `.env.local` file (see `.env.example`).

In production, either serve `dist/` from FastAPI on the same origin, or set `VITE_API_BASE_URL` at build time and allow that origin in the backend's CORS list.

## Layout

```
src/
├── api/client.ts          # typed fetch wrapper: base URL, timeouts, friendly 429/503/network errors
├── types/api.ts           # mirrors of the backend Pydantic models + SSE event contract
├── hooks/useSession.ts    # session UUID kept in localStorage (the LangGraph thread_id)
├── hooks/useChatStream.ts # POST /api/chat/stream, reads SSE with a ReadableStream reader
├── chat/                  # ChatTab UI
└── App.tsx                # tab shell
```

## Backend contract used by the Chat tab

`POST /api/chat/stream` with `{ "session_id": "<uuid>", "message": "1-2000 chars" }`, answered as server-sent events. Every `data:` line is JSON:

| event     | data                                                        |
|-----------|-------------------------------------------------------------|
| `agent`   | `"finance_qa"` or `["finance_qa", "market"]`               |
| `token`   | `"partial text"` (markdown is rendered)                    |
| `sources` | `[{ title, url, category, score }]`                         |
| `done`    | `{}` or `{ sources, agent_used, disclaimer }` (required: the stream counts as cut off without it) |
| `error`   | `"message"` or `{ error_code, message }`                   |

`DELETE /api/sessions/{id}` is called (best effort) when the user starts a new chat, and the client switches to a new session ID.

Backend notes: do not wrap the streaming route in `GZipMiddleware` (it buffers SSE), and only forward tokens from the answering agent's LLM call, not from the router's classifier.
