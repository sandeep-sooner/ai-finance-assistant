"""Server-sent-event formatting. Kept separate so it can be unit-tested without FastAPI."""
from __future__ import annotations

import json
from typing import Any


def sse(event: str, data: Any) -> str:
    """Format one SSE frame. json.dumps escapes newlines, so a token can never break framing."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
