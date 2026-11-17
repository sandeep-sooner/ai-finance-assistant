import json

from src.core.sse import sse


def test_frame_shape():
    assert sse("token", "hi") == 'event: token\ndata: "hi"\n\n'


def test_newlines_in_tokens_cannot_break_framing():
    frame = sse("token", "line1\n\nline2")
    assert frame.count("\n\n") == 1  # only the terminator
    assert json.loads(frame.split("data: ")[1]) == "line1\n\nline2"
