from fastapi import APIRouter, Depends, Path, Request, Response

from src.web_app.api.deps import api_rate_limit

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.delete("/{session_id}", status_code=204, dependencies=[Depends(api_rate_limit)])
async def delete_session(
    request: Request,
    session_id: str = Path(min_length=8, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
):
    """Forget a conversation's server-side memory. Idempotent."""
    await request.app.state.checkpointer.adelete_thread(session_id)
    return Response(status_code=204)
