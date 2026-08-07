import sys
from pathlib import Path

from fastapi import Header, HTTPException

sys.path.append(str(Path(__file__).resolve().parent.parent / "training"))
from config import API_KEY  # noqa: E402


def verify_api_key(x_api_key: str = Header(default=None)):
    """Fails closed: if API_KEY isn't configured on the server at all,
    every request is rejected rather than silently allowed through."""
    if not API_KEY:
        raise HTTPException(status_code=503, detail="API key auth not configured on server")
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")