"""Health-check endpoint — used by Docker, CI/CD, and monitoring systems.

Returns a JSON object indicating whether the server is fully operational.
The status is "ok" only when both the ML model and the DuckDB connection are
available. "degraded" means the server is running but prediction endpoints
will return 503.

HTTP status codes:
    200 OK                    — status "ok": model loaded, DB connected.
    503 Service Unavailable   — status "degraded": model not loaded or DB gone.

Docker healthcheck example::

    HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse


router = APIRouter(tags=["ops"])


@router.get("/health")
def health_check(request: Request) -> JSONResponse:
    """Return the operational status of the API server.

    Args:
        request: FastAPI request object used to access ``app.state``.

    Returns:
        200 with ``{"status": "ok", ...}`` when model and DB are loaded.
        503 with ``{"status": "degraded", ...}`` otherwise.
    """
    model = getattr(request.app.state, "model", None)
    db = getattr(request.app.state, "db", None)
    status = "ok" if (model is not None and db is not None) else "degraded"
    body = {
        "status": status,
        "model_loaded": model is not None,
        "db_connected": db is not None,
    }
    http_status = 200 if status == "ok" else 503
    return JSONResponse(content=body, status_code=http_status)
