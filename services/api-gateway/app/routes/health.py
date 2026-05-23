import time

from fastapi import APIRouter, Request

from app.models.health import CheckResult, HealthResponse, ReadyResponse

router = APIRouter(tags=["health"])

_START_TIME = time.monotonic()


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
async def health() -> HealthResponse:
    """Return 200 as long as the process is running."""
    return HealthResponse(status="ok", uptime_seconds=time.monotonic() - _START_TIME)


@router.get("/ready", response_model=ReadyResponse, summary="Readiness probe")
async def ready(request: Request) -> ReadyResponse:
    """Check downstream dependencies and return readiness status.

    Returns 200 even when degraded so Kubernetes keeps routing traffic;
    the ``status`` field signals the actual health to consumers.
    """
    checks: dict[str, CheckResult] = {}

    # Redis check
    t0 = time.monotonic()
    try:
        await request.app.state.redis.ping()
        checks["redis"] = CheckResult(
            status="ok",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = CheckResult(
            status="error",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            message=str(exc),
        )

    # Postgres control-plane check
    t0 = time.monotonic()
    try:
        await request.app.state.control_plane.ping()
        checks["postgres"] = CheckResult(
            status="ok",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )
    except Exception as exc:  # noqa: BLE001
        checks["postgres"] = CheckResult(
            status="error",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            message=str(exc),
        )

    # Qdrant check
    t0 = time.monotonic()
    try:
        await request.app.state.qdrant.get_collections()
        checks["qdrant"] = CheckResult(
            status="ok",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )
    except Exception as exc:  # noqa: BLE001
        checks["qdrant"] = CheckResult(
            status="error",
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            message=str(exc),
        )

    all_ok = all(c.status == "ok" for c in checks.values())
    return ReadyResponse(
        status="ready" if all_ok else "not_ready",
        checks=checks,
    )
