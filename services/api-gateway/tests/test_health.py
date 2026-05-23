"""Tests for /health and /ready endpoints."""

from unittest.mock import AsyncMock, MagicMock

from fakeredis.aioredis import FakeRedis
from httpx import AsyncClient
from moiraweave_shared.control_plane import InMemoryControlPlaneRepository
from pytest_mock import MockerFixture


async def test_health_returns_ok(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["uptime_seconds"] >= 0


async def test_ready_all_ok(client: AsyncClient, mock_qdrant: MagicMock) -> None:
    response = await client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["redis"]["status"] == "ok"
    assert body["checks"]["postgres"]["status"] == "ok"
    assert body["checks"]["qdrant"]["status"] == "ok"


async def test_ready_redis_degraded(
    client: AsyncClient, fake_redis: FakeRedis, mocker: MockerFixture
) -> None:
    # Given: Redis ping raises ConnectionError
    # ``app.state.redis`` IS ``fake_redis`` — patch ping on the same object.
    mocker.patch.object(
        fake_redis, "ping", AsyncMock(side_effect=ConnectionError("down"))
    )

    # When
    response = await client.get("/ready")

    # Then
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["redis"]["status"] == "error"
    assert "down" in body["checks"]["redis"]["message"]


async def test_ready_qdrant_degraded(
    client: AsyncClient, mock_qdrant: MagicMock
) -> None:
    # Given: Qdrant raises ConnectionError on collections check
    mock_qdrant.get_collections.side_effect = ConnectionError("qdrant-down")

    # When
    response = await client.get("/ready")

    # Then
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["qdrant"]["status"] == "error"


async def test_ready_postgres_degraded(
    client: AsyncClient,
    control_plane: InMemoryControlPlaneRepository,
    mocker: MockerFixture,
) -> None:
    mocker.patch.object(
        control_plane,
        "ping",
        AsyncMock(side_effect=ConnectionError("postgres-down")),
    )

    response = await client.get("/ready")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["postgres"]["status"] == "error"


async def test_ready_latency_ms_present(client: AsyncClient) -> None:
    response = await client.get("/ready")
    body = response.json()
    assert body["checks"]["redis"]["latency_ms"] >= 0
    assert body["checks"]["postgres"]["latency_ms"] >= 0
    assert body["checks"]["qdrant"]["latency_ms"] >= 0
