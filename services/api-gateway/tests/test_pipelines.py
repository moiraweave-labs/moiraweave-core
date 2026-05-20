"""Tests for GET /v1/pipelines and POST /v1/pipelines/{id}/jobs."""

import json
from unittest.mock import patch

import pytest
from fakeredis.aioredis import FakeRedis
from httpx import AsyncClient
from moiraweave_shared.pipeline import PipelineDefinition


def _audio_rag_pipeline() -> PipelineDefinition:
    return PipelineDefinition.model_validate(
        {
            "name": "audio-rag",
            "version": "1.0",
            "description": "test pipeline",
            "trigger": {"type": "redis-stream", "stream": "pipelines:audio-rag:jobs"},
            "steps": [
                {
                    "id": "transcribe",
                    "task": "audio-transcribe",
                    "url": "http://audio-transcribe-whisper:8000",
                }
            ],
        }
    )


@pytest.fixture
def _patch_load_pipelines():
    """Patch moiraweave_shared.pipeline.load_pipelines everywhere it's imported."""
    with (
        patch(
            "app.routes.pipelines.load_pipelines",
            return_value=[_audio_rag_pipeline()],
        ) as mock,
    ):
        yield mock


# ---------------------------------------------------------------------------
# GET /v1/pipelines
# ---------------------------------------------------------------------------


async def test_list_pipelines_returns_pipeline(
    _patch_load_pipelines, client: AsyncClient
) -> None:
    resp = await client.get("/v1/pipelines")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 1
    item = body[0]
    assert item["id"] == "audio-rag"
    assert item["name"] == "audio-rag"
    assert item["stream"] == "pipelines:audio-rag:jobs"
    assert item["steps"][0]["id"] == "transcribe"
    assert item["steps"][0]["task"] == "audio-transcribe"


async def test_list_pipelines_returns_empty_on_error(client: AsyncClient) -> None:
    with patch("app.routes.pipelines.load_pipelines", side_effect=Exception("boom")):
        resp = await client.get("/v1/pipelines")
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# POST /v1/pipelines/{id}/jobs
# ---------------------------------------------------------------------------


async def test_submit_job_returns_202(
    _patch_load_pipelines, auth_client: AsyncClient, fake_redis: FakeRedis
) -> None:
    resp = await auth_client.post(
        "/v1/pipelines/audio-rag/jobs",
        json={"payload": {"audio_url": "http://example.com/audio.mp3"}},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert "job_id" in body
    assert body["pipeline_id"] == "audio-rag"
    assert body["status"] == "pending"
    assert "created_at" in body


async def test_submit_job_stores_status_in_redis(
    _patch_load_pipelines, auth_client: AsyncClient, fake_redis: FakeRedis
) -> None:
    resp = await auth_client.post(
        "/v1/pipelines/audio-rag/jobs",
        json={"payload": {"audio_url": "http://example.com/audio.mp3"}},
    )
    job_id = resp.json()["job_id"]
    status = await fake_redis.hget(f"pipeline:job:{job_id}", "status")
    assert status == "pending"


async def test_submit_job_publishes_to_stream(
    _patch_load_pipelines, auth_client: AsyncClient, fake_redis: FakeRedis
) -> None:
    await auth_client.post(
        "/v1/pipelines/audio-rag/jobs",
        json={"payload": {"audio_url": "http://example.com/audio.mp3"}},
    )
    entries = await fake_redis.xrange("pipelines:audio-rag:jobs")
    assert len(entries) == 1
    _msg_id, fields = entries[0]
    assert fields["pipeline_id"] == "audio-rag"
    # payload must be valid JSON
    parsed = json.loads(fields["payload"])
    assert parsed["audio_url"] == "http://example.com/audio.mp3"


async def test_submit_job_unknown_pipeline_returns_404(
    _patch_load_pipelines, auth_client: AsyncClient
) -> None:
    resp = await auth_client.post(
        "/v1/pipelines/unknown-pipeline/jobs",
        json={"payload": {}},
    )
    assert resp.status_code == 404


async def test_submit_job_requires_auth(
    _patch_load_pipelines, client: AsyncClient
) -> None:
    resp = await client.post(
        "/v1/pipelines/audio-rag/jobs",
        json={"payload": {}},
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# GET /v1/pipelines/jobs/{job_id}
# ---------------------------------------------------------------------------


async def test_get_pipeline_job_status_returns_payload(
    auth_client: AsyncClient, fake_redis: FakeRedis
) -> None:
    job_id = "pipeline-job-1"
    await fake_redis.hset(
        f"pipeline:job:{job_id}",
        mapping={
            "status": "completed",
            "pipeline_id": "audio-rag",
            "user": "testuser",
            "created_at": "2026-01-01T00:00:00+00:00",
            "completed_at": "2026-01-01T00:01:00+00:00",
            "result": json.dumps({"transcript": "ok"}),
        },
    )

    resp = await auth_client.get(f"/v1/pipelines/jobs/{job_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["pipeline_id"] == "audio-rag"
    assert body["status"] == "completed"
    assert body["result"]["transcript"] == "ok"


async def test_get_pipeline_job_status_returns_404(
    auth_client: AsyncClient,
) -> None:
    resp = await auth_client.get("/v1/pipelines/jobs/missing-job")
    assert resp.status_code == 404


async def test_get_pipeline_job_status_returns_403_for_other_user(
    auth_client: AsyncClient, fake_redis: FakeRedis
) -> None:
    job_id = "pipeline-job-2"
    await fake_redis.hset(
        f"pipeline:job:{job_id}",
        mapping={
            "status": "pending",
            "pipeline_id": "audio-rag",
            "user": "another-user",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )

    resp = await auth_client.get(f"/v1/pipelines/jobs/{job_id}")
    assert resp.status_code == 403


# ---------------------------------------------------------------------------
# GET /v1/pipelines/jobs
# ---------------------------------------------------------------------------


async def test_list_pipeline_jobs_returns_user_jobs(
    auth_client: AsyncClient, fake_redis: FakeRedis
) -> None:
    for i, st in enumerate(("pending", "completed", "failed")):
        await fake_redis.hset(
            f"pipeline:job:list-job-{i}",
            mapping={
                "status": st,
                "pipeline_id": "audio-rag",
                "user": "testuser",
                "created_at": f"2026-01-0{i + 1}T00:00:00+00:00",
            },
        )

    resp = await auth_client.get("/v1/pipelines/jobs")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body) == 3
    # newest first
    assert body[0]["created_at"] > body[1]["created_at"]


async def test_list_pipeline_jobs_filters_by_pipeline_id(
    auth_client: AsyncClient, fake_redis: FakeRedis
) -> None:
    await fake_redis.hset(
        "pipeline:job:filter-job-a",
        mapping={
            "status": "pending",
            "pipeline_id": "audio-rag",
            "user": "testuser",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )
    await fake_redis.hset(
        "pipeline:job:filter-job-b",
        mapping={
            "status": "pending",
            "pipeline_id": "image-search",
            "user": "testuser",
            "created_at": "2026-01-02T00:00:00+00:00",
        },
    )

    resp = await auth_client.get("/v1/pipelines/jobs?pipeline_id=audio-rag")
    assert resp.status_code == 200
    body = resp.json()
    assert all(j["pipeline_id"] == "audio-rag" for j in body)


async def test_list_pipeline_jobs_excludes_other_users(
    auth_client: AsyncClient, fake_redis: FakeRedis
) -> None:
    await fake_redis.hset(
        "pipeline:job:other-user-job",
        mapping={
            "status": "pending",
            "pipeline_id": "audio-rag",
            "user": "someone-else",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )

    resp = await auth_client.get("/v1/pipelines/jobs")
    assert resp.status_code == 200
    assert resp.json() == []


async def test_list_pipeline_jobs_requires_auth(client: AsyncClient) -> None:
    resp = await client.get("/v1/pipelines/jobs")
    assert resp.status_code == 401
