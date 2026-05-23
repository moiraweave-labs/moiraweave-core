"""End-to-end tests for the MoiraWeave workload flow.

These tests require the full docker-compose stack. They are excluded from the
default ``pytest`` run and execute through ``make test-e2e``.
"""

import httpx
import pytest

from tests.e2e.conftest import poll_run

pytestmark = pytest.mark.e2e


async def test_workload_list_includes_echo_model(
    authed_client: httpx.AsyncClient,
) -> None:
    resp = await authed_client.get("/v1/workloads")
    assert resp.status_code == 200
    workloads = resp.json()
    names = [workload["name"] for workload in workloads]
    assert "echo-model" in names, f"echo-model workload not found. Got: {names}"


async def test_submit_echo_model_run_returns_202(
    authed_client: httpx.AsyncClient,
) -> None:
    resp = await authed_client.post(
        "/v1/workloads/echo-model/runs",
        json={"payload": {"text": "hello moiraweave"}},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert "run_id" in body
    assert body["status"] == "queued"
    assert body["workload_name"] == "echo-model"


async def test_echo_model_run_completes_with_input_in_result(
    authed_client: httpx.AsyncClient,
) -> None:
    submit = await authed_client.post(
        "/v1/workloads/echo-model/runs",
        json={"payload": {"text": "round-trip", "value": "42"}},
    )
    assert submit.status_code == 202
    run_id = submit.json()["run_id"]

    run = await poll_run(authed_client, run_id)

    assert run["status"] == "succeeded", f"Run failed: {run.get('error')}"
    result = run["result"]
    assert result is not None
    assert "text" in result, f"'text' not in result: {result}"
    assert result["text"] == "round-trip"


async def test_unauthenticated_submit_returns_401(
    http_client: httpx.AsyncClient,
) -> None:
    client = httpx.AsyncClient(base_url=http_client.base_url, timeout=10.0)
    async with client:
        resp = await client.post(
            "/v1/workloads/echo-model/runs",
            json={"payload": {"text": "no-auth"}},
        )
    assert resp.status_code == 401


async def test_run_not_found_returns_404(authed_client: httpx.AsyncClient) -> None:
    resp = await authed_client.get("/v1/runs/00000000-0000-0000-0000-000000000000")
    assert resp.status_code == 404


async def test_submit_to_unknown_workload_returns_404(
    authed_client: httpx.AsyncClient,
) -> None:
    resp = await authed_client.post(
        "/v1/workloads/does-not-exist/runs",
        json={"payload": {"text": "nope"}},
    )
    assert resp.status_code == 404
