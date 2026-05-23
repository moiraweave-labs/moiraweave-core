"""Tests for agent runtime adapters."""

from __future__ import annotations

from typing import Any

from moiraweave_shared.workloads import WorkloadDefinition

from app.agent_adapters import (
    HttpAgentAdapter,
    build_agent_adapter,
    extract_assistant_message,
)


def _agent_workload(**spec_overrides: Any) -> WorkloadDefinition:
    spec: dict[str, Any] = {
        "type": "agent-service",
        "image": "ghcr.io/example/agent:latest",
        "execution": {"mode": "session", "timeoutSeconds": 60},
        "ports": [{"name": "http", "port": 8000}],
    }
    spec.update(spec_overrides)
    return WorkloadDefinition.model_validate(
        {
            "apiVersion": "moiraweave.io/v1alpha1",
            "kind": "Workload",
            "metadata": {"name": "agent"},
            "spec": spec,
        }
    )


async def test_generic_agent_without_endpoint_accepts_locally() -> None:
    adapter = build_agent_adapter(_agent_workload(ports=[]), timeout_seconds=1.0)

    response = await adapter.send_message({"session_id": "session-1", "message": "hi"})

    assert response["accepted"] is True
    assert response["mode"] == "local-accept"


async def test_hermes_adapter_routes_session_messages(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_post(
        self: HttpAgentAdapter, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {"response": "ack"}

    monkeypatch.setattr(HttpAgentAdapter, "_post", fake_post)
    workload = _agent_workload(
        endpoint="http://hermes:8000",
        agent={"adapter": "hermes"},
    )
    adapter = build_agent_adapter(workload, timeout_seconds=1.0)

    response = await adapter.send_message({"session_id": "s-1", "message": "hello"})

    assert captured["endpoint"] == "http://hermes:8000/sessions/s-1/messages"
    assert captured["payload"]["message"] == "hello"
    assert response["accepted"] is True
    assert response["adapter"] == "hermes"


async def test_generic_adapter_status_cancel_and_artifacts(monkeypatch) -> None:
    calls: list[str] = []

    async def fake_get(self: HttpAgentAdapter, endpoint: str) -> dict[str, Any]:
        del self
        calls.append(endpoint)
        if endpoint.endswith("/artifacts"):
            return {"artifacts": [{"name": "out.txt"}]}
        return {"status": "running"}

    async def fake_post(
        self: HttpAgentAdapter, endpoint: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        del self, payload
        calls.append(endpoint)
        return {"ok": True}

    monkeypatch.setattr(HttpAgentAdapter, "_get", fake_get)
    monkeypatch.setattr(HttpAgentAdapter, "_post", fake_post)
    adapter = build_agent_adapter(
        _agent_workload(
            endpoint="http://agent:8000",
            agent={
                "adapter": "generic-http",
                "statusPath": "/status",
                "cancelPath": "/stop",
                "artifactsPath": "/artifacts",
            },
        ),
        timeout_seconds=1.0,
    )

    assert (await adapter.get_status({}))["status"] == "running"
    assert (await adapter.cancel({}))["accepted"] is True
    assert (await adapter.list_artifacts({}))[0]["name"] == "out.txt"
    assert calls == [
        "http://agent:8000/status",
        "http://agent:8000/stop",
        "http://agent:8000/artifacts",
    ]


async def test_adapter_uses_service_port_when_endpoint_is_omitted(monkeypatch) -> None:
    captured: dict[str, str] = {}

    async def fake_get(self: HttpAgentAdapter, endpoint: str) -> dict[str, Any]:
        del self
        captured["endpoint"] = endpoint
        return {"status": "ok"}

    monkeypatch.setattr(HttpAgentAdapter, "_get", fake_get)
    adapter = build_agent_adapter(_agent_workload(), timeout_seconds=1.0)

    await adapter.get_status({})

    assert captured["endpoint"] == "http://agent:8000/health"


def test_extract_assistant_message_common_shapes() -> None:
    assert extract_assistant_message({"response": "done"}) == "done"
    assert extract_assistant_message({"assistant": {"content": "nested"}}) == "nested"
    assert extract_assistant_message({"accepted": True}) is None
