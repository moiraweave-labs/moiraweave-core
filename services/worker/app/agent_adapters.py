"""Agent runtime adapters.

MoiraWeave owns the control plane around agents.  The actual reasoning loop,
tools, memory, and runtime-specific behavior stay inside the deployed agent.
Adapters provide the small operational contract MoiraWeave needs: dispatch a
message, check status, request cancellation, and discover artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol

import httpx

if TYPE_CHECKING:
    from moiraweave_shared.workloads import WorkloadDefinition


class AgentAdapter(Protocol):
    """Operational contract for agent runtimes."""

    name: str

    async def send_message(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def get_status(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def cancel(self, payload: dict[str, Any]) -> dict[str, Any]: ...

    async def list_artifacts(self, payload: dict[str, Any]) -> list[dict[str, Any]]: ...


class HttpAgentAdapter:
    """HTTP adapter used by generic, Hermes, and OpenClaw-style agents."""

    def __init__(self, workload: WorkloadDefinition, *, timeout_seconds: float) -> None:
        self.workload = workload
        self.timeout_seconds = timeout_seconds
        self.name = _adapter_name(workload)

    async def send_message(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = self._url(payload, operation="message")
        if endpoint is None:
            return {
                "workload": self.workload.metadata.name,
                "session_id": payload.get("session_id"),
                "accepted": True,
                "adapter": self.name,
                "mode": "local-accept",
            }
        response = await self._post(endpoint, payload)
        response.setdefault("accepted", True)
        response.setdefault("adapter", self.name)
        return response

    async def get_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = self._url(payload, operation="status")
        if endpoint is None:
            return {"status": "unknown", "adapter": self.name}
        return await self._get(endpoint)

    async def cancel(self, payload: dict[str, Any]) -> dict[str, Any]:
        endpoint = self._url(payload, operation="cancel")
        if endpoint is None:
            return {"accepted": False, "reason": "cancel endpoint not configured"}
        response = await self._post(endpoint, payload)
        response.setdefault("accepted", True)
        return response

    async def list_artifacts(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        endpoint = self._url(payload, operation="artifacts")
        if endpoint is None:
            return []
        response = await self._get(endpoint)
        artifacts = response.get("artifacts")
        return artifacts if isinstance(artifacts, list) else []

    async def _post(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(endpoint, json=payload)
            response.raise_for_status()
            return _json_dict(response)

    async def _get(self, endpoint: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.get(endpoint)
            response.raise_for_status()
            return _json_dict(response)

    def _url(self, payload: dict[str, Any], *, operation: str) -> str | None:
        base = _base_endpoint(self.workload)
        if base is None:
            return None
        path = self._path(payload, operation=operation)
        return _join_url(base, path)

    def _path(self, payload: dict[str, Any], *, operation: str) -> str:
        agent_spec = self.workload.spec.agent
        configured = {
            "message": agent_spec.messagePath,
            "status": agent_spec.statusPath,
            "cancel": agent_spec.cancelPath,
            "artifacts": agent_spec.artifactsPath,
        }[operation]
        if configured:
            return configured

        session_id = str(payload.get("session_id") or "")
        if self.name in {"hermes", "openclaw"} and session_id:
            defaults = {
                "message": f"/sessions/{session_id}/messages",
                "status": f"/sessions/{session_id}",
                "cancel": f"/sessions/{session_id}/cancel",
                "artifacts": f"/sessions/{session_id}/artifacts",
            }
            return defaults[operation]

        defaults = {
            "message": "",
            "status": "/health",
            "cancel": "/cancel",
            "artifacts": "/artifacts",
        }
        return defaults[operation]


def build_agent_adapter(
    workload: WorkloadDefinition, *, timeout_seconds: float
) -> AgentAdapter:
    """Build the adapter declared by a workload manifest."""

    return HttpAgentAdapter(workload, timeout_seconds=timeout_seconds)


def extract_assistant_message(result: Mapping[str, Any]) -> str | None:
    """Extract a readable agent response from common runtime response shapes."""

    for key in ("message", "response", "content", "text", "output"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value

    nested = result.get("assistant")
    if isinstance(nested, Mapping):
        content = nested.get("message") or nested.get("content")
        if isinstance(content, str) and content.strip():
            return content
    return None


def _adapter_name(workload: WorkloadDefinition) -> str:
    if workload.spec.adapter:
        return "generic-http" if workload.spec.adapter == "generic" else workload.spec.adapter
    if workload.spec.agent.adapter == "generic":
        return "generic-http"
    return workload.spec.agent.adapter


def _base_endpoint(workload: WorkloadDefinition) -> str | None:
    if workload.spec.endpoint:
        return workload.spec.endpoint.rstrip("/")
    if not workload.spec.ports:
        return None
    port = workload.spec.ports[0].port
    return f"http://{workload.metadata.name}:{port}"


def _join_url(base: str, path: str) -> str:
    if not path:
        return base
    return f"{base.rstrip('/')}/{path.lstrip('/')}"


def _json_dict(response: httpx.Response) -> dict[str, Any]:
    data = response.json()
    return data if isinstance(data, dict) else {"response": data}
