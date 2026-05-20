"""Generic pipeline runner — executes a PipelineDefinition by calling steps via HTTP.

Each step is called via the KServe V2 ``POST /v2/models/{id}/infer`` endpoint.
The runner converts the job payload dict to V2 input tensors, passes outputs
from one step as inputs to the next, and returns the final output dict.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel

if TYPE_CHECKING:
    from moiraweave_shared.pipeline import PipelineDefinition, StepConfig

from app.config import get_settings

logger = logging.getLogger(__name__)


class _InferTensor(BaseModel):
    """Single tensor entry in a KServe V2 inference response."""

    name: str
    data: list[Any]


class _InferResponse(BaseModel):
    """KServe V2 inference response envelope (outputs only)."""

    outputs: list[_InferTensor] = []


class PipelineRunner:
    """Execute a :class:`~moiraweave_shared.pipeline.PipelineDefinition` by calling
    each step's ``/v2/models/{id}/infer`` endpoint in declaration order.

    :param pipeline: Validated pipeline definition.
    """

    def __init__(self, pipeline: PipelineDefinition) -> None:
        self._pipeline = pipeline

    async def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute the pipeline for a single job payload.

        Each step receives either the previous step's output (when
        ``input_from`` is ``None``) or the named upstream step's output (when
        ``input_from`` is set to a step ID).  The first step always receives
        the original job *payload*.

        :param payload: Flat ``{name: value}`` dict of job inputs.
        :returns: Final step output as a ``{tensor_name: value}`` dict.
        :raises httpx.HTTPStatusError: If any step returns a non-2xx response.
        :raises httpx.RequestError: If any step is unreachable or times out.
        :raises ValueError: If ``input_from`` references a step that has not run yet.
        """
        step_outputs: dict[str, dict[str, Any]] = {}
        last_output: dict[str, Any] = payload

        async with httpx.AsyncClient(
            timeout=get_settings().step_timeout_seconds
        ) as client:
            for step in self._pipeline.steps:
                if step.input_from is not None:
                    if step.input_from not in step_outputs:
                        raise ValueError(
                            f"Step '{step.id}' declares input_from='{step.input_from}' "
                            f"but that step ID has not executed yet or does not exist."
                        )
                    step_input = step_outputs[step.input_from]
                else:
                    step_input = last_output

                output = await _call_step(step, step_input, client)
                step_outputs[step.id] = output
                last_output = output

        return last_output

    async def check_ready(self, timeout: float = 5.0) -> bool:
        """Return ``True`` when every step's ``GET /v2/health/ready`` responds 200.

        :param timeout: Per-step request timeout in seconds.
        :returns: ``True`` only when all steps are healthy.
        """
        async with httpx.AsyncClient(timeout=timeout) as client:
            for step in self._pipeline.steps:
                try:
                    resp = await client.get(f"{step.url}/v2/health/ready")
                    if not resp.is_success:
                        logger.warning(
                            "step_not_ready step=%s status=%d",
                            step.id,
                            resp.status_code,
                        )
                        return False
                except httpx.RequestError:
                    logger.warning("step_unreachable step=%s url=%s", step.id, step.url)
                    return False
        return True


async def _call_step(
    step: StepConfig, payload: dict[str, Any], client: httpx.AsyncClient
) -> dict[str, Any]:
    """POST *payload* to a step's V2 infer endpoint and return the output dict.

    :param step: Step configuration (id, url).
    :param payload: ``{name: value}`` dict of input tensors.
    :param client: Shared async HTTP client (connection pool reused across steps).
    :returns: ``{tensor_name: value}`` dict from the step's response outputs.
    :raises httpx.HTTPStatusError: When the step returns a non-2xx status.
    :raises httpx.RequestError: When the step is unreachable or times out.
    :raises pydantic.ValidationError: When the response body does not conform to V2 schema.
    """
    url = f"{step.url}/v2/models/{step.id}/infer"
    request_body = {
        "inputs": [
            {"name": k, "shape": [1], "datatype": "BYTES", "data": [str(v)]}
            for k, v in payload.items()
        ],
    }
    resp = await client.post(url, json=request_body)
    resp.raise_for_status()

    parsed = _InferResponse.model_validate(resp.json())
    return {t.name: t.data[0] if len(t.data) == 1 else t.data for t in parsed.outputs}
