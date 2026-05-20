"""Tests for PipelineRunner — unit tests using httpx mock transport."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from pydantic import ValidationError
from moiraweave_shared.pipeline import PipelineDefinition

from app.pipeline_runner import PipelineRunner


def _make_pipeline(url: str = "http://step:8000") -> PipelineDefinition:
    return PipelineDefinition.model_validate(
        {
            "name": "test-pipeline",
            "version": "1.0",
            "trigger": {"type": "redis-stream", "stream": "test:jobs"},
            "steps": [{"id": "transcribe", "task": "audio-transcribe", "url": url}],
        }
    )


def _v2_response(outputs: list[dict[str, Any]]) -> httpx.Response:
    return httpx.Response(200, json={"outputs": outputs})


async def test_run_single_step_returns_output() -> None:
    pipeline = _make_pipeline()

    with patch("app.pipeline_runner.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "outputs": [
                {
                    "name": "transcript",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": ["hello"],
                }
            ]
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        runner = PipelineRunner(pipeline)
        result = await runner.run({"audio_url": "http://audio.example.com/file.mp3"})

    assert result == {"transcript": "hello"}
    mock_client.post.assert_called_once()
    call_url = mock_client.post.call_args[0][0]
    assert call_url == "http://step:8000/v2/models/transcribe/infer"


async def test_run_passes_previous_output_as_next_input() -> None:
    """Multi-step pipeline: first step output becomes second step input."""
    pipeline = PipelineDefinition.model_validate(
        {
            "name": "two-step",
            "version": "1.0",
            "trigger": {"type": "redis-stream", "stream": "two:jobs"},
            "steps": [
                {"id": "step-a", "task": "task-a", "url": "http://step-a:8000"},
                {"id": "step-b", "task": "task-b", "url": "http://step-b:8000"},
            ],
        }
    )

    call_count = 0

    async def _fake_post(url: str, **kwargs: Any) -> MagicMock:
        nonlocal call_count
        call_count += 1
        resp = MagicMock()
        if "step-a" in url:
            resp.json.return_value = {
                "outputs": [
                    {"name": "mid", "datatype": "BYTES", "shape": [1], "data": ["v"]}
                ]
            }
        else:
            resp.json.return_value = {
                "outputs": [
                    {"name": "final", "datatype": "BYTES", "shape": [1], "data": ["ok"]}
                ]
            }
        resp.raise_for_status = MagicMock()
        return resp

    with patch("app.pipeline_runner.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=_fake_post)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        runner = PipelineRunner(pipeline)
        result = await runner.run({"input": "data"})

    assert result == {"final": "ok"}
    assert call_count == 2


async def test_check_ready_returns_true_when_all_steps_healthy() -> None:
    pipeline = _make_pipeline()

    with patch("app.pipeline_runner.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_resp = AsyncMock()
        mock_resp.is_success = True
        mock_client.get = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        runner = PipelineRunner(pipeline)
        assert await runner.check_ready() is True


async def test_check_ready_returns_false_when_step_unreachable() -> None:
    pipeline = _make_pipeline()

    with patch("app.pipeline_runner.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.RequestError("conn refused"))
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        runner = PipelineRunner(pipeline)
        assert await runner.check_ready() is False


# ---------------------------------------------------------------------------
# Error path tests
# ---------------------------------------------------------------------------


def _make_two_step_pipeline() -> PipelineDefinition:
    return PipelineDefinition.model_validate(
        {
            "name": "two-step",
            "version": "1.0",
            "trigger": {"type": "redis-stream", "stream": "two:jobs"},
            "steps": [
                {"id": "step-a", "task": "task-a", "url": "http://step-a:8000"},
                {"id": "step-b", "task": "task-b", "url": "http://step-b:8000"},
            ],
        }
    )


async def test_run_raises_on_step_http_error() -> None:
    """HTTPStatusError from any step propagates out of run()."""
    pipeline = _make_pipeline()

    with patch("app.pipeline_runner.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=MagicMock(),
            response=MagicMock(status_code=500),
        )
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        runner = PipelineRunner(pipeline)
        with pytest.raises(httpx.HTTPStatusError):
            await runner.run({"audio_url": "http://audio.example.com/file.mp3"})


async def test_run_raises_on_step_timeout() -> None:
    """RequestError (e.g. timeout) from any step propagates out of run()."""
    pipeline = _make_pipeline()

    with patch("app.pipeline_runner.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(
            side_effect=httpx.TimeoutException("Request timed out")
        )
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        runner = PipelineRunner(pipeline)
        with pytest.raises(httpx.TimeoutException):
            await runner.run({"audio_url": "http://audio.example.com/file.mp3"})


async def test_run_raises_on_malformed_response() -> None:
    """Pydantic ValidationError raised when step response body is not V2-shaped."""
    pipeline = _make_pipeline()

    with patch("app.pipeline_runner.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_resp = MagicMock()
        # outputs must be a list, not a string
        mock_resp.json.return_value = {"outputs": "not-a-list"}
        mock_resp.raise_for_status = MagicMock()
        mock_client.post = AsyncMock(return_value=mock_resp)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        runner = PipelineRunner(pipeline)
        with pytest.raises(ValidationError):
            await runner.run({"audio_url": "http://audio.example.com/file.mp3"})


async def test_run_input_from_routes_to_named_step_output() -> None:
    """input_from='step-a' routes step-a output to step-b even in a 3-step chain."""
    pipeline = PipelineDefinition.model_validate(
        {
            "name": "fan-out",
            "version": "1.0",
            "trigger": {"type": "redis-stream", "stream": "fan:jobs"},
            "steps": [
                {"id": "step-a", "task": "task-a", "url": "http://step-a:8000"},
                {"id": "step-b", "task": "task-b", "url": "http://step-b:8000"},
                # step-c reads from step-a directly, not from step-b
                {
                    "id": "step-c",
                    "task": "task-c",
                    "url": "http://step-c:8000",
                    "input_from": "step-a",
                },
            ],
        }
    )

    received_inputs: dict[str, dict[str, Any]] = {}

    async def _fake_post(url: str, json: dict[str, Any], **kwargs: Any) -> MagicMock:
        step_id = url.split("/v2/models/")[1].split("/infer")[0]
        received_inputs[step_id] = {
            inp["name"]: inp["data"][0] for inp in json["inputs"]
        }
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "outputs": [
                {
                    "name": f"out_{step_id}",
                    "datatype": "BYTES",
                    "shape": [1],
                    "data": [f"result_{step_id}"],
                }
            ]
        }
        return resp

    with patch("app.pipeline_runner.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=_fake_post)
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        runner = PipelineRunner(pipeline)
        result = await runner.run({"initial": "data"})

    # step-a should have received the original payload
    assert received_inputs["step-a"] == {"initial": "data"}
    # step-c should have received step-a's output, not step-b's
    assert received_inputs["step-c"] == {"out_step-a": "result_step-a"}
    # final result is step-c's output (last step)
    assert result == {"out_step-c": "result_step-c"}


async def test_run_raises_when_input_from_references_unknown_step() -> None:
    """ValueError raised when input_from names a step that has not run yet."""
    pipeline = PipelineDefinition.model_validate(
        {
            "name": "bad-ref",
            "version": "1.0",
            "trigger": {"type": "redis-stream", "stream": "bad:jobs"},
            "steps": [
                {
                    "id": "step-a",
                    "task": "task-a",
                    "url": "http://step-a:8000",
                    "input_from": "nonexistent-step",
                },
            ],
        }
    )

    with patch("app.pipeline_runner.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client_cls.return_value = mock_client

        runner = PipelineRunner(pipeline)
        with pytest.raises(ValueError, match="nonexistent-step"):
            await runner.run({"input": "data"})
