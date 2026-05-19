"""Tests for app.mlflow_logger — MLflow inference metrics logger."""

from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.mlflow_logger import _log_run_sync, log_inference_metrics


@pytest.fixture
def settings() -> Settings:
    return Settings(
        redis_url="redis://localhost:6379/0",  # type: ignore[arg-type]
        qdrant_collection="test-collection",
    )


# ---------------------------------------------------------------------------
# _log_run_sync — synchronous MLflow call (executed inside thread executor)
# ---------------------------------------------------------------------------


def test_log_run_sync_completed_job(settings: Settings) -> None:
    """Happy path: a completed job is logged with expected tags and metrics."""
    with (
        patch("app.mlflow_logger.mlflow.set_tracking_uri") as mock_uri,
        patch("app.mlflow_logger.mlflow.set_experiment") as mock_exp,
        patch("app.mlflow_logger.mlflow.start_run") as mock_run,
        patch("app.mlflow_logger.mlflow.set_tags") as mock_tags,
        patch("app.mlflow_logger.mlflow.log_metric") as mock_metric,
        patch("app.mlflow_logger.mlflow.log_param") as mock_param,
    ):
        mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_run.return_value.__exit__ = MagicMock(return_value=False)

        _log_run_sync(
            settings,
            job_id="job-abc",
            duration_seconds=1.5,
            status="completed",
            tokens_per_second=120.0,
        )

    mock_uri.assert_called_once()
    mock_exp.assert_called_once()
    mock_run.assert_called_once_with(run_name="job-job-abc")

    tags_call = mock_tags.call_args[0][0]
    assert tags_call["job.id"] == "job-abc"
    assert tags_call["job.status"] == "completed"

    calls = mock_metric.call_args_list
    metric_names = {c[0][0] for c in calls}
    assert "duration_seconds" in metric_names
    assert "success" in metric_names
    assert "tokens_per_second" in metric_names

    # success=1.0 for a completed job
    success_call = next(c for c in calls if c[0][0] == "success")
    assert success_call[0][1] == 1.0

    mock_param.assert_not_called()


def test_log_run_sync_failed_job(settings: Settings) -> None:
    """A failed job logs success=0.0 and the error_type param."""
    with (
        patch("app.mlflow_logger.mlflow.set_tracking_uri"),
        patch("app.mlflow_logger.mlflow.set_experiment"),
        patch("app.mlflow_logger.mlflow.start_run") as mock_run,
        patch("app.mlflow_logger.mlflow.set_tags"),
        patch("app.mlflow_logger.mlflow.log_metric") as mock_metric,
        patch("app.mlflow_logger.mlflow.log_param") as mock_param,
    ):
        mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_run.return_value.__exit__ = MagicMock(return_value=False)

        _log_run_sync(
            settings,
            job_id="job-fail",
            duration_seconds=0.5,
            status="failed",
            error_type="TimeoutError",
        )

    success_call = next(c for c in mock_metric.call_args_list if c[0][0] == "success")
    assert success_call[0][1] == 0.0

    mock_param.assert_called_once_with("error_type", "TimeoutError")


def test_log_run_sync_no_optional_fields(settings: Settings) -> None:
    """tokens_per_second and error_type are both omitted when not provided."""
    with (
        patch("app.mlflow_logger.mlflow.set_tracking_uri"),
        patch("app.mlflow_logger.mlflow.set_experiment"),
        patch("app.mlflow_logger.mlflow.start_run") as mock_run,
        patch("app.mlflow_logger.mlflow.set_tags"),
        patch("app.mlflow_logger.mlflow.log_metric") as mock_metric,
        patch("app.mlflow_logger.mlflow.log_param") as mock_param,
    ):
        mock_run.return_value.__enter__ = MagicMock(return_value=MagicMock())
        mock_run.return_value.__exit__ = MagicMock(return_value=False)

        _log_run_sync(settings, job_id="j", duration_seconds=1.0, status="completed")

    metric_names = {c[0][0] for c in mock_metric.call_args_list}
    assert "tokens_per_second" not in metric_names
    mock_param.assert_not_called()


def test_log_run_sync_swallows_mlflow_exception(settings: Settings) -> None:
    """MLflow failures must not propagate — logging is best-effort."""
    with (
        patch(
            "app.mlflow_logger.mlflow.set_tracking_uri",
            side_effect=ConnectionRefusedError("mlflow down"),
        ),
        patch("app.mlflow_logger.logger") as mock_log,
    ):
        # Should not raise
        _log_run_sync(settings, job_id="j", duration_seconds=1.0, status="completed")

    mock_log.warning.assert_called_once()
    assert "mlflow_log_failed" in mock_log.warning.call_args[0][0]


# ---------------------------------------------------------------------------
# log_inference_metrics — async wrapper
# ---------------------------------------------------------------------------


async def test_log_inference_metrics_delegates_to_sync(settings: Settings) -> None:
    """The async wrapper should call _log_run_sync inside the executor."""
    with patch("app.mlflow_logger._log_run_sync") as mock_sync:
        await log_inference_metrics(
            settings,
            job_id="async-job",
            duration_seconds=2.0,
            status="completed",
            tokens_per_second=50.0,
        )

    mock_sync.assert_called_once()
    _, call_kwargs = mock_sync.call_args
    # Keyword arguments are forwarded correctly
    assert call_kwargs.get("tokens_per_second") == 50.0


async def test_log_inference_metrics_failed_job(settings: Settings) -> None:
    """Async path for a failed job forwards error_type."""
    with patch("app.mlflow_logger._log_run_sync") as mock_sync:
        await log_inference_metrics(
            settings,
            job_id="fail-job",
            duration_seconds=0.1,
            status="failed",
            error_type="ValueError",
        )

    call_args, call_kwargs = mock_sync.call_args
    # positional order: settings, job_id, duration_seconds, status
    assert call_args[3] == "failed"
    assert call_kwargs.get("error_type") == "ValueError"
