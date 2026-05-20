"""Pipeline definition loader.

Reads ``pipelines/<name>/pipeline.yaml`` files and validates them with Pydantic.
Both api-gateway (to list/validate pipelines) and worker (to execute pipelines)
use this module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel


class TriggerDefinition(BaseModel):
    """Stream-based trigger for pipeline job ingestion.

    :param type: Trigger mechanism — currently only ``"redis-stream"``.
    :param stream: Redis Stream key, e.g. ``"pipelines:audio-rag:jobs"``.
    """

    type: str
    stream: str


class StepConfig(BaseModel):
    """Single-step declaration inside a pipeline.

    :param id: Unique step identifier used as the KServe V2 model name.
    :param task: Task schema name, e.g. ``"audio-transcribe"``.
    :param url: Base URL of the step service, e.g. ``"http://step:8000"``.
    :param env: Per-step env var overrides (applied by the orchestrator).
    :param input_from: ID of the upstream step whose outputs feed this step's inputs.
                       ``None`` means the step reads from the previous step's output
                       (or from the original job payload if it is the first step).
    """

    id: str
    task: str
    url: str
    env: dict[str, str] = {}
    input_from: str | None = None


class PipelineDefinition(BaseModel):
    """Full pipeline definition as declared in ``pipeline.yaml``.

    :param name: Unique pipeline identifier.
    :param version: Semantic version string.
    :param description: Human-readable description.
    :param trigger: How jobs enter the pipeline.
    :param env: Shared env var defaults for all steps.
    :param steps: Ordered list of steps to execute.
    """

    name: str
    version: str
    description: str = ""
    trigger: TriggerDefinition
    env: dict[str, str] = {}
    steps: list[StepConfig]

    @classmethod
    def from_yaml(cls, path: Path) -> PipelineDefinition:
        """Load and validate a pipeline definition from a YAML file.

        :param path: Path to the ``pipeline.yaml`` file.
        :returns: Validated :class:`PipelineDefinition` instance.
        """
        data: dict[str, Any] = yaml.safe_load(path.read_text())
        return cls.model_validate(data)


def load_pipelines(pipelines_dir: str | Path) -> list[PipelineDefinition]:
    """Load all ``pipeline.yaml`` files found directly under *pipelines_dir*.

    :param pipelines_dir: Directory containing per-pipeline subdirectories,
        each with a ``pipeline.yaml`` file.
    :returns: List of validated :class:`PipelineDefinition` instances sorted
        by pipeline name.
    """
    base = Path(pipelines_dir)
    if not base.exists() or not base.is_dir():
        return []

    pipelines: list[PipelineDefinition] = []
    for yaml_path in sorted(base.glob("*/pipeline.yaml")):
        pipelines.append(PipelineDefinition.from_yaml(yaml_path))
    return pipelines
