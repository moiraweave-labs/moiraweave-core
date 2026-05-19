"""Tests for moiraweave_shared.pipeline — YAML loading and Pydantic validation."""

import textwrap
from pathlib import Path

import pytest
import yaml
from moiraweave_shared.pipeline import (
    PipelineDefinition,
    StepConfig,
    TriggerDefinition,
    load_pipelines,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_PIPELINE_YAML = textwrap.dedent(
    """
    name: audio-rag
    version: "1.0"
    trigger:
      type: redis-stream
      stream: pipelines:audio-rag:jobs
    steps:
      - id: transcribe
        task: audio-transcribe
        url: http://transcribe:8080
    """
)

_FULL_PIPELINE_YAML = textwrap.dedent(
    """
    name: full-pipeline
    version: "2.3"
    description: A full pipeline with all fields
    trigger:
      type: redis-stream
      stream: pipelines:full:jobs
    env:
      LOG_LEVEL: INFO
      DEVICE: cuda
    steps:
      - id: embed
        task: text-embed
        url: http://embed:8080
        env:
          MODEL: BAAI/bge-small-en
      - id: index
        task: vector-index
        url: http://index:8080
        input_from: embed
    """
)


def _write_pipeline(tmp_path: Path, name: str, content: str) -> Path:
    pipeline_dir = tmp_path / name
    pipeline_dir.mkdir(parents=True)
    yaml_file = pipeline_dir / "pipeline.yaml"
    yaml_file.write_text(content)
    return yaml_file


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


def test_pipeline_definition_minimal_fields() -> None:
    """PipelineDefinition accepts a minimal definition without optional fields."""
    data = yaml.safe_load(_MINIMAL_PIPELINE_YAML)
    pipeline = PipelineDefinition.model_validate(data)

    assert pipeline.name == "audio-rag"
    assert pipeline.version == "1.0"
    assert pipeline.description == ""
    assert len(pipeline.steps) == 1
    assert pipeline.steps[0].id == "transcribe"


def test_pipeline_definition_full_fields() -> None:
    """PipelineDefinition correctly parses all optional fields."""
    data = yaml.safe_load(_FULL_PIPELINE_YAML)
    pipeline = PipelineDefinition.model_validate(data)

    assert pipeline.name == "full-pipeline"
    assert pipeline.description == "A full pipeline with all fields"
    assert pipeline.env == {"LOG_LEVEL": "INFO", "DEVICE": "cuda"}
    assert len(pipeline.steps) == 2

    embed_step = pipeline.steps[0]
    assert embed_step.id == "embed"
    assert embed_step.env == {"MODEL": "BAAI/bge-small-en"}
    assert embed_step.input_from is None

    index_step = pipeline.steps[1]
    assert index_step.input_from == "embed"


def test_step_config_defaults() -> None:
    """StepConfig.env and input_from have correct defaults."""
    step = StepConfig(id="s", task="text-embed", url="http://step:8080")
    assert step.env == {}
    assert step.input_from is None


def test_trigger_definition_fields() -> None:
    """TriggerDefinition stores type and stream correctly."""
    trigger = TriggerDefinition(type="redis-stream", stream="test:stream")
    assert trigger.type == "redis-stream"
    assert trigger.stream == "test:stream"


# ---------------------------------------------------------------------------
# PipelineDefinition.from_yaml
# ---------------------------------------------------------------------------


def test_from_yaml_minimal(tmp_path: Path) -> None:
    """from_yaml loads and validates a minimal pipeline YAML file."""
    yaml_path = _write_pipeline(tmp_path, "audio-rag", _MINIMAL_PIPELINE_YAML)
    pipeline = PipelineDefinition.from_yaml(yaml_path)

    assert pipeline.name == "audio-rag"
    assert pipeline.trigger.type == "redis-stream"


def test_from_yaml_full(tmp_path: Path) -> None:
    """from_yaml loads a full pipeline YAML with all optional fields."""
    yaml_path = _write_pipeline(tmp_path, "full", _FULL_PIPELINE_YAML)
    pipeline = PipelineDefinition.from_yaml(yaml_path)

    assert pipeline.name == "full-pipeline"
    assert pipeline.steps[1].input_from == "embed"


def test_from_yaml_invalid_missing_required_field(tmp_path: Path) -> None:
    """from_yaml raises a Pydantic ValidationError when required fields are absent."""
    from pydantic import ValidationError

    bad_yaml = textwrap.dedent(
        """
        version: "1.0"
        trigger:
          type: redis-stream
          stream: test:jobs
        steps: []
        """
    )
    yaml_path = _write_pipeline(tmp_path, "bad", bad_yaml)
    with pytest.raises(ValidationError):
        PipelineDefinition.from_yaml(yaml_path)


# ---------------------------------------------------------------------------
# load_pipelines
# ---------------------------------------------------------------------------


def test_load_pipelines_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    """load_pipelines returns [] when the directory does not exist."""
    result = load_pipelines(tmp_path / "nonexistent")
    assert result == []


def test_load_pipelines_returns_empty_for_empty_dir(tmp_path: Path) -> None:
    """load_pipelines returns [] when the directory has no pipeline subdirs."""
    result = load_pipelines(tmp_path)
    assert result == []


def test_load_pipelines_single(tmp_path: Path) -> None:
    """load_pipelines loads a single pipeline correctly."""
    _write_pipeline(tmp_path, "audio-rag", _MINIMAL_PIPELINE_YAML)
    result = load_pipelines(tmp_path)

    assert len(result) == 1
    assert result[0].name == "audio-rag"


def test_load_pipelines_multiple_sorted(tmp_path: Path) -> None:
    """load_pipelines returns pipelines sorted alphabetically by directory name."""
    _write_pipeline(
        tmp_path,
        "z-pipeline",
        _MINIMAL_PIPELINE_YAML.replace("audio-rag", "z-pipeline"),
    )
    _write_pipeline(
        tmp_path,
        "a-pipeline",
        _MINIMAL_PIPELINE_YAML.replace("audio-rag", "a-pipeline"),
    )
    _write_pipeline(tmp_path, "m-pipeline", _FULL_PIPELINE_YAML)

    result = load_pipelines(tmp_path)

    assert len(result) == 3
    # Results are sorted by glob which is alphabetical
    names = [p.name for p in result]
    assert names == sorted(names)


def test_load_pipelines_ignores_files_at_root(tmp_path: Path) -> None:
    """load_pipelines ignores loose files at the root of the directory."""
    # Loose YAML file directly in pipelines_dir — should be ignored
    (tmp_path / "pipeline.yaml").write_text(_MINIMAL_PIPELINE_YAML)
    result = load_pipelines(tmp_path)
    assert result == []
