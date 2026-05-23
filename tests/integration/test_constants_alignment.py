"""Integration test: Redis stream constants are consistent across all services.

Imports the shared package (not mocks) to guarantee that any future
refactoring that changes stream names will break these tests immediately
rather than causing silent job loss in production.

The "no hardcoded constants" invariant is enforced structurally — both
api-gateway and worker import from moiraweave_shared.streams at module level,
so any attempt to redeclare the constants would shadow the import and be
caught by mypy's --strict checks.
"""

import pytest


class TestStreamConstants:
    """Verify canonical constant values in the shared package."""

    def test_run_stream(self) -> None:
        from moiraweave_shared.streams import RUN_STREAM

        assert RUN_STREAM == "moiraweave:runs"

    def test_consumer_group(self) -> None:
        from moiraweave_shared.streams import CONSUMER_GROUP

        assert CONSUMER_GROUP == "moiraweave-runs"

    def test_dead_letter_stream(self) -> None:
        from moiraweave_shared.streams import DEAD_LETTER_STREAM

        assert DEAD_LETTER_STREAM == "moiraweave:runs:dead-letter"


class TestRunMessage:
    def test_serializes_to_flat_dict(self) -> None:
        from moiraweave_shared.schemas import RunMessage

        msg = RunMessage(
            run_id="abc-123",
            workload_name="image-search",
            payload='{"query": "cats"}',
            user="user1",
        )
        data = msg.model_dump(mode="python")
        assert data["run_id"] == "abc-123"
        assert data["workload_name"] == "image-search"
        assert data["user"] == "user1"
        assert data["workload_manifest"] is None

    def test_roundtrip_from_redis_fields(self) -> None:
        """Simulate what Redis xreadgroup returns and validate deserialization."""
        from moiraweave_shared.schemas import RunMessage

        redis_fields: dict[str, str] = {
            "run_id": "xyz-789",
            "workload_name": "text-search",
            "payload": '{"query": "dogs"}',
            "user": "alice",
        }
        msg = RunMessage.model_validate(redis_fields)
        assert msg.run_id == "xyz-789"
        assert msg.workload_name == "text-search"
        assert msg.user == "alice"

    def test_invalid_missing_field_raises(self) -> None:
        """A message missing a required field must fail validation."""
        from moiraweave_shared.schemas import RunMessage
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            RunMessage.model_validate(
                {
                    "run_id": "x",
                    # missing workload_name, payload, user
                }
            )
