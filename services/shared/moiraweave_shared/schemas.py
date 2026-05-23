"""Pydantic v2 schemas for Redis Stream messages."""

from pydantic import BaseModel


class RunMessage(BaseModel):
    """Payload for a workload run written to the worker dispatch stream.

    ``payload`` is JSON encoded because Redis Streams store scalar values.
    """

    run_id: str
    workload_name: str
    payload: str
    user: str
    workload_manifest: str | None = None
