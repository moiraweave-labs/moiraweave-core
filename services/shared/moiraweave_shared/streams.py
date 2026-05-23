"""Single source of truth for workload runtime Redis keys."""

#: Shared dispatch stream.  Redis is now only the queue/coordination layer.
RUN_STREAM: str = "moiraweave:runs"

#: Consumer group name for the worker fleet.
CONSUMER_GROUP: str = "moiraweave-runs"

#: Dead-letter stream for malformed or unrecoverable dispatch messages.
DEAD_LETTER_STREAM: str = "moiraweave:runs:dead-letter"
