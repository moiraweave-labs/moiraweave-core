"""Minimal KServe V2 echo step for E2E tests.

Returns every input tensor unchanged as output tensors.
No ML models, no external dependencies — pure pass-through.
"""

from moiraweave_step_sdk.base import BaseStep
from moiraweave_step_sdk.models import InferRequest, InferResponse


class EchoStep(BaseStep):
    """Returns all input tensors unchanged as outputs."""

    @property
    def name(self) -> str:
        return "echo"

    @property
    def version(self) -> str:
        return "1"

    @property
    def task(self) -> str:
        return "text-passthrough"

    async def predict(self, request: InferRequest) -> InferResponse:
        return InferResponse(
            model_name=self.name,
            id=request.id,
            outputs=list(request.inputs),
        )


app = EchoStep().build_app()
