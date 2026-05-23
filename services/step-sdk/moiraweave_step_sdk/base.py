"""BaseStep — abstract base class for KServe-compatible workload components.

Each component exposes a KServe V2-compatible REST API via :meth:`build_app`.
Concrete components must implement :attr:`name`, :attr:`version`, and
:meth:`predict`.

Usage::

    class MyStep(BaseStep):
        @property
        def name(self) -> str:
            return "my-step"

        @property
        def version(self) -> str:
            return "1"

        async def predict(self, request: InferRequest) -> InferResponse:
            ...

    app = MyStep().build_app()
"""

from abc import ABC, abstractmethod

from fastapi import FastAPI, HTTPException

from moiraweave_step_sdk.models import (
    InferRequest,
    InferResponse,
    MetadataTensor,
    ModelMetadataResponse,
    ModelReadyResponse,
    ServerLiveResponse,
)


class BaseStep(ABC):
    """Abstract base for a single-model KServe V2 inference component.

    Sub-classes must implement :attr:`name`, :attr:`version`, and
    :meth:`predict`.  Override :meth:`is_ready` to add custom readiness
    logic (e.g. verify that a downstream model service is reachable).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Component identifier, e.g. ``"audio-transcribe-whisper"``."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Step version string, e.g. ``"1"``."""

    @abstractmethod
    async def predict(self, request: InferRequest) -> InferResponse:
        """Run inference on *request* and return an :class:`InferResponse`.

        :param request: Validated KServe V2 inference request.
        :returns: KServe V2 inference response.
        """

    async def is_ready(self) -> bool:
        """Return ``True`` when the step is ready to serve inference.

        Override to add custom readiness checks (model warm-up, downstream
        dependency availability, etc.).  The default always returns ``True``.
        """
        return True

    @property
    def task(self) -> str:
        """MoiraWeave task name, e.g. ``"audio-transcribe"``.

        Override in subclasses to expose task identity via ``GET /v2/models/{name}``.
        """
        return ""

    @property
    def implementation(self) -> str:
        """Implementation identifier, e.g. ``"whisper"``.

        Override in subclasses to expose implementation identity via model metadata.
        """
        return ""

    @property
    def inputs(self) -> list[MetadataTensor]:
        """Input tensor descriptors for the model metadata endpoint.

        Override to return the step's actual input schema.
        """
        return []

    @property
    def outputs(self) -> list[MetadataTensor]:
        """Output tensor descriptors for the model metadata endpoint.

        Override to return the step's actual output schema.
        """
        return []

    def build_app(self) -> FastAPI:
        """Construct and return the FastAPI application for this step.

        Registers the following endpoints:

        * ``GET  /v2/health/live``                  — liveness probe
        * ``GET  /v2/health/ready``                 — server-ready probe
        * ``GET  /v2/models/{model_name}/ready``    — model-ready probe
        * ``POST /v2/models/{model_name}/infer``    — inference

        :returns: Configured :class:`fastapi.FastAPI` instance.
        """
        step = self  # captured by closures below
        app = FastAPI(
            title=step.name,
            version=step.version,
            description=f"MoiraWeave step: {step.name}",
        )

        @app.get("/v2/health/live", response_model=ServerLiveResponse)
        async def live() -> ServerLiveResponse:
            return ServerLiveResponse(live=True)

        @app.get("/v2/health/ready", response_model=ServerLiveResponse)
        async def server_ready() -> ServerLiveResponse:
            return ServerLiveResponse(live=await step.is_ready())

        @app.get(
            "/v2/models/{model_name}",
            response_model=ModelMetadataResponse,
        )
        async def model_metadata(model_name: str) -> ModelMetadataResponse:
            if model_name != step.name:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown model: {model_name!r}",
                )
            return ModelMetadataResponse(
                name=step.name,
                versions=[step.version],
                platform="moiraweave",
                inputs=step.inputs,
                outputs=step.outputs,
                task=step.task,
                implementation=step.implementation,
            )

        @app.get(
            "/v2/models/{model_name}/ready",
            response_model=ModelReadyResponse,
        )
        async def model_ready(model_name: str) -> ModelReadyResponse:
            if model_name != step.name:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown model: {model_name!r}",
                )
            return ModelReadyResponse(name=step.name, ready=await step.is_ready())

        @app.post(
            "/v2/models/{model_name}/infer",
            response_model=InferResponse,
        )
        async def infer(model_name: str, request: InferRequest) -> InferResponse:
            if model_name != step.name:
                raise HTTPException(
                    status_code=404,
                    detail=f"Unknown model: {model_name!r}",
                )
            return await step.predict(request)

        return app
