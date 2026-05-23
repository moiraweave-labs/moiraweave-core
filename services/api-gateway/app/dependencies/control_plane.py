from typing import Annotated

from fastapi import Depends, Request
from moiraweave_shared.control_plane import ControlPlaneRepository


async def get_control_plane(request: Request) -> ControlPlaneRepository:
    """Return the shared control-plane repository stored on app state."""

    return request.app.state.control_plane  # type: ignore[no-any-return]


ControlPlane = Annotated[ControlPlaneRepository, Depends(get_control_plane)]
