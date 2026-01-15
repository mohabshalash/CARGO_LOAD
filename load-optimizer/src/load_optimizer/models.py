from __future__ import annotations

from pydantic import BaseModel, Field

from typing import Tuple

from typing import Optional

class Container(BaseModel):
    """Container model with dimensions."""

    length: float = Field(gt=0, description="Length of the container")
    width: float = Field(gt=0, description="Width of the container")
    height: float = Field(gt=0, description="Height of the container")
    max_payload: Optional[float] = Field(
        default=None,
        gt=0,
        description="Maximum payload in kg")
class Box(BaseModel):
    """Box model with identifier and dimensions (in meters)."""

    id: str = Field(description="Unique identifier for the box")
    length: float = Field(gt=0, description="Length of the box in meters")
    width: float = Field(gt=0, description="Width of the box in meters")
    height: float = Field(gt=0, description="Height of the box in meters")
    weight: Optional[float] = Field(
        default=None,
        gt=0,
        description="Weight in kg")
class Placement(BaseModel):
    """Placement model representing box position and oriented dimensions."""

    box_id: str = Field(description="Identifier of the placed box")
    x: float = Field(ge=0, description="X coordinate of the box position")
    y: float = Field(ge=0, description="Y coordinate of the box position")
    z: float = Field(ge=0, description="Z coordinate of the box position")

    # Store the ACTUAL placed dimensions after rotation: (L, W, H)
    rotation: Tuple[float, float, float] = Field(
        description="Oriented dimensions (L, W, H) of the placed box"
    )
from pydantic import BaseModel, Field

class PackingResult(BaseModel):
    """Standard result returned by solvers."""
    placements: list["Placement"] = Field(default_factory=list)
    unpacked: list["Box"] = Field(default_factory=list)
    used_volume: float = 0.0
    container_volume: float = 0.0
    fill_rate: float = 0.0

Container.model_rebuild()
Box.model_rebuild()
