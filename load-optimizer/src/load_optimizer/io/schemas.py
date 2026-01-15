"""Data schemas for input/output operations."""

from typing import List, Optional
from pydantic import BaseModel, Field

class ItemSchema(BaseModel):
    """Schema for an item."""
    width: float = Field(gt=0, description="Width of the item")
    height: float = Field(gt=0, description="Height of the item")
    depth: float = Field(gt=0, description="Depth of the item")
    weight: float = Field(ge=0, default=0.0, description="Weight of the item")

class ContainerSchema(BaseModel):
    """Schema for a container."""
    width: float = Field(gt=0, description="Width of the container")
    height: float = Field(gt=0, description="Height of the container")
    depth: float = Field(gt=0, description="Depth of the container")
    max_weight: Optional[float] = Field(None, ge=0, description="Maximum weight capacity")

class PackingRequestSchema(BaseModel):
    """Schema for a packing request."""
    container: ContainerSchema
    items: List[ItemSchema] = Field(min_length=1, description="List of items to pack")

class PackingResultSchema(BaseModel):
    """Schema for a packing result."""
    bins: List[List[int]] = Field(description="List of bins, each containing item indices")
    utilization: float = Field(ge=0, le=1, description="Container utilization ratio")
    num_bins: int = Field(ge=1, description="Number of bins used")







