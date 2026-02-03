"""FastAPI endpoint for load optimizer."""

from __future__ import annotations

import json
import logging
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

import certifi
import httpx
from fastapi import FastAPI, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, APIError
from fastapi import Request
logger = logging.getLogger(__name__)

from load_optimizer.models import Container, Box
from load_optimizer.capacity import max_units_single_sku, allocate_mixed_skus_greedy
from load_optimizer.brokeragent import agent_conversation, calculate_quote, calculate_savings
from pydantic import BaseModel, Field
# Pydantic models for /parse endpoint structured outputs
class ParsedItem(BaseModel):
    """Parsed item from logistics text."""
    sku: str = Field(description="SKU identifier")
    l: float = Field(gt=0, description="Length in cm")
    w: float = Field(gt=0, description="Width in cm")
    h: float = Field(gt=0, description="Height in cm")
    weight: float = Field(gt=0, description="Weight in kg")
    qty: int = Field(gt=0, description="Quantity")


class ParsedContainer(BaseModel):
    """Parsed container from logistics text."""
    type: str = Field(description="Container type (e.g., 20, 20HC, 40, 40HC, 48HC, 53HC)")


class ParsedShipment(BaseModel):
    """Parsed shipment from logistics text."""
    container: ParsedContainer = Field(description="Container information")
    items: list[ParsedItem] = Field(description="List of items")


class ParseResponse(BaseModel):
    """Response model for /parse endpoint."""
    shipment: ParsedShipment = Field(description="Shipment information")


# FastAPI app instance (exactly one)
app = FastAPI(
    title="Load Optimizer API",
    description="Container load optimization service",
)

# CORS middleware applied immediately after app creation
# Allows www.interconnect360.com and interconnect360.com (both with optional port :443)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https://(www\.)?interconnect360\.com(?::\d+)?$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)


@app.get("/_deploy_check")
async def deploy_check(request: Request):
    """Debug endpoint to verify deployment and CORS configuration."""
    return {
        "app": "load-optimizer-api",
        "origin": request.headers.get("origin"),
        "host": request.headers.get("host"),
        "x_forwarded_host": request.headers.get("x-forwarded-host"),
        "x_forwarded_proto": request.headers.get("x-forwarded-proto"),
        "method": request.method,
    }

def build_plan(shipment: dict[str, Any]) -> dict[str, Any]:
    """
    Build plan from shipment data.
    
    Args:
        shipment: Dict with container_preset/container and boxes array
    
    Returns:
        Plan dict with container, summary, single_sku_capacity, mixed_allocations
    """
    from load_optimizer.containers import get_container_dims
    
    # Parse container
    container_kwargs = {}
    if "container_preset" in shipment:
        dims = get_container_dims(shipment["container_preset"])
        container_kwargs.update(dims)
    elif "container" in shipment:
        container_kwargs.update(shipment["container"])
    else:
        raise ValueError("shipment must include either 'container_preset' or 'container'")
    
    if "container" in shipment:
        container_kwargs.update(shipment["container"])
    
    container = Container(**container_kwargs)
    
    # Get items from boxes array
    items = shipment.get("boxes", [])
    
    # Step 1: Compute single_sku_capacity and mixed_allocations
    # Convert items to Box objects for capacity calculations
    items_for_capacity = []
    for item in items:
        if isinstance(item, dict):
            # Create Box from dict
            sku = item.get("sku", item.get("id", "UNKNOWN"))
            length_m, width_m, height_m = None, None, None
            
            if "dims_m" in item:
                dims = item["dims_m"]
                length_m = float(dims.get("L", dims.get("length", 0)))
                width_m = float(dims.get("W", dims.get("width", 0)))
                height_m = float(dims.get("H", dims.get("height", 0)))
            elif "dims_cm" in item:
                dims = item["dims_cm"]
                length_m = float(dims.get("L", dims.get("length", 0))) / 100.0
                width_m = float(dims.get("W", dims.get("width", 0))) / 100.0
                height_m = float(dims.get("H", dims.get("height", 0))) / 100.0
            elif "dims_in" in item:
                dims = item["dims_in"]
                length_m = float(dims.get("L", dims.get("length", 0))) * 0.0254
                width_m = float(dims.get("W", dims.get("width", 0))) * 0.0254
                height_m = float(dims.get("H", dims.get("height", 0))) * 0.0254
            else:
                # Skip if no dimensions
                continue
            
            weight_kg = float(item.get("weight_kg", item.get("weight", 0.0)))
            box = Box(
                id=sku,
                length=length_m,
                width=width_m,
                height=height_m,
                weight=weight_kg if weight_kg > 0 else None,
            )
            items_for_capacity.append(box)
        else:
            items_for_capacity.append(item)
    
    # Compute single_sku_capacity
    single_sku_capacity = [max_units_single_sku(container, item) for item in items_for_capacity]
    
    # Compute mixed_allocations (use original items dict format)
    mixed_allocations_raw = allocate_mixed_skus_greedy(container, items)
    
    # Calculate theoretical maximums and enforce hard caps
    import math
    container_volume = container.length * container.width * container.height
    container_max_weight = container.max_payload if container.max_payload is not None else None
    
    # Build a map of items by SKU for lookup
    items_by_sku = {}
    for item in items:
        if isinstance(item, dict):
            sku = item.get("sku", item.get("id", "UNKNOWN"))
            items_by_sku[sku] = item
    
    # Process allocations with theoretical maximums and hard caps
    mixed_allocations = []
    total_requested_units = 0
    total_allocated_units = 0
    
    for alloc in mixed_allocations_raw:
        sku = alloc["sku"]
        unit_volume = alloc["unit_volume"]
        unit_weight = alloc["unit_weight"]
        
        # Get requested quantity from original item
        item = items_by_sku.get(sku, {})
        requested_units = int(item.get("quantity", item.get("quantity_requested", 0)))
        total_requested_units += requested_units
        
        # Calculate theoretical maximums
        theoretical_max_by_volume = math.floor(container_volume / unit_volume) if unit_volume > 0 else 0
        if container_max_weight is not None and unit_weight > 0:
            theoretical_max_by_weight = math.floor(container_max_weight / unit_weight)
        else:
            theoretical_max_by_weight = float('inf')
        
        # Enforce hard cap: allocated_units <= min(requested, theoretical_max_by_volume, theoretical_max_by_weight)
        allocated_units = alloc["allocated"]
        max_allowed = min(
            requested_units,
            theoretical_max_by_volume,
            theoretical_max_by_weight if theoretical_max_by_weight != float('inf') else requested_units
        )
        allocated_units = min(allocated_units, max_allowed)
        total_allocated_units += allocated_units
        
        # Calculate unloaded units
        unloaded_units = requested_units - allocated_units
        
        # Determine limiting_factor
        if allocated_units == theoretical_max_by_volume and requested_units > allocated_units:
            limiting_factor = "VOLUME"
        elif allocated_units == theoretical_max_by_weight and requested_units > allocated_units:
            limiting_factor = "WEIGHT"
        elif allocated_units < min(theoretical_max_by_volume, theoretical_max_by_weight if theoretical_max_by_weight != float('inf') else theoretical_max_by_volume) and requested_units > allocated_units:
            limiting_factor = "GEOMETRY/PLACEMENT"
        else:
            limiting_factor = "UNKNOWN"
        
        mixed_allocations.append({
            "sku": sku,
            "allocated_qty": allocated_units,
            "requested_units": requested_units,
            "allocated_units": allocated_units,
            "unloaded_units": unloaded_units,
            "limiting_factor": limiting_factor,
            "theoretical_max_by_volume": theoretical_max_by_volume,
            "theoretical_max_by_weight": theoretical_max_by_weight if theoretical_max_by_weight != float('inf') else None,
            "unit_volume": unit_volume,
            "unit_weight": unit_weight,
        })
    
    # Step 2: Compute summary from mixed_allocations (with capped allocated_units)
    used_volume = sum(a["allocated_units"] * a["unit_volume"] for a in mixed_allocations)
    volume_fill_rate = min(used_volume / container_volume, 1.0) if container_volume > 0 else 0.0
    used_weight = sum(a["allocated_units"] * a["unit_weight"] for a in mixed_allocations)
    if container_max_weight is not None and container_max_weight > 0:
        weight_fill_rate = min(used_weight / container_max_weight, 1.0)
    else:
        weight_fill_rate = 0.0
    remaining_volume = container_volume - used_volume
    remaining_weight = container_max_weight - used_weight if container_max_weight is not None else None
    
    # Step 3: Generate actual placements by calling pack_boxes with allocated quantities
    boxes_for_packing = []
    for alloc in mixed_allocations:
        sku = alloc["sku"]
        allocated_qty = alloc["allocated_units"]
        
        # Find the Box template for this SKU
        box_template = None
        for item_box in items_for_capacity:
            if item_box.id == sku or (sku in item_box.id and item_box.id.startswith(sku)):
                box_template = item_box
                break
        
        if box_template is None:
            # Fallback: try to find in items_by_sku and create Box
            item = items_by_sku.get(sku, {})
            if item:
                # Use the same conversion logic as above
                length_m, width_m, height_m = None, None, None
                if "dims_m" in item:
                    dims = item["dims_m"]
                    length_m = float(dims.get("L", dims.get("length", 0)))
                    width_m = float(dims.get("W", dims.get("width", 0)))
                    height_m = float(dims.get("H", dims.get("height", 0)))
                elif "dims_cm" in item:
                    dims = item["dims_cm"]
                    length_m = float(dims.get("L", dims.get("length", 0))) / 100.0
                    width_m = float(dims.get("W", dims.get("width", 0))) / 100.0
                    height_m = float(dims.get("H", dims.get("height", 0))) / 100.0
                elif "dims_in" in item:
                    dims = item["dims_in"]
                    length_m = float(dims.get("L", dims.get("length", 0))) * 0.0254
                    width_m = float(dims.get("W", dims.get("width", 0))) * 0.0254
                    height_m = float(dims.get("H", dims.get("height", 0))) * 0.0254
                
                if length_m and width_m and height_m:
                    weight_kg = float(item.get("weight_kg", item.get("weight", 0.0)))
                    box_template = Box(
                        id=sku,
                        length=length_m,
                        width=width_m,
                        height=height_m,
                        weight=weight_kg if weight_kg > 0 else None,
                    )
        
        # Generate boxes for packing
        if box_template:
            for i in range(allocated_qty):
                box = Box(
                    id=f"{sku}_{i:04d}",
                    length=box_template.length,
                    width=box_template.width,
                    height=box_template.height,
                    weight=box_template.weight,
                )
                boxes_for_packing.append(box)
    
    # Call pack_boxes to get actual placements
    from load_optimizer.packing.first_fit import pack_boxes
    packing_result = pack_boxes(container, boxes_for_packing)
    
    # Never store Placement objects in response dicts (not JSON serializable).
    # Keep raw placements ONLY in a local variable.
    raw_placements = packing_result.placements
    
    # Step 4: Build plan dict (DO NOT include raw Placement objects - they're not JSON serializable)
    plan = {
        "container": {
            "length": container.length,
            "width": container.width,
            "height": container.height,
            "max_weight": container.max_payload,
        },
        "summary": {
            "container_volume": container_volume,
            "used_volume": used_volume,
            "volume_fill_rate": volume_fill_rate,
            "used_weight": used_weight,
            "weight_fill_rate": weight_fill_rate,
            "remaining_volume": remaining_volume,
            "remaining_weight": remaining_weight,
            "requested_units": total_requested_units,
            "allocated_units": total_allocated_units,
            "unloaded_units": total_requested_units - total_allocated_units,
        },
        "single_sku_capacity": single_sku_capacity,
        "mixed_allocations": mixed_allocations,
    }
    
    return plan, raw_placements


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize incoming Wix payload to optimizer format.
    
    Input:
        {
            "shipment": {
                "container": { "type": "40HC" },
                "items": [
                    { "sku": "A", "l": 50, "w": 40, "h": 30, "weight": 18, "qty": 10 }
                ]
            }
        }
    
    Output:
        {
            "container": {
                "length": <number>,
                "width": <number>,
                "height": <number>,
                "max_weight": <number>
            },
            "boxes": [
                {
                    "sku": "A",
                    "dims_cm": { "L": 50, "W": 40, "H": 30 },
                    "weight_kg": 18,
                    "quantity": 10
                }
            ]
        }
    """
    # Container preset map (hardcoded)
    # Values in mm, convert to meters for optimizer
    CONTAINER_PRESETS = {
        "40HC": {
            "length": 12032.0 / 1000.0,  # Convert mm to meters
            "width": 2352.0 / 1000.0,
            "height": 2698.0 / 1000.0,
            "max_payload": 26500.0,
        }
    }
    
    # Extract shipment from payload
    if "shipment" in payload:
        shipment_data = payload["shipment"]
    else:
        shipment_data = payload
    
    normalized = {}
    
    # Normalize container
    container_data = shipment_data.get("container", {})
    container_type = container_data.get("type", "")
    
    if container_type in CONTAINER_PRESETS:
        preset = CONTAINER_PRESETS[container_type]
        normalized["container"] = {
            "length": preset["length"],
            "width": preset["width"],
            "height": preset["height"],
            "max_payload": preset["max_payload"],
        }
    else:
        # Fallback: use container data as-is if no preset match
        normalized["container"] = container_data
    
    # Normalize items to boxes format
    items = shipment_data.get("items", [])
    boxes = []
    for item in items:
        # Convert l/w/h to dims_cm (assuming input is in cm)
        box = {
            "sku": item.get("sku", item.get("id", "UNKNOWN")),
            "dims_cm": {
                "L": float(item.get("l", item.get("length", 0))),
                "W": float(item.get("w", item.get("width", 0))),
                "H": float(item.get("h", item.get("height", 0))),
            },
            "weight_kg": float(item.get("weight", item.get("weight_kg", 0.0))),
            "quantity": int(item.get("qty", item.get("quantity", 1))),
        }
        boxes.append(box)
    
    normalized["boxes"] = boxes
    
    return normalized


def validate_input(request: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """
    Validate input and return normalized shipment and list of missing fields.
    
    Returns:
        (normalized_shipment, missing_fields)
        If missing_fields is non-empty, return friendly error.
    """
    missing_fields = []
    
    # Extract shipment
    if "shipment" in request:
        shipment_data = request["shipment"]
    else:
        shipment_data = request
    
    # Check container
    container_data = shipment_data.get("container", {})
    if not container_data:
        missing_fields.append("Container size (length, width, height)")
    else:
        container_type = container_data.get("type", "")
        if not container_type:
            # Check if explicit dimensions provided
            if not all(k in container_data for k in ["length", "width", "height"]):
                missing_fields.append("Container size (length, width, height)")
    
    # Check items
    items = shipment_data.get("items", [])
    if not items:
        missing_fields.append("Items to optimize")
    else:
        for i, item in enumerate(items):
            # Check dimensions
            has_dims = any(
                k in item for k in ["l", "length", "w", "width", "h", "height", 
                                   "dims_cm", "dims_m", "dims_in"]
            )
            if not has_dims:
                missing_fields.append(f"Item {i+1} dimensions")
            
            # Check quantity (optional but recommended)
            has_qty = any(k in item for k in ["qty", "quantity"])
            if not has_qty:
                # Not required, but we'll note it
                pass
    
    # Normalize if no critical missing fields
    if not missing_fields:
        try:
            normalized = normalize_payload(request)
            return normalized, []
        except Exception:
            # If normalization fails, treat as missing info
            missing_fields.append("Item dimensions")
            missing_fields.append("Quantity per item")
    
    return {}, missing_fields


def build_placements_render(raw_placements: list) -> list[dict[str, Any]]:
    """
    Build placements_render from raw Placement objects (JSON primitives only).
    
    Args:
        raw_placements: List of Placement objects
        
    Returns:
        List of dicts with x, y, z, dims (all JSON-serializable primitives)
    """
    from load_optimizer.models import Placement
    
    placements_render = []
    
    for placement in raw_placements:
        # Extract dimensions: prefer rotation tuple, fallback to length/width/height
        dims = [0.0, 0.0, 0.0]
        
        if isinstance(placement, Placement):
            # Extract rotation dimensions (L, W, H)
            rotation = placement.rotation
            if isinstance(rotation, (tuple, list)) and len(rotation) == 3:
                dims = [float(rotation[0]), float(rotation[1]), float(rotation[2])]
            elif hasattr(placement, "length") and hasattr(placement, "width") and hasattr(placement, "height"):
                dims = [float(placement.length), float(placement.width), float(placement.height)]
            
            placement_dict = {
                "x": float(placement.x),
                "y": float(placement.y),
                "z": float(placement.z),
                "dims": dims,
            }
            placements_render.append(placement_dict)
        elif isinstance(placement, dict):
            # Handle dict format
            rotation = placement.get("rotation")
            if isinstance(rotation, (tuple, list)) and len(rotation) == 3:
                dims = [float(rotation[0]), float(rotation[1]), float(rotation[2])]
            else:
                # Fallback to length/width/height
                dims = [
                    float(placement.get("length", placement.get("L", 0))),
                    float(placement.get("width", placement.get("W", 0))),
                    float(placement.get("height", placement.get("H", 0))),
                ]
            
            placement_dict = {
                "x": float(placement.get("x", 0)),
                "y": float(placement.get("y", 0)),
                "z": float(placement.get("z", 0)),
                "dims": dims,
            }
            placements_render.append(placement_dict)
    
    return placements_render


def extract_render_data(plan: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """
    Extract lightweight rendering data from plan.
    
    Returns:
        (placements_render, container_render)
        placements_render: List of lightweight placement dicts with x, y, z, dims (all JSON primitives)
        container_render: Dict with L, W, H
    """
    from load_optimizer.models import Placement
    
    placements_render = []
    container_render = {}
    
    # Extract container dimensions
    container = plan.get("container", {})
    if container:
        container_render = {
            "L": float(container.get("length", 0)),
            "W": float(container.get("width", 0)),
            "H": float(container.get("height", 0)),
        }
    
    # Extract placements from plan if they exist
    # Check multiple possible locations for placements
    placements = []
    if "_packing_result_placements" in plan:
        # Use temporary key to access placements without including them in JSON
        placements = plan["_packing_result_placements"]
    elif "placements" in plan:
        placements = plan["placements"]
    elif "packing_result" in plan:
        packing_result = plan["packing_result"]
        if hasattr(packing_result, "placements"):
            placements = packing_result.placements
        elif isinstance(packing_result, dict) and "placements" in packing_result:
            placements = packing_result["placements"]
    
    # Convert Placement objects to lightweight dicts with JSON-serializable primitives only
    for placement in placements:
        # Extract dimensions: prefer rotation tuple, fallback to length/width/height
        dims = [0.0, 0.0, 0.0]
        
        if isinstance(placement, Placement):
            # Extract rotation dimensions (L, W, H)
            rotation = placement.rotation
            if isinstance(rotation, (tuple, list)) and len(rotation) == 3:
                dims = [float(rotation[0]), float(rotation[1]), float(rotation[2])]
            elif hasattr(placement, "length") and hasattr(placement, "width") and hasattr(placement, "height"):
                dims = [float(placement.length), float(placement.width), float(placement.height)]
            
            placement_dict = {
                "x": float(placement.x),
                "y": float(placement.y),
                "z": float(placement.z),
                "dims": dims,
            }
            placements_render.append(placement_dict)
        elif isinstance(placement, dict):
            # Handle dict format
            rotation = placement.get("rotation")
            if isinstance(rotation, (tuple, list)) and len(rotation) == 3:
                dims = [float(rotation[0]), float(rotation[1]), float(rotation[2])]
            else:
                # Fallback to length/width/height
                dims = [
                    float(placement.get("length", placement.get("L", 0))),
                    float(placement.get("width", placement.get("W", 0))),
                    float(placement.get("height", placement.get("H", 0))),
                ]
            
            placement_dict = {
                "x": float(placement.get("x", 0)),
                "y": float(placement.get("y", 0)),
                "z": float(placement.get("z", 0)),
                "dims": dims,
            }
            placements_render.append(placement_dict)
    
    # Safeguard: Ensure all values are JSON primitives (float/int)
    for p in placements_render:
        assert isinstance(p["x"], (int, float)), f"x must be number, got {type(p['x'])}"
        assert isinstance(p["y"], (int, float)), f"y must be number, got {type(p['y'])}"
        assert isinstance(p["z"], (int, float)), f"z must be number, got {type(p['z'])}"
        assert isinstance(p["dims"], list) and len(p["dims"]) == 3, "dims must be list of 3 numbers"
        assert all(isinstance(d, (int, float)) for d in p["dims"]), "dims elements must be numbers"
    
    return placements_render, container_render


def format_output(plan: dict[str, Any], include_render: bool = False, raw_placements: list | None = None) -> dict[str, Any]:
    """
    Format plan output with guaranteed fields and user-friendly summary.
    """
    summary = plan.get("summary", {})
    mixed_allocations = plan.get("mixed_allocations", [])
    
    # Calculate units
    requested_units = summary.get("requested_units", 0)
    allocated_units = summary.get("allocated_units", 0)
    units_loaded = int(allocated_units)
    units_unloaded = max(0, int(requested_units - allocated_units))
    
    # Get fill rates
    volume_fill_rate = summary.get("volume_fill_rate", 0.0)
    weight_fill_rate = summary.get("weight_fill_rate", None)
    has_weight = plan.get("container", {}).get("max_weight") is not None
    
    volume_fill_pct = round(volume_fill_rate * 100.0, 1)
    weight_fill_pct = round(weight_fill_rate * 100.0, 1) if weight_fill_rate is not None else None
    
    # Determine limiting factor
    limiting_factor = "none"
    limiting_reason = "All requested units were successfully loaded."
    
    if volume_fill_pct >= 99.5:  # ≈ 100%
        limiting_factor = "volume"
        limiting_reason = "Container volume was fully utilized."
    elif has_weight and weight_fill_pct is not None and weight_fill_pct >= 99.5:
        limiting_factor = "weight"
        limiting_reason = "Container weight capacity was fully utilized."
    elif units_unloaded > 0:
        # Check if it's a count limit or dimensions
        limiting_factor = "dimensions"
        limiting_reason = "Some items could not fit due to their dimensions."
    
    # Build summary string
    weight_line = f"⚖️ Weight Fill: {weight_fill_pct:.1f}%" if weight_fill_pct is not None else "⚖️ Weight Fill: N/A"
    summary_text = f"""🚢 Optimization Complete
📦 Volume Fill: {volume_fill_pct:.1f}%
{weight_line}
✅ Units Loaded: {units_loaded}
❌ Units Unloaded: {units_unloaded}
🔎 Limiting Factor: {limiting_factor} — {limiting_reason}"""
    
    # Build response with guaranteed fields
    response = {
        "metrics": {
            "units_loaded": units_loaded,
            "units_unloaded": units_unloaded,
            "limiting_factor": limiting_factor,
            "limiting_reason": limiting_reason,
        },
        "summary": summary_text,
        "plan": plan,  # Keep full plan for backward compatibility (no Placement objects)
    }
    
    # Add rendering data if requested
    if include_render:
        # Build placements_render from raw_placements (never store Placement objects in response dicts)
        placements_render = build_placements_render(raw_placements) if raw_placements else []
        
        # Get container_render from plan
        container_render = {}
        container = plan.get("container", {})
        if container:
            container_render = {
                "L": float(container.get("length", 0)),
                "W": float(container.get("width", 0)),
                "H": float(container.get("height", 0)),
            }
        
        # Always include placements_render when render=1 (empty [] if no placements)
        response["placements_render"] = placements_render
        # Add debug placements_count
        response["placements_count"] = len(placements_render)
        if container_render:
            response["container_render"] = container_render
    
    return response


# B) Add endpoint: POST /optimize
@app.post("/optimize")
async def optimize(
    request: dict[str, Any],
    render: int = Query(0, description="Include rendering data (1) or not (0)")
) -> dict[str, Any]:
    """
    Optimize shipment and return plan.
    
    Input (request body):
        {
            "shipment": {
                "container": { "type": "40HC" },
                "items": [
                    { "sku": "A", "l": 50, "w": 40, "h": 30, "weight": 18, "qty": 10 }
                ]
            }
        }
    
    Returns:
        Response with metrics, summary, and plan
    """
    try:
        # Validate input
        normalized_shipment, missing_fields = validate_input(request)
        
        if missing_fields:
            # Return friendly 422 error
            error_response = {
                "error": "MISSING_INFORMATION",
                "summary": "⚠️ Missing information\nPlease enter the missing details to run the optimization.",
                "details": missing_fields,
            }
            return Response(
                content=json.dumps(error_response),
                status_code=422,
                media_type="application/json"
            )
        
        # Build plan
        # Never store Placement objects in response dicts (not JSON serializable).
        plan, raw_placements = build_plan(normalized_shipment)
        
        # Write plan.json using absolute path
        plan_path = Path(__file__).parent / "plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, sort_keys=True)
        
        # Check if render is requested (query param or request body flag)
        include_render = render == 1 or request.get("render") == 1
        
        # Format output
        response = format_output(plan, include_render=include_render, raw_placements=raw_placements if include_render else None)
        
        # Log one concise line
        logger.info(
            f"loaded_units={response['metrics']['units_loaded']}, "
            f"unloaded_units={response['metrics']['units_unloaded']}, "
            f"limiting_factor={response['metrics']['limiting_factor']}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"ERROR in /optimize endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


# D) Add endpoint: POST /parse
@app.post("/parse")
async def parse(request: dict[str, Any]) -> dict[str, Any]:
    """
    Parse free-text logistics description into structured shipment JSON.
    
    Input (request body):
        {
            "text": "600 cartons 50x40x30 cm, 18kg each, 40HC"
        }
    
    Returns:
        {
            "shipment": {
                "container": { "type": "40HC" },
                "items": [
                    { "sku": "A", "l": 50, "w": 40, "h": 30, "weight": 18, "qty": 600 }
                ]
            }
        }
    """
    try:
        # Get text from request
        text = request.get("text", "")
        if not text or not isinstance(text, str):
            raise HTTPException(status_code=400, detail="Missing or invalid 'text' field in request")
        
        # Read OPENAI_API_KEY from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set on server")
        
        # Initialize OpenAI client with httpx client for CA bundle and timeout
        http_client = httpx.Client(timeout=30.0, verify=certifi.where())
        client = OpenAI(api_key=api_key, http_client=http_client)
        
        # System instruction
        system_instruction = """You are a logistics parser. Parse free-text shipment descriptions into structured JSON.

Examples:
- "600 cartons 50x40x30 cm, 18kg each, 40HC" → container type: "40HC", items: [{sku: "A", l: 50, w: 40, h: 30, weight: 18, qty: 600}]
- "200 boxes 60x50x40, 25kg, 20HC container" → container type: "20HC", items: [{sku: "A", l: 60, w: 50, h: 40, weight: 25, qty: 200}]

Rules:
- Default units: cm for dimensions, kg for weight (unless explicitly stated otherwise)
- Extract container type from text (20, 20HC, 40, 40HC, 48HC, 53HC)
- Extract dimensions (length x width x height)
- Extract weight per unit
- Extract quantity
- Use simple SKU identifiers (A, B, C, etc.) if not specified
- Return ONLY valid JSON matching the schema, no prose or explanations"""
        
        # Call OpenAI Responses API with structured outputs (with retries for connection/timeout)
        response = None
        last_error = None
        backoff_delays = [1.0, 2.0]  # 1s, 2s backoff
        
        for attempt in range(3):  # Initial attempt + 2 retries
            try:
                response = client.responses.parse(
                    model="gpt-4o-mini",
                    input=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": text}
                    ],
                    text_format=ParseResponse,
                    temperature=0.0  # Deterministic output
                )
                break  # Success, exit retry loop
            except (APIConnectionError, APITimeoutError) as e:
                last_error = e
                if attempt < 2:  # Retry if not last attempt
                    delay = backoff_delays[attempt]
                    print(f"DEBUG /parse: OpenAI connection/timeout error (attempt {attempt + 1}/3), retrying after {delay}s...")
                    print(f"DEBUG /parse: Exception type: {type(e).__name__}, message: {str(e)}")
                    time.sleep(delay)
                else:
                    # Last attempt failed, will be handled below
                    break
            except Exception as e:
                # Non-retryable error, break immediately
                last_error = e
                break
        
        if response is None:
            # All retries exhausted or non-retryable error
            raise last_error if last_error else Exception("OpenAI API call failed with no error")
        
        # Extract parsed result from response
        # Responses.parse() returns parsed Pydantic model in output_parsed
        if not hasattr(response, 'output_parsed') or not response.output_parsed:
            raise ValueError("OpenAI response has no output_parsed")
        parsed_result = response.output_parsed.model_dump()
        
        # Validation: check items not empty and container type present
        shipment = parsed_result.get("shipment", {})
        container = shipment.get("container", {})
        container_type = container.get("type", "")
        items = shipment.get("items", [])
        
        if not container_type:
            raise HTTPException(
                status_code=400,
                detail="Parsed result missing container type"
            )
        
        if not items or len(items) == 0:
            raise HTTPException(
                status_code=400,
                detail="Parsed result has no items"
            )
        
        # Return parsed shipment JSON
        return parsed_result
        
    except HTTPException:
        raise
    except Exception as e:
        # Print detailed error information to terminal
        print("ERROR in /parse endpoint:")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {str(e)}")
        print(f"Exception repr: {repr(e)}")
        print("Full traceback:")
        traceback.print_exc()
        
        # Return HTTP 500 with exception class name and message (truncated to 300 chars)
        error_class = type(e).__name__
        error_msg = str(e)
        error_detail = f"{error_class}: {error_msg}"
        if len(error_detail) > 300:
            error_detail = error_detail[:297] + "..."
        raise HTTPException(status_code=500, detail=error_detail)


# BrokerAgent endpoints
@app.post("/agent")
async def agent(request: dict[str, Any]) -> dict[str, Any]:
    """
    BrokerAgent conversational endpoint for collecting shipment data.
    Uses DB-backed session memory (sessions, messages, session_state).

    Session handling:
    - If request includes session_id, reuse it (do not generate a new one).
    - If missing, generate a UUID session_id exactly once and return it.
    - Upsert into sessions (insert if missing; update last_active_at if exists).
    - Insert user message and assistant message into messages with the same session_id.
    - Always return JSON containing session_id and assistant_message.

    Input (request body):
        {
            "session_id": "optional-unique-session-id",
            "message": "I need to ship 100 cartons from New York to London"
        }

    Returns:
        {
            "session_id": str,
            "assistant_message": str,
            "draft_json": {...},
            "missing_fields": [...],
            "ready_for_quote": bool,
            "ready_for_optimize": bool,
            "ready_for_savings": bool,
        }
    """
    from load_optimizer.db import get_conn, put_conn

    try:
        # Reuse provided session_id; generate exactly once if missing
        session_id = request.get("session_id")
        if not session_id or not isinstance(session_id, str) or not session_id.strip():
            session_id = str(uuid.uuid4())

        message = request.get("message", "")
        if not message or not isinstance(message, str):
            raise HTTPException(status_code=400, detail="message must be a non-empty string")

        conn = None
        try:
            conn = get_conn()
            cur = conn.cursor()

            # Upsert sessions: insert if missing, update last_active_at if exists
            cur.execute(
                """
                INSERT INTO sessions (session_id, last_active_at)
                VALUES (%s, now())
                ON CONFLICT (session_id) DO UPDATE SET last_active_at = now()
                """,
                (session_id,),
            )
            # Insert user message (same session_id)
            cur.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (%s, 'user', %s)",
                (session_id, message),
            )

            # Load last 20 messages (chronological: DESC LIMIT 20 then reverse)
            cur.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = %s
                ORDER BY created_at DESC
                LIMIT 20
                """,
                (session_id,),
            )
            rows = cur.fetchall()
            recent_messages = [{"role": r[0], "content": r[1] or ""} for r in reversed(rows)]

            # Load all session_state for this session
            cur.execute(
                "SELECT key, value FROM session_state WHERE session_id = %s",
                (session_id,),
            )
            session_state_rows = cur.fetchall()
            session_state = {}
            for k, v in session_state_rows:
                session_state[k] = v if isinstance(v, dict) else (json.loads(v) if isinstance(v, str) else v)

            cur.close()
            put_conn(conn)
            conn = None
        finally:
            if conn is not None:
                put_conn(conn)

        result = await agent_conversation(session_id, message, recent_messages, session_state)

        # Persist assistant message and updated draft; commit and return connection to pool in finally
        conn = None
        try:
            conn = get_conn()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO messages (session_id, role, content) VALUES (%s, 'assistant', %s)",
                (session_id, result["assistant_message"]),
            )
            draft = result.get("draft_json") or {}
            cur.execute(
                """
                INSERT INTO session_state (session_id, key, value, updated_at)
                VALUES (%s, 'draft', %s::jsonb, now())
                ON CONFLICT (session_id, key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()
                """,
                (session_id, json.dumps(draft) if isinstance(draft, dict) else draft),
            )
            cur.execute(
                "UPDATE sessions SET last_active_at = now() WHERE session_id = %s",
                (session_id,),
            )
            conn.commit()
            cur.close()
            put_conn(conn)
            conn = None
        finally:
            if conn is not None:
                put_conn(conn)

        # Always return JSON with session_id and assistant_message
        return {
            "session_id": session_id,
            "assistant_message": result["assistant_message"],
            "draft_json": result.get("draft_json"),
            "missing_fields": result.get("missing_fields", []),
            "ready_for_quote": result.get("ready_for_quote"),
            "ready_for_optimize": result.get("ready_for_optimize"),
            "ready_for_savings": result.get("ready_for_savings"),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent endpoint error: {repr(e)}", exc_info=True)
        error_detail = str(e)[:300] if str(e) else "Unknown error"
        raise HTTPException(status_code=500, detail=f"Agent endpoint failed: {error_detail}")


@app.post("/quote")
async def quote(request: dict[str, Any]) -> dict[str, Any]:
    """
    Calculate baseline quote based on draft_json.
    
    Input (request body):
        {
            "draft_json": {
                "origin_country": "USA",
                "origin_city": "New York",
                "destination_country": "UK",
                "destination_city": "London",
                "container_type": "40HC",
                "quantity": 100,
                "unit_weight": 18.0,
                "unit_weight_unit": "kg",
                "unit_length": 50.0,
                "unit_width": 40.0,
                "unit_height": 30.0,
                "unit_dimension_unit": "cm",
                "product_type": "cartons"
            }
        }
    
    Returns:
        {
            "total_usd": float,
            "per_unit_usd": float,
            "breakdown": [{"label": str, "usd": float}]
        }
    """
    try:
        draft_json = request.get("draft_json")
        if not draft_json or not isinstance(draft_json, dict):
            raise HTTPException(status_code=400, detail="draft_json is required and must be a dict")
        
        result = calculate_quote(draft_json)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quote endpoint error: {repr(e)}", exc_info=True)
        error_detail = str(e)[:300] if str(e) else "Unknown error"
        raise HTTPException(status_code=500, detail=f"Quote calculation failed: {error_detail}")


@app.post("/savings")
async def savings(request: dict[str, Any]) -> dict[str, Any]:
    """
    Calculate CargoPool savings based on optimize_result.
    
    Input (request body):
        {
            "draft_json": {...},
            "quote_result": {"total_usd": 3800.0, "per_unit_usd": 38.0, ...},
            "optimize_result": {"metrics": {"volume_fill_rate": 0.75, "weight_fill_rate": 0.60}, ...}
        }
    
    Returns:
        {
            "savings_pct": float,
            "baseline_per_unit": float,
            "pooled_per_unit": float,
            "explanation": str
        }
    """
    try:
        draft_json = request.get("draft_json")
        quote_result = request.get("quote_result")
        optimize_result = request.get("optimize_result")
        
        if not draft_json or not isinstance(draft_json, dict):
            raise HTTPException(status_code=400, detail="draft_json is required and must be a dict")
        if not quote_result or not isinstance(quote_result, dict):
            raise HTTPException(status_code=400, detail="quote_result is required and must be a dict")
        if not optimize_result or not isinstance(optimize_result, dict):
            raise HTTPException(status_code=400, detail="optimize_result is required and must be a dict")
        
        result = calculate_savings(draft_json, quote_result, optimize_result)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Savings endpoint error: {repr(e)}", exc_info=True)
        error_detail = str(e)[:300] if str(e) else "Unknown error"
        raise HTTPException(status_code=500, detail=f"Savings calculation failed: {error_detail}")


@app.get("/debug/session/{session_id}")
async def debug_session(session_id: str) -> dict[str, Any]:
    """Return session row, last 20 messages, and all session_state for testing."""
    from load_optimizer.db import get_conn, put_conn

    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()

        cur.execute(
            "SELECT id, session_id, user_id, created_at, last_active_at, metadata FROM sessions WHERE session_id = %s",
            (session_id,),
        )
        row = cur.fetchone()
        session_row = None
        if row:
            session_row = {
                "id": row[0],
                "session_id": row[1],
                "user_id": row[2],
                "created_at": str(row[3]) if row[3] else None,
                "last_active_at": str(row[4]) if row[4] else None,
                "metadata": row[5],
            }

        cur.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE session_id = %s
            ORDER BY created_at DESC
            LIMIT 20
            """,
            (session_id,),
        )
        msg_rows = cur.fetchall()
        last_20_messages = [
            {"role": r[0], "content": r[1], "created_at": str(r[2]) if r[2] else None}
            for r in reversed(msg_rows)
        ]

        cur.execute(
            "SELECT key, value, updated_at FROM session_state WHERE session_id = %s",
            (session_id,),
        )
        state_rows = cur.fetchall()
        session_state = {}
        for r in state_rows:
            val = r[1]
            if isinstance(val, str):
                try:
                    val = json.loads(val)
                except Exception:
                    pass
            session_state[r[0]] = {"value": val, "updated_at": str(r[2]) if r[2] else None}

        cur.close()
        put_conn(conn)
        return {
            "session": session_row,
            "last_20_messages": last_20_messages,
            "session_state": session_state,
        }
    except Exception as e:
        if conn is not None:
            put_conn(conn)
        logger.warning(f"Debug session failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# C) Add health check: GET /health
@app.get("/health")
async def health() -> dict[str, Any]:
    """Health check endpoint."""
    return {
        "ok": True,
        "has_openai_key": bool(os.getenv("OPENAI_API_KEY"))
    }


@app.get("/health/db")
async def health_db() -> dict[str, Any]:
    """Check DB connectivity via connection pool (SELECT 1)."""
    from load_optimizer.db import get_conn, put_conn
    conn = None
    try:
        conn = get_conn()
        cur = conn.cursor()
        cur.execute("SELECT 1")
        cur.close()
        return {"db": "ok"}
    except Exception as e:
        logger.warning(f"Health DB check failed: {e}")
        raise HTTPException(status_code=503, detail="Database unavailable")
    finally:
        if conn is not None:
            put_conn(conn)

