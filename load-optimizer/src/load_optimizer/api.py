"""FastAPI endpoint for load optimizer."""

from __future__ import annotations

import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import certifi
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, APIError

from load_optimizer.models import Container, Box
from load_optimizer.capacity import max_units_single_sku, allocate_mixed_skus_greedy
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


app = FastAPI()

# A) Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://platform.interconnect360.com"],  # Temporarily allow all origins (replace with Wix domain later)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    
    # Map allocated to allocated_qty for summary calculation
    mixed_allocations = []
    for alloc in mixed_allocations_raw:
        mixed_allocations.append({
            "sku": alloc["sku"],
            "allocated_qty": alloc["allocated"],
            "limiting_factor": alloc["limiting_factor"],
            "unit_volume": alloc["unit_volume"],
            "unit_weight": alloc["unit_weight"],
        })
    
    # Step 2: Compute summary from mixed_allocations
    container_volume = container.length * container.width * container.height
    used_volume = sum(a["allocated_qty"] * a["unit_volume"] for a in mixed_allocations)
    volume_fill_rate = used_volume / container_volume if container_volume > 0 else 0.0
    used_weight = sum(a["allocated_qty"] * a["unit_weight"] for a in mixed_allocations)
    container_max_weight = container.max_payload if container.max_payload is not None else 0.0
    weight_fill_rate = used_weight / container_max_weight if container_max_weight > 0 else 0.0
    remaining_volume = container_volume - used_volume
    remaining_weight = container_max_weight - used_weight if container.max_payload is not None else None
    total_allocated_units = sum(a["allocated_qty"] for a in mixed_allocations)
    
    # Step 3: Build plan dict
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
            "total_allocated_units": total_allocated_units,
        },
        "single_sku_capacity": single_sku_capacity,
        "mixed_allocations": mixed_allocations,
    }
    
    return plan


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


# B) Add endpoint: POST /optimize
@app.post("/optimize")
async def optimize(request: dict[str, Any]) -> dict[str, Any]:
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
        Plan dict with container, summary, single_sku_capacity, mixed_allocations
    """
    try:
        # Task 2: DEBUG prints
        print("DEBUG /optimize called")
        print(f"DEBUG incoming payload keys: {list(request.keys())}")
        
        # Task 3: Normalize incoming payload
        normalized_shipment = normalize_payload(request)
        print(f"DEBUG normalized shipment keys: {list(normalized_shipment.keys())}")
        
        # Task 4: Build plan
        plan = build_plan(normalized_shipment)
        
        # Task 4: Write plan.json using absolute path
        import json
        plan_path = Path(__file__).parent / "plan.json"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2, sort_keys=True)
        file_size = plan_path.stat().st_size
        print(f"DEBUG plan.json written to: {plan_path}")
        print(f"DEBUG plan.json file size: {file_size} bytes")
        
        # Task 5: Return plan JSON
        return plan
        
    except Exception as e:
        # Task 1: Error handling
        print("ERROR in /optimize endpoint:")
        traceback.print_exc()
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
        
        # Initialize OpenAI client with httpx client for CA bundle and timeout
        http_client = httpx.Client(timeout=30.0, verify=certifi.where())
        client = OpenAI(http_client=http_client)
        
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


# C) Add health check: GET /health
@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}

