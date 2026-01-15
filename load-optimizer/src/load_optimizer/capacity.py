"""Capacity analysis: compute theoretical and feasible max cases per container."""

from __future__ import annotations

import math
from typing import Any

from load_optimizer.models import Container, Box
from load_optimizer.packing.first_fit import pack_boxes


def compute_max_cases_theoretical(
    container: Container,
    products: list[dict[str, Any]],
    safety: float = 0.95,
) -> dict[str, Any]:
    """
    Compute theoretical maximum cases per SKU based on volume and weight constraints.
    
    For each product:
    - Compute case_volume_m3 from case_dims_cm (cm->m) or dims_in (in->m) or dims_m
    - q_vol = floor(container_volume * safety / case_volume)
    - q_wt = floor(container.max_payload / case_weight) if max_payload else very_large
    - q_max = min(q_vol, q_wt, product.max_cases if present else very_large)
    
    Args:
        container: Container with dimensions and optional max_payload
        products: List of product dicts with sku, case_dims_cm/dims_in/dims_m, case_weight_kg/weight_kg, optional max_cases
        safety: Safety factor for volume calculation (default 0.95)
    
    Returns:
        dict with:
            - sku -> q_max (theoretical max cases)
            - details: dict[sku, dict] with q_vol, q_wt, q_max, case_volume_m3
    """
    container_volume = container.length * container.width * container.height
    very_large = 1_000_000
    
    results: dict[str, int] = {}
    details: dict[str, dict[str, Any]] = {}
    
    for product in products:
        sku = product.get("sku", product.get("id", "UNKNOWN"))
        
        # Extract case dimensions and convert to meters
        case_volume_m3 = None
        if "case_volume_m3" in product:
            case_volume_m3 = float(product["case_volume_m3"])
        elif "case_dims_cm" in product:
            dims = product["case_dims_cm"]
            L_cm = float(dims.get("L", dims.get("length", 0)))
            W_cm = float(dims.get("W", dims.get("width", 0)))
            H_cm = float(dims.get("H", dims.get("height", 0)))
            # Convert cm³ to m³
            case_volume_m3 = (L_cm * W_cm * H_cm) / 1_000_000.0
        elif "dims_cm" in product:
            dims = product["dims_cm"]
            L_cm = float(dims.get("L", dims.get("length", 0)))
            W_cm = float(dims.get("W", dims.get("width", 0)))
            H_cm = float(dims.get("H", dims.get("height", 0)))
            case_volume_m3 = (L_cm * W_cm * H_cm) / 1_000_000.0
        elif "dims_in" in product:
            dims = product["dims_in"]
            L_in = float(dims.get("L", dims.get("length", 0)))
            W_in = float(dims.get("W", dims.get("width", 0)))
            H_in = float(dims.get("H", dims.get("height", 0)))
            # Convert inches to meters: in * 0.0254
            L_m = L_in * 0.0254
            W_m = W_in * 0.0254
            H_m = H_in * 0.0254
            case_volume_m3 = L_m * W_m * H_m
        elif "dims_m" in product:
            dims = product["dims_m"]
            L_m = float(dims.get("L", dims.get("length", 0)))
            W_m = float(dims.get("W", dims.get("width", 0)))
            H_m = float(dims.get("H", dims.get("height", 0)))
            case_volume_m3 = L_m * W_m * H_m
        else:
            raise ValueError(f"Product {sku} must have case_volume_m3, case_dims_cm, dims_cm, dims_in, or dims_m")
        
        # Extract case weight
        case_weight_kg = float(product.get("case_weight_kg", product.get("weight_kg", 0.0)))
        
        # Calculate volume-based max
        if case_volume_m3 > 0:
            q_vol = math.floor((container_volume * safety) / case_volume_m3)
        else:
            q_vol = very_large
        
        # Calculate weight-based max
        if container.max_payload is not None and case_weight_kg > 0:
            q_wt = math.floor(container.max_payload / case_weight_kg)
        else:
            q_wt = very_large
        
        # Apply product max_cases constraint if present
        max_cases = product.get("max_cases", very_large)
        if isinstance(max_cases, (int, float)):
            max_cases = int(max_cases)
        else:
            max_cases = very_large
        
        q_max = min(q_vol, q_wt, max_cases)
        
        results[sku] = q_max
        details[sku] = {
            "q_vol": q_vol,
            "q_wt": q_wt,
            "q_max": q_max,
            "case_volume_m3": case_volume_m3,
            "case_weight_kg": case_weight_kg,
        }
    
    return {
        "theoretical_max": results,
        "details": details,
    }


def compute_max_cases_feasible(
    container: Container,
    product: dict[str, Any],
    q_max: int,
) -> int:
    """
    Use binary search to find the maximum feasible quantity that can actually be packed.
    
    For each mid value:
    - Generate mid boxes (same dims/weight)
    - Call pack_boxes(container, boxes)
    - Feasible if len(result.unpacked) == 0
    
    Args:
        container: Container with dimensions and optional max_payload
        product: Product dict with sku, case_dims_cm/dims_in/dims_m, case_weight_kg/weight_kg
        q_max: Theoretical maximum (upper bound for binary search)
    
    Returns:
        Best feasible quantity (0 if none feasible)
    """
    if q_max <= 0:
        return 0
    
    # Extract dimensions and weight for box generation
    sku = product.get("sku", product.get("id", "UNKNOWN"))
    
    # Get dimensions in meters
    length_m, width_m, height_m = None, None, None
    if "case_dims_cm" in product or "dims_cm" in product:
        dims = product.get("case_dims_cm", product.get("dims_cm", {}))
        length_m = float(dims.get("L", dims.get("length", 0))) / 100.0
        width_m = float(dims.get("W", dims.get("width", 0))) / 100.0
        height_m = float(dims.get("H", dims.get("height", 0))) / 100.0
    elif "dims_in" in product:
        dims = product["dims_in"]
        length_m = float(dims.get("L", dims.get("length", 0))) * 0.0254
        width_m = float(dims.get("W", dims.get("width", 0))) * 0.0254
        height_m = float(dims.get("H", dims.get("height", 0))) * 0.0254
    elif "dims_m" in product:
        dims = product["dims_m"]
        length_m = float(dims.get("L", dims.get("length", 0)))
        width_m = float(dims.get("W", dims.get("width", 0)))
        height_m = float(dims.get("H", dims.get("height", 0)))
    else:
        raise ValueError(f"Product {sku} must have case_dims_cm, dims_cm, dims_in, or dims_m")
    
    weight_kg = float(product.get("case_weight_kg", product.get("weight_kg", 0.0)))
    
    # Binary search: find maximum feasible quantity
    left, right = 0, q_max
    best_feasible = 0
    
    while left <= right:
        mid = (left + right) // 2
        
        # Generate mid boxes
        boxes = []
        for i in range(mid):
            boxes.append(Box(
                id=f"{sku}_{i:04d}",
                length=length_m,
                width=width_m,
                height=height_m,
                weight=weight_kg if weight_kg > 0 else None,
            ))
        
        # Test packing
        result = pack_boxes(container, boxes)
        
        # Check if all boxes fit
        if len(result.unpacked) == 0:
            # All boxes fit, try higher quantity
            best_feasible = mid
            left = mid + 1
        else:
            # Some boxes don't fit, try lower quantity
            right = mid - 1
    
    return best_feasible


def max_units_single_sku(container: Container, item: Box) -> dict[str, Any]:
    """
    Calculate maximum units of a single SKU that can fit in a container using grid-based packing.
    
    Tests all 6 axis-aligned rotations and selects the one that maximizes units.
    Considers both geometry (grid fit) and weight constraints.
    
    Args:
        container: Container with dimensions and optional max_payload
        item: Box representing the item/SKU to pack
    
    Returns:
        dict with:
            - sku: Item identifier
            - max_units: Maximum number of units that fit
            - best_rotation: Tuple (L, W, H) of the best rotation
            - grid: Dict with nx, ny, nz grid dimensions
            - limiter: "geometry" or "weight" indicating the constraining factor
    """
    L = float(item.length)
    W = float(item.width)
    H = float(item.height)
    
    # Generate all 6 axis-aligned rotations
    rotations = [
        (L, W, H),  # 0: (L, W, H)
        (L, H, W),  # 1: (L, H, W)
        (W, L, H),  # 2: (W, L, H)
        (W, H, L),  # 3: (W, H, L)
        (H, L, W),  # 4: (H, L, W)
        (H, W, L),  # 5: (H, W, L)
    ]
    
    # Calculate weight-based limit
    item_weight = float(item.weight) if item.weight is not None and item.weight > 0 else 0.0
    if container.max_payload is not None and item_weight > 0:
        weight_units = math.floor(container.max_payload / item_weight)
    else:
        weight_units = float('inf')
    
    best_rotation = None
    best_units = 0
    best_grid = None
    limiter = "geometry"
    
    # Test each rotation
    for rot_l, rot_w, rot_h in rotations:
        # Calculate grid dimensions
        nx = math.floor(container.length / rot_l) if rot_l > 0 else 0
        ny = math.floor(container.width / rot_w) if rot_w > 0 else 0
        nz = math.floor(container.height / rot_h) if rot_h > 0 else 0
        
        geometry_units = nx * ny * nz
        
        # Determine feasible units (min of geometry and weight)
        if weight_units == float('inf'):
            feasible = geometry_units
            current_limiter = "geometry"
        else:
            feasible = min(geometry_units, weight_units)
            if geometry_units <= weight_units:
                current_limiter = "geometry"
            else:
                current_limiter = "weight"
        
        # Update best if this rotation is better
        if feasible > best_units:
            best_units = feasible
            best_rotation = (rot_l, rot_w, rot_h)
            best_grid = {"nx": nx, "ny": ny, "nz": nz}
            limiter = current_limiter
    
    # Extract SKU from item id (assume format "SKU_####" or just use id)
    sku = item.id
    if "_" in item.id:
        sku = item.id.rsplit("_", 1)[0]
    
    return {
        "sku": sku,
        "max_units": best_units,
        "best_rotation": best_rotation,
        "grid": best_grid,
        "limiter": limiter,
    }


def allocate_mixed_skus_greedy(
    container: Container,
    items: list[dict[str, Any] | Box],
) -> list[dict[str, Any]]:
    """
    Greedy mixed-SKU allocator that allocates units across multiple SKUs.
    
    Does NOT perform 3D packing - only calculates theoretical allocations based on
    grid-based capacity and remaining resources.
    
    Steps:
    1) Precompute single-SKU capacity and unit_volume for each item
    2) Track remaining_volume and remaining_weight
    3) Sort SKUs by priority (desc) or by 1/unit_volume (desc)
    4) For each SKU, allocate min(max_by_volume, max_by_weight, max_by_request)
    5) Decrement remaining resources and record limiting factor
    
    Args:
        container: Container with dimensions and optional max_payload
        items: List of items (Box objects or dicts with sku, dims, weight, quantity_requested, priority)
    
    Returns:
        List of allocation dicts, each with:
            - sku: SKU identifier
            - allocated: Number of units allocated
            - limiting_factor: "volume", "weight", or "request"
            - unit_volume: Volume per unit
            - unit_weight: Weight per unit
    """
    container_volume = container.length * container.width * container.height
    container_weight = container.max_payload if container.max_payload is not None else float('inf')
    
    # Step 1: Precompute single-SKU capacity and unit_volume for each item
    item_data = []
    for item in items:
        # Convert item to Box if needed
        if isinstance(item, Box):
            box = item
            sku = item.id
            if "_" in item.id:
                sku = item.id.rsplit("_", 1)[0]
            quantity_requested = None
            priority = None
        else:
            # Extract dimensions and create Box
            sku = item.get("sku", item.get("id", "UNKNOWN"))
            length_m, width_m, height_m = None, None, None
            
            if "dims_m" in item:
                dims = item["dims_m"]
                length_m = float(dims.get("L", dims.get("length", 0)))
                width_m = float(dims.get("W", dims.get("width", 0)))
                height_m = float(dims.get("H", dims.get("height", 0)))
            elif "dims_cm" in item or "case_dims_cm" in item:
                dims = item.get("dims_cm", item.get("case_dims_cm", {}))
                length_m = float(dims.get("L", dims.get("length", 0))) / 100.0
                width_m = float(dims.get("W", dims.get("width", 0))) / 100.0
                height_m = float(dims.get("H", dims.get("height", 0))) / 100.0
            elif "dims_in" in item:
                dims = item["dims_in"]
                length_m = float(dims.get("L", dims.get("length", 0))) * 0.0254
                width_m = float(dims.get("W", dims.get("width", 0))) * 0.0254
                height_m = float(dims.get("H", dims.get("height", 0))) * 0.0254
            else:
                raise ValueError(f"Item {sku} must have dims_m, dims_cm, case_dims_cm, or dims_in")
            
            weight_kg = float(item.get("weight_kg", item.get("case_weight_kg", item.get("weight", 0.0))))
            box = Box(
                id=sku,
                length=length_m,
                width=width_m,
                height=height_m,
                weight=weight_kg if weight_kg > 0 else None,
            )
            quantity_requested = item.get("quantity_requested", item.get("quantity", None))
            priority = item.get("priority", None)
        
        # Precompute capacity using max_units_single_sku
        capacity_info = max_units_single_sku(container, box)
        
        # Calculate unit_volume and unit_weight
        unit_volume = float(box.length) * float(box.width) * float(box.height)
        unit_weight = float(box.weight) if box.weight is not None and box.weight > 0 else 0.0
        
        item_data.append({
            "sku": sku,
            "box": box,
            "capacity_info": capacity_info,
            "unit_volume": unit_volume,
            "unit_weight": unit_weight,
            "quantity_requested": quantity_requested,
            "priority": priority,
        })
    
    # Step 3: Sort SKUs by priority (desc) or by 1/unit_volume (desc)
    def sort_key(item_info: dict[str, Any]) -> tuple[float, float]:
        # First sort by priority (desc), then by 1/unit_volume (desc)
        priority_val = item_info["priority"]
        if priority_val is None:
            priority_val = 0.0
        inv_volume = 1.0 / item_info["unit_volume"] if item_info["unit_volume"] > 0 else 0.0
        return (-priority_val, -inv_volume)  # Negative for descending
    
    item_data.sort(key=sort_key)
    
    # Step 2 & 4: Greedy allocation
    remaining_volume = container_volume
    remaining_weight = container_weight
    allocations = []
    
    for item_info in item_data:
        sku = item_info["sku"]
        unit_volume = item_info["unit_volume"]
        unit_weight = item_info["unit_weight"]
        quantity_requested = item_info["quantity_requested"]
        
        # Calculate max by volume
        if unit_volume > 0:
            max_by_volume = math.floor(remaining_volume / unit_volume)
        else:
            max_by_volume = float('inf')
        
        # Calculate max by weight
        if unit_weight > 0 and remaining_weight != float('inf'):
            max_by_weight = math.floor(remaining_weight / unit_weight)
        else:
            max_by_weight = float('inf')
        
        # Calculate max by request
        if quantity_requested is not None:
            max_by_request = int(quantity_requested)
        else:
            max_by_request = float('inf')
        
        # Allocate minimum of all three
        allocated = int(min(max_by_volume, max_by_weight, max_by_request))
        
        # Determine limiting factor
        if allocated == max_by_request:
            limiting_factor = "request"
        elif allocated == max_by_weight:
            limiting_factor = "weight"
        else:
            limiting_factor = "volume"
        
        # Decrement remaining resources
        remaining_volume -= allocated * unit_volume
        if remaining_weight != float('inf'):
            remaining_weight -= allocated * unit_weight
        
        allocations.append({
            "sku": sku,
            "allocated": allocated,
            "limiting_factor": limiting_factor,
            "unit_volume": unit_volume,
            "unit_weight": unit_weight,
        })
    
    return allocations


def assemble_and_write_plan(
    container: Container,
    items: list[dict[str, Any] | Box],
    allocations: list[dict[str, Any]],
    output_path: str = "plan.json",
) -> dict[str, Any]:
    """
    Assemble final plan and write output.
    
    Builds plan dict with:
    - container: Container dimensions and max_payload
    - summary: Fill rates, remaining capacity, totals
    - single_sku_capacity: Max units per SKU (single-SKU capacity)
    - mixed_allocations: Greedy allocation results
    
    Computes fill rates and remaining capacity, then writes using write_plan()
    and prints only the summary.
    
    Args:
        container: Container with dimensions and optional max_payload
        items: List of items used for allocation
        allocations: List of allocation dicts from allocate_mixed_skus_greedy()
        output_path: Path to write plan JSON file
    
    Returns:
        The assembled plan dict
    """
    import json
    from pathlib import Path
    
    # Import write_plan from cli (avoid circular import by importing at function level)
    from load_optimizer.cli import write_plan
    
    container_volume = container.length * container.width * container.height
    container_weight = container.max_payload if container.max_payload is not None else float('inf')
    
    # Compute single-SKU capacity for each item
    single_sku_capacity = {}
    for item in items:
        # Convert item to Box if needed
        if isinstance(item, Box):
            box = item
            sku = item.id
            if "_" in item.id:
                sku = item.id.rsplit("_", 1)[0]
        else:
            sku = item.get("sku", item.get("id", "UNKNOWN"))
            # Create Box for capacity calculation
            length_m, width_m, height_m = None, None, None
            if "dims_m" in item:
                dims = item["dims_m"]
                length_m = float(dims.get("L", dims.get("length", 0)))
                width_m = float(dims.get("W", dims.get("width", 0)))
                height_m = float(dims.get("H", dims.get("height", 0)))
            elif "dims_cm" in item or "case_dims_cm" in item:
                dims = item.get("dims_cm", item.get("case_dims_cm", {}))
                length_m = float(dims.get("L", dims.get("length", 0))) / 100.0
                width_m = float(dims.get("W", dims.get("width", 0))) / 100.0
                height_m = float(dims.get("H", dims.get("height", 0))) / 100.0
            elif "dims_in" in item:
                dims = item["dims_in"]
                length_m = float(dims.get("L", dims.get("length", 0))) * 0.0254
                width_m = float(dims.get("W", dims.get("width", 0))) * 0.0254
                height_m = float(dims.get("H", dims.get("height", 0))) * 0.0254
            else:
                continue  # Skip if no dimensions
            
            weight_kg = float(item.get("weight_kg", item.get("case_weight_kg", item.get("weight", 0.0))))
            box = Box(
                id=sku,
                length=length_m,
                width=width_m,
                height=height_m,
                weight=weight_kg if weight_kg > 0 else None,
            )
        
        capacity_info = max_units_single_sku(container, box)
        single_sku_capacity[sku] = capacity_info
    
    # Compute totals and remaining capacity
    total_allocated_volume = sum(alloc["allocated"] * alloc["unit_volume"] for alloc in allocations)
    total_allocated_weight = sum(alloc["allocated"] * alloc["unit_weight"] for alloc in allocations)
    total_allocated_units = sum(alloc["allocated"] for alloc in allocations)
    
    remaining_volume = container_volume - total_allocated_volume
    remaining_weight = container_weight - total_allocated_weight if container_weight != float('inf') else None
    
    volume_fill_rate = total_allocated_volume / container_volume if container_volume > 0 else 0.0
    weight_fill_rate = total_allocated_weight / container_weight if container_weight != float('inf') and container_weight > 0 else None
    
    # Build plan dict
    plan = {
        "container": {
            "length": container.length,
            "width": container.width,
            "height": container.height,
            "max_payload": container.max_payload,
            "volume": container_volume,
        },
        "summary": {
            "total_allocated_units": total_allocated_units,
            "total_allocated_volume": total_allocated_volume,
            "total_allocated_weight": total_allocated_weight,
            "container_volume": container_volume,
            "container_weight": container.max_payload,
            "remaining_volume": remaining_volume,
            "remaining_weight": remaining_weight,
            "volume_fill_rate": volume_fill_rate,
            "weight_fill_rate": weight_fill_rate,
        },
        "single_sku_capacity": single_sku_capacity,
        "mixed_allocations": allocations,
    }
    
    # Write plan using existing write_plan function
    write_plan(plan, output_path)
    
    # Print only summary
    print(json.dumps(plan["summary"], indent=2, sort_keys=True))
    
    return plan


