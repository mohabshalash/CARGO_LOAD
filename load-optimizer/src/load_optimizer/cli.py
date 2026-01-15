from __future__ import annotations

import argparse
import json
from pathlib import Path

from load_optimizer.models import Container, Box
from load_optimizer.packing.first_fit import pack_boxes
from load_optimizer.economic_optimizer import optimize_quantities, save_solution

# #region agent log
LOG_PATH = r"c:\Users\mohab\OneDrive\Interconnect 360\I360 Load Optimizor\.cursor\debug.log"
def _log(location: str, message: str, data: dict, hypothesis_id: str = ""):
    import json
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps({"sessionId": "debug-session", "runId": "run1", "hypothesisId": hypothesis_id, "location": location, "message": message, "data": data, "timestamp": __import__("time").time() * 1000}) + "\n")
    except: pass
# #endregion


def load_input(path: Path):
    data = json.loads(path.read_text())

    container_kwargs = {}

    # 1) Load dimensions from preset OR explicit container
    if "container_preset" in data:
        from load_optimizer.containers import get_container_dims
        dims = get_container_dims(data["container_preset"])
        print(f"✅ Using container preset: {data['container_preset']} -> {dims}")
        container_kwargs.update(dims)
    elif "container" in data:
        container_kwargs.update(data["container"])
    else:
        raise ValueError("Input must include either 'container_preset' or 'container'")

    # 2) Merge optional container overrides (payload, etc.)
    if "container" in data:
        # This safely adds max_payload or future fields
        container_kwargs.update(data["container"])

    container = Container(**container_kwargs)

    boxes = []
    for b in data.get("boxes", []):
        # Format 1: has id, length, width, height (assumed meters)
        if "id" in b and "length" in b and "width" in b and "height" in b:
            boxes.append(Box(**b))
        # Format 2: sku, dims_in, quantity (convert inches to meters: in * 0.0254)
        elif "sku" in b and "dims_in" in b:
            sku = b["sku"]
            dims_in = b["dims_in"]
            weight_kg = b.get("weight_kg", 0.0)
            quantity = b.get("quantity", 1)
            
            for i in range(quantity):
                box_dict = {
                    "id": f"{sku}_{i:04d}",
                    "length": float(dims_in["L"]) * 0.0254,
                    "width": float(dims_in["W"]) * 0.0254,
                    "height": float(dims_in["H"]) * 0.0254,
                    "weight": float(weight_kg),
                }
                boxes.append(Box(**box_dict))
        # Format 3: sku, dims_cm, quantity (convert cm to meters: cm / 100)
        elif "sku" in b and "dims_cm" in b:
            sku = b["sku"]
            dims_cm = b["dims_cm"]
            weight_kg = b.get("weight_kg", 0.0)
            quantity = b.get("quantity", 1)
            
            for i in range(quantity):
                box_dict = {
                    "id": f"{sku}_{i:04d}",
                    "length": float(dims_cm["L"]) / 100.0,
                    "width": float(dims_cm["W"]) / 100.0,
                    "height": float(dims_cm["H"]) / 100.0,
                    "weight": float(weight_kg),
                }
                boxes.append(Box(**box_dict))
        # Format 4: sku, dims_m, quantity (already in meters)
        elif "sku" in b and "dims_m" in b:
            sku = b["sku"]
            dims_m = b["dims_m"]
            weight_kg = b.get("weight_kg", 0.0)
            quantity = b.get("quantity", 1)
            
            for i in range(quantity):
                box_dict = {
                    "id": f"{sku}_{i:04d}",
                    "length": float(dims_m["L"]),
                    "width": float(dims_m["W"]),
                    "height": float(dims_m["H"]),
                    "weight": float(weight_kg),
                }
                boxes.append(Box(**box_dict))
    
    # Debug print: container dims (m) and first box dims (m)
    if boxes:
        print(f"DEBUG container dims (m): {container.length:.3f} x {container.width:.3f} x {container.height:.3f}")
        print(f"DEBUG first box dims (m): {boxes[0].length:.3f} x {boxes[0].width:.3f} x {boxes[0].height:.3f}")
    
    return container, boxes, data


def generate_boxes_from_quantities(skus: list[dict], sku_quantities: dict[str, int]) -> list[Box]:
    """Generate Box objects from SKU quantities."""
    boxes = []
    # #region agent log
    _log("cli.py:40", "generate_boxes_from_quantities entry", {"skus_count": len(skus), "sku_quantities": sku_quantities}, "B")
    # #endregion
    for sku_data in skus:
        sku_id = sku_data["sku"]
        quantity = sku_quantities.get(sku_id, 0)
        # #region agent log
        _log("cli.py:45", "Processing SKU", {"sku_id": sku_id, "quantity": quantity, "sku_data_keys": list(sku_data.keys())}, "B")
        # #endregion
        
        # Handle case_dims_cm (convert cm to m) or case_volume_m3
        if "case_dims_cm" in sku_data:
            # Convert cm to meters
            dims = sku_data["case_dims_cm"]
            case_length = float(dims.get("L", dims.get("length", 1.0))) / 100.0
            case_width = float(dims.get("W", dims.get("width", 1.0))) / 100.0
            case_height = float(dims.get("H", dims.get("height", 1.0))) / 100.0
        elif "case_volume_m3" in sku_data:
            case_volume = sku_data["case_volume_m3"]
            # Try to get explicit dimensions, otherwise estimate from volume
            if "case_length" in sku_data and "case_width" in sku_data and "case_height" in sku_data:
                case_length = sku_data["case_length"]
                case_width = sku_data["case_width"]
                case_height = sku_data["case_height"]
            else:
                # Estimate as cube root of volume (simple approximation)
                case_length = case_width = case_height = (case_volume ** (1.0/3.0))
        else:
            # #region agent log
            _log("cli.py:60", "Missing dimensions", {"sku_id": sku_id}, "B")
            # #endregion
            raise ValueError(f"SKU {sku_id} must have either 'case_dims_cm' or 'case_volume_m3'")
        
        case_weight = sku_data["case_weight_kg"]
        
        # #region agent log
        _log("cli.py:66", "Creating boxes", {"sku_id": sku_id, "quantity": quantity, "dims": [case_length, case_width, case_height]}, "B")
        # #endregion
        for i in range(quantity):
            boxes.append(Box(
                id=f"{sku_id}_{i+1}",
                length=case_length,
                width=case_width,
                height=case_height,
                weight=case_weight,
            ))
    # #region agent log
    _log("cli.py:75", "generate_boxes_from_quantities exit", {"boxes_count": len(boxes)}, "B")
    # #endregion
    return boxes


def main():
    parser = argparse.ArgumentParser(description="Load Optimizer CLI")
    parser.add_argument("--input", required=True, help="Input shipment JSON file")
    parser.add_argument("--output", required=True, help="Output plan JSON file")
    parser.add_argument(
        "--mode",
        choices=["single", "multi", "economic", "capacity"],
        default="single",
        help="single = one container, multi = multiple containers, economic = optimize quantities first, capacity = compute max cases per SKU",
    )
    parser.add_argument(
        "--solution",
        help="Output path for economic solution JSON (required for economic mode)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Write placements to JSONL file and plan_full.json with sample",
    )
    parser.add_argument(
        "--skip_sku_counts",
        action="store_true",
        help="Skip SKU counting to improve performance",
    )

    args = parser.parse_args()
    
    print("DEBUG mode=", args.mode)

    container, boxes, data = load_input(Path(args.input))
    # #region agent log
    _log("cli.py:89", "After load_input", {"mode": args.mode, "data_keys": list(data.keys()), "initial_boxes_count": len(boxes)}, "A")
    # #endregion
    
    # Economic optimization mode
    if args.mode == "economic":
        # #region agent log
        _log("cli.py:92", "Entering economic mode", {"has_skus": "skus" in data, "has_products": "products" in data, "has_boxes": "boxes" in data}, "A")
        # #endregion
        if not args.solution:
            raise ValueError("Economic mode requires --solution output path")
        
        # Convert shipment.json format to economic mode format
        if "boxes" in data:
            # Convert boxes array to products format
            products_data = []
            for box_item in data["boxes"]:
                # Handle dims_m (meters) -> convert to case_dims_cm
                if "dims_m" in box_item:
                    dims_m = box_item["dims_m"]
                    case_dims_cm = {
                        "L": float(dims_m.get("L", dims_m.get("length", 0))) * 100.0,
                        "W": float(dims_m.get("W", dims_m.get("width", 0))) * 100.0,
                        "H": float(dims_m.get("H", dims_m.get("height", 0))) * 100.0,
                    }
                else:
                    raise ValueError(f"Box {box_item.get('sku', 'unknown')} must have 'dims_m' field")
                
                product = {
                    "sku": box_item.get("sku", box_item.get("id", "UNKNOWN")),
                    "case_dims_cm": case_dims_cm,
                    "case_weight_kg": float(box_item.get("weight_kg", box_item.get("weight", 0))),
                    "min_cases": 0,
                    "max_cases": int(box_item.get("quantity", box_item.get("max_cases", 5000))),
                }
                products_data.append(product)
        elif "skus" in data or "products" in data:
            # Use existing products/skus format
            products_data = data.get("products", data.get("skus", []))
        else:
            raise ValueError("Economic mode requires 'boxes', 'skus', or 'products' in input JSON")
        # #region agent log
        _log("cli.py:99", "Products/SKUs data", {"count": len(products_data), "first_item_keys": list(products_data[0].keys()) if products_data else []}, "B")
        # #endregion
        
        print("✅ Running economic quantity optimizer...")
        solution = optimize_quantities(container, products_data)
        # #region agent log
        _log("cli.py:103", "After optimize_quantities", {"solution_keys": list(solution.keys()), "sku_quantities": solution.get("sku_quantities", {})}, "D")
        # #endregion
        save_solution(solution, Path(args.solution))
        
        # Generate boxes from recommended quantities
        boxes = generate_boxes_from_quantities(products_data, solution["sku_quantities"])
        # #region agent log
        _log("cli.py:108", "After generate_boxes_from_quantities", {"boxes_count": len(boxes), "first_box_id": boxes[0].id if boxes else None}, "E")
        # #endregion
        print(f"✅ Generated {len(boxes)} boxes from optimized quantities")
        
        # Use single container mode after economic optimization
        args.mode = "single"
    
    if args.mode == "capacity":
        from load_optimizer.capacity import compute_max_cases_theoretical, compute_max_cases_feasible
        
        # Load products/SKUs from input
        if "boxes" in data:
            # Convert boxes array to products format
            products_data = []
            for box_item in data["boxes"]:
                product = {
                    "sku": box_item.get("sku", box_item.get("id", "UNKNOWN")),
                }
                # Copy dimension fields as-is
                if "dims_m" in box_item:
                    product["dims_m"] = box_item["dims_m"]
                elif "dims_cm" in box_item:
                    product["dims_cm"] = box_item["dims_cm"]
                elif "case_dims_cm" in box_item:
                    product["case_dims_cm"] = box_item["case_dims_cm"]
                elif "dims_in" in box_item:
                    product["dims_in"] = box_item["dims_in"]
                
                # Copy weight
                if "weight_kg" in box_item:
                    product["weight_kg"] = box_item["weight_kg"]
                elif "case_weight_kg" in box_item:
                    product["case_weight_kg"] = box_item["case_weight_kg"]
                elif "weight" in box_item:
                    product["weight_kg"] = box_item["weight"]
                
                # Copy max_cases if present
                if "max_cases" in box_item:
                    product["max_cases"] = box_item["max_cases"]
                
                products_data.append(product)
        elif "skus" in data or "products" in data:
            products_data = data.get("products", data.get("skus", []))
        else:
            raise ValueError("Capacity mode requires 'boxes', 'skus', or 'products' in input JSON")
        
        print("✅ Computing theoretical max cases per SKU...")
        theoretical_result = compute_max_cases_theoretical(container, products_data)
        
        print("✅ Computing feasible max cases per SKU (this may take a while)...")
        feasible_max: dict[str, int] = {}
        for product in products_data:
            sku = product.get("sku", product.get("id", "UNKNOWN"))
            q_max = theoretical_result["theoretical_max"].get(sku, 0)
            print(f"  Testing {sku}: theoretical_max={q_max}...")
            feasible = compute_max_cases_feasible(container, product, q_max)
            feasible_max[sku] = feasible
            print(f"  {sku}: feasible_max={feasible}")
        
        # Build output
        output = {
            "container_preset": data.get("container_preset"),
            "container": {
                "length": container.length,
                "width": container.width,
                "height": container.height,
                "max_payload": container.max_payload,
            },
            "theoretical_max": theoretical_result["theoretical_max"],
            "feasible_max": feasible_max,
            "details": theoretical_result["details"],
        }
        
        # Write output
        write_plan(output, args.output)
        if "summary" in output:
            print(json.dumps(output["summary"], indent=2, sort_keys=True))
        print(f"✅ Capacity analysis written to {args.output}")
        return
    
    if args.mode == "single":
        print("DEBUG boxes_to_pack=", len(boxes))
        pack_fn = pack_boxes
        print("DEBUG packer_fn=", pack_fn.__module__, pack_fn.__name__)
        
        if len(boxes) == 0:
            print("DEBUG reason: no boxes generated from input")
        
        result = pack_boxes(container, boxes)
        
        if len(result.placements) == 0 and len(boxes) > 0:
            print("DEBUG reason: packer returned no placements (check return shape)")
        
        # Print packing results
        packed_count = len(result.placements)
        unpacked_count = len(result.unpacked)
        requested = len(boxes)
        
        # Calculate SKU counts (skip if flag set)
        counts_by_sku = {}
        if not args.skip_sku_counts:
            from collections import Counter
            packed_sku_counts = Counter()
            unpacked_sku_counts = Counter()
            for p in result.placements:
                if "_" in p.box_id:
                    sku = p.box_id.rsplit("_", 1)[0]
                    packed_sku_counts[sku] += 1
            for b in result.unpacked:
                if "_" in b.id:
                    sku = b.id.rsplit("_", 1)[0]
                    unpacked_sku_counts[sku] += 1
            for sku in set(list(packed_sku_counts.keys()) + list(unpacked_sku_counts.keys())):
                counts_by_sku[sku] = {
                    "packed": packed_sku_counts.get(sku, 0),
                    "unpacked": unpacked_sku_counts.get(sku, 0),
                }
        
        # Print calculations
        print(f"Packed {packed_count}/{requested}, Unpacked {unpacked_count}, Fill={result.fill_rate:.3f}, UsedVol={result.used_volume:.3f}, ContainerVol={result.container_volume:.3f}")
        
        # Build output summary
        container_preset = data.get("container_preset")
        container_dims = {"length": container.length, "width": container.width, "height": container.height}
        
        output = {
            "container_preset": container_preset,
            "container_dims": container_dims,
            "requested_boxes": len(boxes),
            "packed_boxes": len(result.placements),
            "unpacked_boxes": len(result.unpacked),
            "used_volume": result.used_volume,
            "container_volume": result.container_volume,
            "fill_rate": result.fill_rate,
            "counts_by_sku": counts_by_sku,
        }
        
        # Write full placements if --full flag
        if args.full:
            # Write placements to JSONL file (one per line)
            jsonl_path = Path(args.output).parent / f"{Path(args.output).stem}_placements.jsonl"
            with open(jsonl_path, "w") as f:
                for p in result.placements:
                    f.write(json.dumps(p.model_dump(), separators=(",", ":")) + "\n")
            print(f"✅ Placements written to {jsonl_path}")
            
            # Write plan_full.json with summary + sample
            placements_sample = [p.model_dump() for p in result.placements[:100]]
            full_output = {
                "container_preset": container_preset,
                "container_dims": container_dims,
                "requested_boxes": len(boxes),
                "packed_boxes": len(result.placements),
                "unpacked_boxes": len(result.unpacked),
                "used_volume": result.used_volume,
                "container_volume": result.container_volume,
                "fill_rate": result.fill_rate,
                "placements_sample": placements_sample,
            }
            full_path = Path(args.output).parent / f"{Path(args.output).stem}_full.json"
            write_plan(full_output, str(full_path))
            if "summary" in full_output:
                print(json.dumps(full_output["summary"], indent=2, sort_keys=True))
            print(f"✅ Full summary written to {full_path}")

    else:
        from load_optimizer.packing.multi_container import pack_multiple
        
        print("DEBUG boxes_to_pack=", len(boxes))
        pack_fn = pack_multiple
        print("DEBUG packer_fn=", pack_fn.__module__, pack_fn.__name__)
        
        if len(boxes) == 0:
            print("DEBUG reason: no boxes generated from input")
        
        plans, leftover = pack_multiple(container, boxes)
        
        if len(plans) == 0 or (len(plans) > 0 and len(plans[0].placements) == 0) and len(boxes) > 0:
            print("DEBUG reason: packer returned no placements (check return shape)")
        
        # Print packing results (aggregated across all containers)
        packed_count = sum(len(plan.placements) for plan in plans)
        unpacked_count = len(leftover)
        requested = len(boxes)
        total_used_volume = sum(plan.used_volume for plan in plans)
        total_container_volume = sum(plan.container_volume for plan in plans)
        avg_fill_rate = (total_used_volume / total_container_volume) if total_container_volume > 0 else 0.0
        
        # Calculate SKU counts (skip if flag set)
        counts_by_sku = {}
        if not args.skip_sku_counts:
            from collections import Counter
            packed_sku_counts = Counter()
            unpacked_sku_counts = Counter()
            for plan in plans:
                for p in plan.placements:
                    if "_" in p.box_id:
                        sku = p.box_id.rsplit("_", 1)[0]
                        packed_sku_counts[sku] += 1
            for b in leftover:
                if "_" in b.id:
                    sku = b.id.rsplit("_", 1)[0]
                    unpacked_sku_counts[sku] += 1
            for sku in set(list(packed_sku_counts.keys()) + list(unpacked_sku_counts.keys())):
                counts_by_sku[sku] = {
                    "packed": packed_sku_counts.get(sku, 0),
                    "unpacked": unpacked_sku_counts.get(sku, 0),
                }
        
        # Print calculations
        print(f"Packed {packed_count}/{requested}, Unpacked {unpacked_count}, Fill={avg_fill_rate:.3f}, UsedVol={total_used_volume:.3f}, ContainerVol={total_container_volume:.3f}")
        
        # Build output summary
        container_preset = data.get("container_preset")
        container_dims = {"length": container.length, "width": container.width, "height": container.height}
        
        output = {
            "container_preset": container_preset,
            "container_dims": container_dims,
            "requested_boxes": len(boxes),
            "packed_boxes": packed_count,
            "unpacked_boxes": unpacked_count,
            "used_volume": total_used_volume,
            "container_volume": total_container_volume,
            "fill_rate": avg_fill_rate,
            "counts_by_sku": counts_by_sku,
        }
        
        # Write full placements if --full flag
        if args.full:
            # Write placements to JSONL file (one per line)
            jsonl_path = Path(args.output).parent / f"{Path(args.output).stem}_placements.jsonl"
            with open(jsonl_path, "w") as f:
                for plan in plans:
                    for p in plan.placements:
                        f.write(json.dumps(p.model_dump(), separators=(",", ":")) + "\n")
            print(f"✅ Placements written to {jsonl_path}")
            
            # Write plan_full.json with summary + sample
            containers_sample = [
                {
                    "placements_sample": [p.model_dump() for p in plan.placements[:100]],
                    "used_volume": plan.used_volume,
                    "container_volume": plan.container_volume,
                    "fill_rate": plan.fill_rate,
                }
                for plan in plans
            ]
            full_output = {
                "container_preset": container_preset,
                "container_dims": container_dims,
                "requested_boxes": len(boxes),
                "packed_boxes": packed_count,
                "unpacked_boxes": unpacked_count,
                "used_volume": total_used_volume,
                "container_volume": total_container_volume,
                "fill_rate": avg_fill_rate,
                "containers": containers_sample,
                "leftover": [b.id for b in leftover],
            }
            full_path = Path(args.output).parent / f"{Path(args.output).stem}_full.json"
            write_plan(full_output, str(full_path))
            if "summary" in full_output:
                print(json.dumps(full_output["summary"], indent=2, sort_keys=True))
            print(f"✅ Full summary written to {full_path}")

    # Plan assembly: compute capacity and allocations, then write final plan
    from load_optimizer.capacity import max_units_single_sku, allocate_mixed_skus_greedy
    
    # Get items from original data (boxes array)
    items = data.get("boxes", [])
    
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
                # Skip if no dimensions or already a Box
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
    
    # Step 4: Call write_plan at the end
    write_plan(plan, args.output)
    print(json.dumps(plan["summary"], indent=2, sort_keys=True))
    
    # Step 5: Debug line
    output_path = Path(args.output).resolve()
    cwd = Path.cwd()
    print(f"DEBUG output_path={output_path}")
    print(f"DEBUG cwd={cwd}")


def write_plan(plan: dict, path: str = "plan.json") -> None:
    """
    Write a plan dictionary to a JSON file.
    
    Creates parent folders if needed, writes JSON with indent=2 and sort_keys=True,
    and overwrites the file on every run.
    
    Args:
        plan: Dictionary containing the plan data
        path: Output file path (default: "plan.json")
    """
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"DEBUG write_plan: writing to {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
