"""Economic quantity optimizer using OR-Tools CP-SAT - maximizes fill rate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ortools.sat.python import cp_model

from load_optimizer.models import Container


def optimize_quantities(
    container: Container,
    skus: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Optimize case quantities per SKU to maximize container fill rate (volume and weight utilization).
    
    Objective: Maximize combined fill rate (weight + volume utilization)
    Strategy: Prioritize SKUs that contribute most to both weight and volume fill rates.
    
    Args:
        container: Container with dimensions and max_payload
        skus: List of SKU dicts with keys:
            - sku: str identifier
            - case_weight_kg: float
            - case_volume_m3: float (or case_dims_cm to calculate)
            - min_cases: int
            - max_cases: int
    
    Returns:
        Solution dict with recommended quantities and metrics
    """
    model = cp_model.CpModel()
    
    # Calculate case_volume_m3 for each SKU if not provided
    for sku_data in skus:
        if "case_volume_m3" not in sku_data:
            if "case_dims_cm" in sku_data:
                dims = sku_data["case_dims_cm"]
                L_cm = float(dims.get("L", dims.get("length", 1.0)))
                W_cm = float(dims.get("W", dims.get("width", 1.0)))
                H_cm = float(dims.get("H", dims.get("height", 1.0)))
                # Convert cm³ to m³
                sku_data["case_volume_m3"] = (L_cm * W_cm * H_cm) / 1_000_000.0
            else:
                raise ValueError(f"SKU {sku_data.get('sku', 'unknown')} must have either 'case_volume_m3' or 'case_dims_cm'")
    
    # Decision variables: x[sku] = integer cases
    x: dict[str, Any] = {}
    for sku_data in skus:
        sku_id = sku_data["sku"]
        x[sku_id] = model.NewIntVar(
            sku_data["min_cases"],
            sku_data["max_cases"],
            f"x_{sku_id}",
        )
    
    # Container capacity
    container_volume = container.length * container.width * container.height
    container_weight = container.max_payload if container.max_payload is not None else float('inf')
    
    # Constraint: sum(x * case_weight_kg) <= container.max_payload_kg
    if container.max_payload is not None:
        weight_expr = []
        for sku_data in skus:
            sku_id = sku_data["sku"]
            weight_expr.append(x[sku_id] * sku_data["case_weight_kg"])
        model.Add(sum(weight_expr) <= container.max_payload)
    
    # Constraint: sum(x * case_volume_m3) <= container.internal_volume_m3 * 0.95
    volume_limit = container_volume * 0.95
    volume_expr = []
    for sku_data in skus:
        sku_id = sku_data["sku"]
        volume_expr.append(x[sku_id] * sku_data["case_volume_m3"])
    model.Add(sum(volume_expr) <= volume_limit)
    
    # Objective: Maximize combined fill rate
    # Use normalized contribution: weight_fill + volume_fill
    # Scale by large multiplier to ensure integer optimization works well
    SCALE = 1000000
    
    weight_contribution = []
    volume_contribution = []
    
    for sku_data in skus:
        sku_id = sku_data["sku"]
        case_weight = sku_data["case_weight_kg"]
        case_volume = sku_data["case_volume_m3"]
        
        # Normalized contribution to fill rates
        weight_fill_per_case = (case_weight / container_weight) if container_weight < float('inf') else 0.0
        volume_fill_per_case = case_volume / container_volume if container_volume > 0 else 0.0
        
        # Combined fill rate contribution (weighted equally)
        combined_fill = weight_fill_per_case + volume_fill_per_case
        
        weight_contribution.append(x[sku_id] * int(case_weight * SCALE))
        volume_contribution.append(x[sku_id] * int(case_volume * SCALE))
    
    # Maximize: combined weight + volume utilization
    # This effectively maximizes fill rate
    model.Maximize(sum(weight_contribution) + sum(volume_contribution))
    
    # Solve
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"Solver failed with status {status}")
    
    # Extract solution
    solution: dict[str, Any] = {
        "sku_quantities": {},
        "totals": {},
        "fill_rates": {},
        "sku_metrics": {},
        "recommendation": {},
    }
    
    total_weight = 0.0
    total_volume = 0.0
    sku_metrics_temp: dict[str, dict[str, float]] = {}
    
    # First pass: collect quantities and totals
    for sku_data in skus:
        sku_id = sku_data["sku"]
        quantity = int(solver.Value(x[sku_id]))
        solution["sku_quantities"][sku_id] = quantity
        
        if quantity > 0:
            sku_weight = quantity * sku_data["case_weight_kg"]
            sku_volume = quantity * sku_data["case_volume_m3"]
            
            total_weight += sku_weight
            total_volume += sku_volume
            
            sku_metrics_temp[sku_id] = {
                "quantity": quantity,
                "total_weight_kg": sku_weight,
                "total_volume_m3": sku_volume,
            }
    
    # Second pass: calculate fill rate metrics per SKU
    container_weight = container.max_payload if container.max_payload is not None else 0.0
    for sku_id, metrics in sku_metrics_temp.items():
        sku_data = next(s for s in skus if s["sku"] == sku_id)
        
        # Calculate contribution to fill rates
        weight_contribution = metrics["total_weight_kg"] / container_weight if container_weight > 0 else 0.0
        volume_contribution = metrics["total_volume_m3"] / container_volume if container_volume > 0 else 0.0
        
        solution["sku_metrics"][sku_id] = {
            **metrics,
            "weight_fill_contribution": weight_contribution,
            "volume_fill_contribution": volume_contribution,
            "cases_per_m3": metrics["quantity"] / container_volume if container_volume > 0 else 0.0,
            "cases_per_kg": metrics["quantity"] / container_weight if container_weight > 0 else 0.0,
        }
    
    solution["totals"] = {
        "total_cases": sum(solution["sku_quantities"].values()),
        "total_weight_kg": total_weight,
        "total_volume_m3": total_volume,
    }
    
    # Calculate fill rates
    weight_fill_rate = total_weight / container.max_payload if container.max_payload and container.max_payload > 0 else 0.0
    volume_fill_rate = total_volume / container_volume if container_volume > 0 else 0.0
    combined_fill_rate = (weight_fill_rate + volume_fill_rate) / 2.0 if (container.max_payload and container.max_payload > 0) else volume_fill_rate
    
    solution["fill_rates"] = {
        "weight_fill_rate": weight_fill_rate,
        "volume_fill_rate": volume_fill_rate,
        "combined_fill_rate": combined_fill_rate,
    }
    
    # Add recommendation summary
    solution["recommendation"] = {
        "total_cases_recommended": solution["totals"]["total_cases"],
        "weight_utilization_pct": weight_fill_rate * 100.0,
        "volume_utilization_pct": volume_fill_rate * 100.0,
        "combined_utilization_pct": combined_fill_rate * 100.0,
    }
    
    return solution


def save_solution(solution: dict[str, Any], output_path: Path) -> None:
    """Save solution to JSON file."""
    output_path.write_text(json.dumps(solution, indent=2))
    print(f"✅ Economic solution written to {output_path}")

