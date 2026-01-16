"""Regression test for large-qty single-SKU capacity limits."""

from __future__ import annotations

from load_optimizer.api import build_plan
from load_optimizer.containers import get_container_dims
from load_optimizer.models import Container


def test_large_qty_single_sku_capacity_limits() -> None:
    """
    Regression test: 30x40x50 cm, 18kg, qty=4000, container=40HC.
    
    Verifies:
    - allocated_units <= theoretical_max_by_volume
    - unloaded_units > 0
    - limiting_factor is VOLUME (unless placement logic makes it GEOMETRY)
    """
    # Setup: 40HC container
    container_dims = get_container_dims("40HC")
    container = Container(
        length=container_dims["length"],
        width=container_dims["width"],
        height=container_dims["height"],
        max_payload=26500.0,  # 40HC max payload
    )
    
    # Setup: Single SKU with large quantity
    # Dimensions: 30x40x50 cm = 0.3 x 0.4 x 0.5 m = 0.06 m³
    # Weight: 18 kg
    # Quantity: 4000 units
    shipment = {
        "container": {
            "length": container.length,
            "width": container.width,
            "height": container.height,
            "max_payload": container.max_payload,
        },
        "boxes": [
            {
                "sku": "TEST_SKU",
                "dims_cm": {"L": 30, "W": 40, "H": 50},
                "weight_kg": 18.0,
                "quantity": 4000,
            }
        ],
    }
    
    # Build plan
    plan = build_plan(shipment)
    
    # Verify summary fields
    summary = plan["summary"]
    assert "requested_units" in summary
    assert "allocated_units" in summary
    assert "unloaded_units" in summary
    assert summary["requested_units"] == 4000
    assert summary["allocated_units"] <= summary["requested_units"]
    assert summary["unloaded_units"] == summary["requested_units"] - summary["allocated_units"]
    assert summary["unloaded_units"] > 0, "Should have unloaded units for qty=4000"
    
    # Verify mixed_allocations fields
    assert len(plan["mixed_allocations"]) == 1
    allocation = plan["mixed_allocations"][0]
    
    assert "requested_units" in allocation
    assert "allocated_units" in allocation
    assert "unloaded_units" in allocation
    assert "limiting_factor" in allocation
    assert "theoretical_max_by_volume" in allocation
    assert "theoretical_max_by_weight" in allocation
    
    assert allocation["requested_units"] == 4000
    assert allocation["allocated_units"] <= allocation["requested_units"]
    assert allocation["unloaded_units"] == allocation["requested_units"] - allocation["allocated_units"]
    assert allocation["unloaded_units"] > 0
    
    # Verify hard cap: allocated_units <= theoretical_max_by_volume
    assert allocation["allocated_units"] <= allocation["theoretical_max_by_volume"], \
        f"allocated_units ({allocation['allocated_units']}) must be <= theoretical_max_by_volume ({allocation['theoretical_max_by_volume']})"
    
    # Verify limiting_factor
    assert allocation["limiting_factor"] in ["VOLUME", "WEIGHT", "GEOMETRY/PLACEMENT", "UNKNOWN"]
    
    # If allocated equals theoretical_max_by_volume and requested > allocated, should be VOLUME
    # (unless placement logic makes it GEOMETRY/PLACEMENT)
    if allocation["allocated_units"] == allocation["theoretical_max_by_volume"] and \
       allocation["requested_units"] > allocation["allocated_units"]:
        assert allocation["limiting_factor"] in ["VOLUME", "GEOMETRY/PLACEMENT"], \
            f"Expected VOLUME or GEOMETRY/PLACEMENT, got {allocation['limiting_factor']}"
    
    # Verify fill rates are clamped to <= 1.0
    assert summary["volume_fill_rate"] <= 1.0
    assert summary["weight_fill_rate"] <= 1.0

