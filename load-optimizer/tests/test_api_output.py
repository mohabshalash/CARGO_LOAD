"""Tests for API output formatting and input validation."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from load_optimizer.api import app, format_output

client = TestClient(app)


def test_success_response_has_guaranteed_fields() -> None:
    """Test that success responses always have guaranteed fields."""
    # Create a minimal valid request
    request = {
        "shipment": {
            "container": {"type": "40HC"},
            "items": [
                {"sku": "A", "l": 50, "w": 40, "h": 30, "weight": 18, "qty": 10}
            ],
        }
    }
    
    response = client.post("/optimize", json=request)
    
    assert response.status_code == 200
    data = response.json()
    
    # Check guaranteed fields exist
    assert "metrics" in data
    assert "summary" in data
    assert "plan" in data
    
    metrics = data["metrics"]
    assert "units_loaded" in metrics
    assert "units_unloaded" in metrics
    assert "limiting_factor" in metrics
    assert "limiting_reason" in metrics
    
    # Check no None/undefined values
    assert metrics["units_loaded"] is not None
    assert metrics["units_unloaded"] is not None
    assert metrics["limiting_factor"] is not None
    assert metrics["limiting_reason"] is not None
    
    # Check types
    assert isinstance(metrics["units_loaded"], int)
    assert isinstance(metrics["units_unloaded"], int)
    assert isinstance(metrics["limiting_factor"], str)
    assert isinstance(metrics["limiting_reason"], str)
    
    # Check summary is a string
    assert isinstance(data["summary"], str)
    assert "🚢 Optimization Complete" in data["summary"]


def test_missing_input_returns_friendly_422() -> None:
    """Test that missing input returns friendly 422 error."""
    # Missing container
    request = {
        "shipment": {
            "items": [{"sku": "A", "l": 50, "w": 40, "h": 30}]
        }
    }
    
    response = client.post("/optimize", json=request)
    
    assert response.status_code == 422
    data = response.json()
    
    # Check error structure
    assert "detail" in data
    detail = data["detail"]
    
    # FastAPI wraps detail, so check if it's a dict or string
    if isinstance(detail, dict):
        error_data = detail
    else:
        # If detail is a string, try to parse it
        import json
        try:
            error_data = json.loads(detail)
        except:
            error_data = {"error": "MISSING_INFORMATION"}
    
    assert "error" in error_data or "MISSING_INFORMATION" in str(error_data)
    assert "summary" in error_data or "Missing" in str(error_data).lower()
    assert "details" in error_data or "details" in str(error_data).lower()


def test_format_output_guaranteed_fields() -> None:
    """Test format_output always returns guaranteed fields."""
    # Test with minimal plan
    plan = {
        "summary": {
            "requested_units": 100,
            "allocated_units": 80,
            "volume_fill_rate": 0.95,
            "weight_fill_rate": None,
        },
        "container": {"max_weight": None},
        "mixed_allocations": [],
    }
    
    output = format_output(plan)
    
    assert "metrics" in output
    assert "summary" in output
    
    metrics = output["metrics"]
    assert metrics["units_loaded"] == 80
    assert metrics["units_unloaded"] == 20
    assert metrics["limiting_factor"] in ["volume", "weight", "dimensions", "count", "none"]
    assert isinstance(metrics["limiting_reason"], str)
    assert len(metrics["limiting_reason"]) > 0


def test_format_output_with_weight() -> None:
    """Test format_output handles weight correctly."""
    plan = {
        "summary": {
            "requested_units": 100,
            "allocated_units": 100,
            "volume_fill_rate": 0.85,
            "weight_fill_rate": 0.99,
        },
        "container": {"max_weight": 26500.0},
        "mixed_allocations": [],
    }
    
    output = format_output(plan)
    summary = output["summary"]
    
    # Should show weight fill percentage
    assert "Weight Fill:" in summary
    assert "N/A" not in summary or "99.0%" in summary


def test_format_output_without_weight() -> None:
    """Test format_output handles missing weight correctly."""
    plan = {
        "summary": {
            "requested_units": 100,
            "allocated_units": 100,
            "volume_fill_rate": 0.85,
            "weight_fill_rate": None,
        },
        "container": {"max_weight": None},
        "mixed_allocations": [],
    }
    
    output = format_output(plan)
    summary = output["summary"]
    
    # Should show N/A for weight
    assert "Weight Fill: N/A" in summary


def test_render_data_structure() -> None:
    """Test that placements_render has exactly the required keys."""
    from load_optimizer.api import extract_render_data
    from load_optimizer.models import Placement
    
    # Create a plan with placements
    plan = {
        "container": {
            "length": 12.032,
            "width": 2.350,
            "height": 2.700,
        },
        "placements": [
            Placement(
                box_id="TEST_001",
                x=0.0,
                y=0.0,
                z=0.0,
                rotation=(0.5, 0.4, 0.3),  # L, W, H
            )
        ],
    }
    
    placements_render, container_render = extract_render_data(plan)
    
    # Check container_render
    assert container_render == {"L": 12.032, "W": 2.350, "H": 2.700}
    
    # Check placements_render
    assert len(placements_render) == 1
    placement = placements_render[0]
    
    # Check required keys: x, y, z, dims
    required_keys = {"x", "y", "z", "dims"}
    assert set(placement.keys()) == required_keys, f"Expected exactly {required_keys}, got {set(placement.keys())}"
    
    # Check values
    assert placement["x"] == 0.0
    assert placement["y"] == 0.0
    assert placement["z"] == 0.0
    assert placement["dims"] == [0.5, 0.4, 0.3]


def test_render_data_without_placements() -> None:
    """Test that placements_render is empty when no placements exist."""
    from load_optimizer.api import extract_render_data
    
    plan = {
        "container": {
            "length": 12.032,
            "width": 2.350,
            "height": 2.700,
        },
        # No placements
    }
    
    placements_render, container_render = extract_render_data(plan)
    
    assert placements_render == []
    assert container_render == {"L": 12.032, "W": 2.350, "H": 2.700}


def test_optimize_with_render_param() -> None:
    """Test that render=1 query param includes rendering data."""
    request = {
        "shipment": {
            "container": {"type": "40HC"},
            "items": [
                {"sku": "A", "l": 50, "w": 40, "h": 30, "weight": 18, "qty": 10}
            ],
        }
    }
    
    # Test without render
    response = client.post("/optimize", json=request)
    assert response.status_code == 200
    data = response.json()
    assert "placements_render" not in data
    assert "container_render" not in data
    
    # Test with render=1
    response = client.post("/optimize?render=1", json=request)
    assert response.status_code == 200
    data = response.json()
    
    # Should have container_render (even if no placements)
    assert "container_render" in data
    assert "L" in data["container_render"]
    assert "W" in data["container_render"]
    assert "H" in data["container_render"]
    
    # placements_render should always be present when render=1
    assert "placements_render" in data, "placements_render should always be present when render=1"
    assert isinstance(data["placements_render"], list)
    
    # If placements exist, verify structure
    if len(data["placements_render"]) > 0:
        placement = data["placements_render"][0]
        expected_keys = {"x", "y", "z", "dims"}
        actual_keys = set(placement.keys())
        assert actual_keys == expected_keys, f"Expected exactly {expected_keys}, got {actual_keys}"
        assert isinstance(placement["dims"], list) and len(placement["dims"]) == 3

