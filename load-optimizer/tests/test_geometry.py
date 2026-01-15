from __future__ import annotations

from load_optimizer.geometry import boxes_overlap


def test_boxes_overlap_overlapping() -> None:
    """Test that overlapping boxes are detected."""
    # Box a: (0, 0, 0) to (2, 2, 2)
    a = (0.0, 0.0, 0.0, 2.0, 2.0, 2.0)
    # Box b: (1, 1, 1) to (3, 3, 3) - overlaps with a
    b = (1.0, 1.0, 1.0, 3.0, 3.0, 3.0)
    
    assert boxes_overlap(a, b) is True


def test_boxes_overlap_not_overlapping() -> None:
    """Test that non-overlapping boxes are detected."""
    # Box a: (0, 0, 0) to (1, 1, 1)
    a = (0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    # Box b: (2, 2, 2) to (3, 3, 3) - does not overlap with a
    b = (2.0, 2.0, 2.0, 3.0, 3.0, 3.0)
    
    assert boxes_overlap(a, b) is False







