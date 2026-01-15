""""Geometry utilities for load optimization."""

from __future__ import annotations
import os

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .models import Box, Container, Placement


def boxes_overlap(
    a: tuple[float, float, float, float, float, float],
    b: tuple[float, float, float, float, float, float],
) -> bool:
    """
    Axis-aligned bounding box (AABB) overlap test.

    a, b are bounds: (x1, y1, z1, x2, y2, z2)

    Overlap exists only if they overlap on ALL 3 axes with positive volume.
    Touching faces/edges (ax2 == bx1) is NOT considered overlap.
    """
    ax1, ay1, az1, ax2, ay2, az2 = a
    bx1, by1, bz1, bx2, by2, bz2 = b

    return (ax1 < bx2 and ax2 > bx1) and (ay1 < by2 and ay2 > by1) and (az1 < bz2 and az2 > bz1)


def _dims_from_rotation(box: "Box", rotation) -> tuple[float, float, float]:
    """
    Convert rotation to oriented dims (L,W,H).

    Supported:
    - rotation as (L,W,H) tuple/list -> used directly
    - rotation as int code 0..5 -> mapped using the incoming box dims
    """
    # If rotation already stores actual dims, use them.
    if isinstance(rotation, (tuple, list)) and len(rotation) == 3:
        L, W, H = rotation
        return float(L), float(W), float(H)

    # Otherwise interpret as a 0..5 orientation code.
    rot = int(rotation)
    L, W, H = float(box.length), float(box.width), float(box.height)
    orientations = [
        (L, W, H),
        (L, H, W),
        (W, L, H),
        (W, H, L),
        (H, L, W),
        (H, W, L),
    ]
    return orientations[rot % 6]


def _placement_bounds(x: float, y: float, z: float, dims: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    L, W, H = dims
    return (x, y, z, x + L, y + W, z + H)


def can_place(box: "Box", placement: "Placement", container: "Container", existing_placements: list["Placement"]) -> bool:
    """
    Check if a box can be placed at the given position:
    - inside container bounds
    - no overlap with existing placements

    IMPORTANT:
    - For the NEW placement, we can derive dims from (box + placement.rotation).
    - For EXISTING placements, we must be able to read their placed dims.
      This implementation expects existing_placement.rotation to be a (L,W,H) tuple/list.
      If it is an int code, we cannot derive dims without the original Box -> return False for correctness.
    """
    # #region agent log
    DEBUG_LOG = (os.getenv("LO_DEBUG", "0") == "1")
    LOG_PATH = r"c:\Users\mohab\OneDrive\Interconnect 360\I360 Load Optimizor\.cursor\debug.log"
    def _log(loc: str, msg: str, d: dict):
        if not DEBUG_LOG:
            return
        import json, time
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"location": loc, "message": msg, "data": d, "timestamp": time.time() * 1000}) + "\n")
        except: pass
    # #endregion
    
    # Candidate dims (support tuple or int code)
    cand_dims = _dims_from_rotation(box, placement.rotation)

    # Candidate bounds
    new_bounds = _placement_bounds(float(placement.x), float(placement.y), float(placement.z), cand_dims)

    # Container bounds check
    _, _, _, new_x2, new_y2, new_z2 = new_bounds
    if new_x2 > float(container.length) or new_y2 > float(container.width) or new_z2 > float(container.height):
        return False

    # Overlap check
    for p in existing_placements:
        # We must know existing placed dims; expect tuple/list in p.rotation
        if not (isinstance(p.rotation, (tuple, list)) and len(p.rotation) == 3):
            # Without dims for existing boxes, overlap check cannot be done correctly.
            return False

        other_dims = (float(p.rotation[0]), float(p.rotation[1]), float(p.rotation[2]))
        other_bounds = _placement_bounds(float(p.x), float(p.y), float(p.z), other_dims)

        if boxes_overlap(new_bounds, other_bounds):
            return False

    return True
