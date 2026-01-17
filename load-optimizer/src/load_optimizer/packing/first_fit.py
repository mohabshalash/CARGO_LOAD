# src/load_optimizer/packing/first_fit.py

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from load_optimizer.models import Box, Container, Placement
from load_optimizer.geometry import can_place

from load_optimizer.models import PackingResult
from load_optimizer.metrics import compute_metrics
def box_volume(box: Box) -> float:
    return float(box.length) * float(box.width) * float(box.height)


def rotations_6(box: Box) -> list[tuple[float, float, float, int]]:
    """
    Return the 6 axis-aligned orientations plus a rotation code 0..5.
    rotation code meaning:
      0:(L,W,H) 1:(L,H,W) 2:(W,L,H) 3:(W,H,L) 4:(H,L,W) 5:(H,W,L)
    """
    L, W, H = float(box.length), float(box.width), float(box.height)
    dims = [
        (L, W, H, 0),
        (L, H, W, 1),
        (W, L, H, 2),
        (W, H, L, 3),
        (H, L, W, 4),
        (H, W, L, 5),
    ]
    # Optional: remove duplicates (for cubes, many rotations are identical)
    seen = set()
    out: list[tuple[float, float, float, int]] = []
    for a, b, c, r in dims:
        key = (a, b, c)
        if key not in seen:
            seen.add(key)
            out.append((a, b, c, r))
    return out


def generate_candidate_points(placements: list[Placement], boxes_by_id: dict[str, Box]) -> list[tuple[float, float, float]]:
    """
    Extreme-points style candidates:
      start with origin,
      add (x+L, y, z), (x, y+W, z), (x, y, z+H) for each placed box.
    IMPORTANT: This version assumes Placement.rotation stores (L,W,H) dims.
    """
    points: set[tuple[float, float, float]] = {(0.0, 0.0, 0.0)}

    for p in placements:
        # p.rotation is stored as (L, W, H)
        L = float(p.rotation[0])
        W = float(p.rotation[1])
        H = float(p.rotation[2])

        x = float(p.x)
        y = float(p.y)
        z = float(p.z)

        points.add((x + L, y, z))
        points.add((x, y + W, z))
        points.add((x, y, z + H))

    # Sort by (y, z, x) to enforce floor-first placement: all y=0 candidates before y>0
    return sorted(points, key=lambda t: (t[1], t[2], t[0]))


def compute_fill_rate(container: Container, placements: list[Placement], boxes_by_id: dict[str, Box]) -> float:
    container_vol = float(container.length) * float(container.width) * float(container.height)
    used_vol = sum(box_volume(boxes_by_id[p.box_id]) for p in placements)
    return 0.0 if container_vol == 0 else used_vol / container_vol

def total_weight(
    placements: list[Placement],
    boxes_by_id: dict[str, Box],
) -> float:
    total = 0.0
    for p in placements:
        box_weight = boxes_by_id[p.box_id].weight
        if box_weight is not None:
            total += box_weight
    return total

def pack_boxes(container: Container, boxes: list[Box]) -> PackingResult:
    """
    First-fit packer that accepts the FIRST feasible placement for each box.
    - Explores candidate points + 6 rotations
    - Accepts first placement that passes geometry and weight constraints
    - Appends AT MOST ONE placement per box
    - Deterministic (no randomness)
    - Fill-rate scoring does NOT block feasibility
    """
    # #region agent log
    LOG_PATH = r"c:\Users\mohab\OneDrive\Interconnect 360\I360 Load Optimizor\.cursor\debug.log"
    DEBUG_LOG = (os.getenv("LO_DEBUG", "0") == "1")
    def _log(loc: str, msg: str, d: dict):
        if not DEBUG_LOG:
            return
        import json, time
        try:
            with open(LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"location": loc, "message": msg, "data": d, "timestamp": time.time() * 1000}) + "\n")
        except: pass
    if DEBUG_LOG:
        _log("first_fit.py:85", "pack_boxes entry", {"boxes_count": len(boxes), "container_dims": [container.length, container.width, container.height], "max_payload": container.max_payload})
    # #endregion
    
    # Sort big boxes first (helps fill)
    boxes_sorted = sorted(boxes, key=box_volume, reverse=True)
    boxes_by_id = {b.id: b for b in boxes_sorted}
    # #region agent log
    if DEBUG_LOG and boxes_sorted:
        _log("first_fit.py:96", "First box", {"id": boxes_sorted[0].id, "dims": [boxes_sorted[0].length, boxes_sorted[0].width, boxes_sorted[0].height], "weight": boxes_sorted[0].weight})
    # #endregion

    placements: list[Placement] = []
    unpacked: list[Box] = []
    current_weight = 0.0

    for box in boxes_sorted:
        # Build candidates from what we already placed
        candidate_points = generate_candidate_points(placements, boxes_by_id)

        placed = False

        # Try each candidate point and rotation - accept FIRST feasible placement
        for (x, y, z) in candidate_points:
            if placed:
                break
                
            for (l, w, h, rot_code) in rotations_6(box):
                candidate = Placement(
                    box_id=box.id,
                    x=float(x),
                    y=float(y),
                    z=float(z),
                    rotation=(float(l), float(w), float(h)),  # store oriented dims
                )
                # Check geometry feasibility
                can_place_result = can_place(box, candidate, container, placements)
                if not can_place_result:
                    continue

                # Check weight constraint
                box_weight = float(box.weight) if box.weight is not None else 0.0
                if container.max_payload is not None:
                    if current_weight + box_weight > container.max_payload:
                        continue

                # First feasible placement found - accept it immediately
                placements.append(candidate)
                current_weight += box_weight
                placed = True
                # #region agent log
                if DEBUG_LOG:
                    _log("first_fit.py:135", "Box placed", {"box_id": box.id, "placement": [x, y, z]})
                # #endregion
                break

        if not placed:
            unpacked.append(box)
            # #region agent log
            if DEBUG_LOG:
                _log("first_fit.py:138", "Box unpacked", {"box_id": box.id})
            # #endregion

    used_volume, container_volume, fill_rate = compute_metrics(container, placements)

    return PackingResult(
        placements=placements,
        unpacked=unpacked,
        used_volume=used_volume,
        container_volume=container_volume,
        fill_rate=fill_rate,
    )

