from __future__ import annotations
from load_optimizer.models import Container, Placement


def placement_volume(p: Placement) -> float:
    L, W, H = p.rotation  # rotation is (L,W,H)
    return float(L) * float(W) * float(H)


def compute_metrics(container: Container, placements: list[Placement]) -> tuple[float, float, float]:
    used_volume = sum(placement_volume(p) for p in placements)
    container_volume = float(container.length) * float(container.width) * float(container.height)
    fill_rate = 0.0 if container_volume == 0 else used_volume / container_volume
    return used_volume, container_volume, fill_rate
