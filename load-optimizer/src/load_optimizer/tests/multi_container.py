from __future__ import annotations

from load_optimizer.models import Container, Box, PackingResult
from load_optimizer.packing.first_fit import pack_boxes


def pack_multiple(
    container: Container,
    boxes: list[Box],
    max_containers: int = 100,
) -> tuple[list[PackingResult], list[Box]]:
    """
    Pack boxes into as many identical containers as needed.

    Returns:
      - plans: list of PackingResult, one per container
      - leftover: boxes that could not be packed (if no progress)
    """
    remaining = list(boxes)
    plans: list[PackingResult] = []

    for _ in range(max_containers):
        if not remaining:
            break

        result = pack_boxes(container, remaining)
        plans.append(result)

        # Progress guard: if we couldn't place anything, stop to avoid infinite loops
        if not result.placements:
            remaining = result.unpacked
            break

        remaining = result.unpacked

    return plans, remaining
