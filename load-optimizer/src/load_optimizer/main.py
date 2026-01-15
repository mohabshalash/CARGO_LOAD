from __future__ import annotations

from load_optimizer.models import Container, Box, Placement
from load_optimizer.packing.first_fit import pack_boxes


def placement_volume(placement: Placement) -> float:
    """
    Placement.rotation is stored as oriented dims: (L, W, H).
    Volume = L * W * H.
    """
    L, W, H = placement.rotation
    return float(L) * float(W) * float(H)


def compute_used_volume(placements: list[Placement]) -> float:
    """Total packed volume."""
    return sum(placement_volume(p) for p in placements)


def compute_fill_rate(container: Container, placements: list[Placement]) -> float:
    """Fill rate = used_volume / container_volume."""
    container_volume = float(container.length) * float(container.width) * float(container.height)
    if container_volume == 0.0:
        return 0.0
    return compute_used_volume(placements) / container_volume


def run_case(container: Container, boxes: list[Box]) -> None:
    print("\n" + "=" * 60)
    print(f"📦 TEST CONTAINER: {container.length} x {container.width} x {container.height}")

    result = pack_boxes(container, boxes)
    placements = result.placements
    unpacked = result.unpacked
    fill_rate = result.fill_rate
    used_volume = result.used_volume
    container_volume = result.container_volume


    used_volume = compute_used_volume(placements)
    container_volume = float(container.length) * float(container.width) * float(container.height)
    fill_rate = compute_fill_rate(container, placements)

    packed_ids = [p.box_id for p in placements]
    unpacked_ids = [b.id for b in unpacked]

    print("✅ Packed   :", packed_ids if packed_ids else "(none)")
    print("❌ Unpacked :", unpacked_ids if unpacked_ids else "(none)")

    print("\n📦 PLACEMENTS:")
    for p in placements:
        print(" ", p)

    print("\n📊 FILL RATE:")
    print(f"  Used volume      : {used_volume:.2f}")
    print(f"  Container volume : {container_volume:.2f}")
    print(f"  Fill rate        : {fill_rate * 100:.2f}%")


def main() -> None:
    print("✅ main() started")

    boxes = [
        Box(id="A", length=8, width=8, height=8),
        Box(id="B", length=3, width=3, height=3),
        Box(id="C", length=12, width=1, height=1),
    ]

    containers = [
        Container(length=10, width=10, height=10),
        Container(length=11, width=11, height=11),
        Container(length=12, width=10, height=10),
        Container(length=10, width=12, height=10),
        Container(length=10, width=10, height=12),
    ]

    for c in containers:
        run_case(c, boxes)


if __name__ == "__main__":
    main()



