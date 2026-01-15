from load_optimizer.models import Container, Box
from load_optimizer.packing.multi_container import pack_multiple
from load_optimizer.geometry import boxes_overlap


def bounds_from_placement(p):
    L, W, H = p.rotation
    x1, y1, z1 = float(p.x), float(p.y), float(p.z)
    x2, y2, z2 = x1 + float(L), y1 + float(W), z1 + float(H)
    return (x1, y1, z1, x2, y2, z2)


def assert_within_container(container, placements):
    for p in placements:
        x1, y1, z1, x2, y2, z2 = bounds_from_placement(p)
        assert x1 >= 0 and y1 >= 0 and z1 >= 0
        assert x2 <= float(container.length)
        assert y2 <= float(container.width)
        assert z2 <= float(container.height)


def assert_no_overlaps(placements):
    bounds = [bounds_from_placement(p) for p in placements]
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            assert not boxes_overlap(bounds[i], bounds[j])


def test_pack_multiple_60_cubes():
    container = Container(length=10, width=10, height=10)
    boxes = [Box(id=f"B{i}", length=3, width=3, height=3) for i in range(60)]

    plans, leftover = pack_multiple(container, boxes)

    # 10/3 = 3 per axis => 27 cubes per container (grid max)
    packed_counts = [len(plan.placements) for plan in plans]

    assert packed_counts[0] == 27
    assert packed_counts[1] == 27
    assert packed_counts[2] == 6
    assert leftover == []

    # Invariants per container
    for plan in plans:
        assert_within_container(container, plan.placements)
        assert_no_overlaps(plan.placements)
