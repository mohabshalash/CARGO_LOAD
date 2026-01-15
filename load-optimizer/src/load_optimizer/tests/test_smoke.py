from load_optimizer.models import Container, Box
from load_optimizer.packing.first_fit import pack_boxes
from load_optimizer.geometry import boxes_overlap


def bounds_from_placement(p):
    L, W, H = p.rotation
    x1, y1, z1 = float(p.x), float(p.y), float(p.z)
    x2, y2, z2 = x1 + L, y1 + W, z1 + H
    return (x1, y1, z1, x2, y2, z2)


def assert_within_container(container, placements):
    for p in placements:
        x1, y1, z1, x2, y2, z2 = bounds_from_placement(p)
        assert x1 >= 0 and y1 >= 0 and z1 >= 0
        assert x2 <= container.length
        assert y2 <= container.width
        assert z2 <= container.height


def assert_no_overlaps(placements):
    bounds = [bounds_from_placement(p) for p in placements]
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            assert not boxes_overlap(bounds[i], bounds[j])


def test_case_10_cube():
    container = Container(length=10, width=10, height=10)
    boxes = [
        Box(id="A", length=8, width=8, height=8),
        Box(id="B", length=3, width=3, height=3),
        Box(id="C", length=12, width=1, height=1),
    ]

    result = pack_boxes(container, boxes)
    placements = result.placements
    unpacked = result.unpacked

    packed_ids = [p.box_id for p in placements]
    unpacked_ids = [b.id for b in unpacked]

    assert packed_ids == ["A"]
    assert unpacked_ids == ["B", "C"]

    assert_within_container(container, placements)
    assert_no_overlaps(placements)


def test_case_11_cube():
    container = Container(length=11, width=11, height=11)
    boxes = [
        Box(id="A", length=8, width=8, height=8),
        Box(id="B", length=3, width=3, height=3),
        Box(id="C", length=12, width=1, height=1),
    ]

    result = pack_boxes(container, boxes)
    placements = result.placements
    unpacked = result.unpacked

    packed_ids = [p.box_id for p in placements]
    unpacked_ids = [b.id for b in unpacked]

    assert packed_ids == ["A", "B"]
    assert unpacked_ids == ["C"]

    assert_within_container(container, placements)
    assert_no_overlaps(placements)


def test_case_12_10_10():
    container = Container(length=12, width=10, height=10)
    boxes = [
        Box(id="A", length=8, width=8, height=8),
        Box(id="B", length=3, width=3, height=3),
        Box(id="C", length=12, width=1, height=1),
    ]

    result = pack_boxes(container, boxes)
    placements = result.placements
    unpacked = result.unpacked

    packed_ids = [p.box_id for p in placements]
    unpacked_ids = [b.id for b in unpacked]

    assert packed_ids == ["A", "B", "C"]
    assert unpacked_ids == []

    assert_within_container(container, placements)
    assert_no_overlaps(placements)
