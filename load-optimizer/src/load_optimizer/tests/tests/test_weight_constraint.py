from load_optimizer.models import Container, Box
from load_optimizer.packing.first_fit import pack_boxes

def test_weight_limit_blocks_second_box():
    container = Container(
        length=10,
        width=10,
        height=10,
        max_payload=1000  # kg
    )

    boxes = [
        Box(id="A", length=2, width=2, height=2, weight=600),
        Box(id="B", length=2, width=2, height=2, weight=600),
    ]

    result = pack_boxes(container, boxes)

    packed_ids = [p.box_id for p in result.placements]
    unpacked_ids = [b.id for b in result.unpacked]

    assert packed_ids == ["A"]
    assert unpacked_ids == ["B"]
