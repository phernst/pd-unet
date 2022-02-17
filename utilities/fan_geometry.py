from typing import NamedTuple


class FanGeometry(NamedTuple):
    source_distance: float
    det_distance: float
    det_count: int
    det_spacing: float


def default_fan_geometry() -> FanGeometry:
    return FanGeometry(
        source_distance=400.,
        det_distance=150.,
        det_count=511,
        det_spacing=1.0,
    )
