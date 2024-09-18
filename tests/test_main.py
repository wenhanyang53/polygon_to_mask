"""Test for reverse polygon."""

from shapely import get_coordinates, wkt

from polygon_to_mask.main import PolygonMask

INPUT_POLYGON = "POLYGON ((-73.58695186851953 45.4667147163283, -73.58698255872136 45.466712016808884, -73.5870746293268 45.46665532690125, -73.58716286365704 45.466628331707135, -73.58733549604226 45.46654194708597, -73.58734316859272 45.466517651411266, -73.5872012264093 45.46637997592129, -73.58716669993227 45.46637997592129, -73.58681376261136 45.46655814420244, -73.5868329439875 45.46659863699361, -73.58695186851953 45.4667147163283))"
OUTPUT_POLYGON = "POLYGON ((1607 786, 1599 787, 1575 808, 1552 818, 1507 850, 1505 859, 1542 910, 1551 910, 1643 844, 1638 829, 1607 786))"
BOUNDS = (
    -73.5931167628097,
    45.463216139171195,
    -73.586476170391364,
    45.468836538585613,
)
WIDTH = 1731
HEIGHT = 2082


def test_affine_transform_reverse() -> None:
    """Test for reverse affine function."""
    pm = PolygonMask(INPUT_POLYGON, BOUNDS, WIDTH, HEIGHT)
    output_polygon = wkt.loads(OUTPUT_POLYGON)
    output_result = get_coordinates(pm.affine_transform_reverse()).round().astype(int)
    output_assert = get_coordinates(output_polygon).astype(int)

    assert (output_result == output_assert).all()
