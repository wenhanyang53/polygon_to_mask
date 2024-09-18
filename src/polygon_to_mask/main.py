"""Reverse a polygon to binary masks."""

from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import shapely
from PIL import Image
from rasterio.transform import from_bounds
from shapely import Polygon, get_coordinates, wkt

TWO_DIMENSION = 2
TWO_DIMENSION_MATRIX = 6
THREE_DIMENSION = 3
THREE_DIMENSION_MATRIX = 12


class PolygonMask:
    """Class to reverse polygon to binary mask."""

    def __init__(
        self,
        geom: str,
        bounds: tuple,
        width: int,
        height: int,
        image_path: Optional[str] = None,
    ) -> None:
        """Init functions.

        geom: A string of polygon ex: "POLYGON ((-73.586 45.466, xxx))"
        bounds: A tuple with coordinates in the order of (xmin, ymin, xmax, ymax)
        width: The width of the original image
        height: The height of the original image
        """
        self.geom = wkt.loads(geom)
        self.width = width
        self.height = height
        transform_data = from_bounds(*bounds, width, height)
        self.matrix = [
            transform_data.a,
            transform_data.b,
            transform_data.d,
            transform_data.e,
            transform_data.xoff,
            transform_data.yoff,
        ]
        if image_path:
            self.image = Image.open(image_path)

    def __call__(self) -> np.ndarray:
        """Call function to reverse polygon."""
        mask = self.reverse_polygons()
        if self.image:
            plt.figure(figsize=(20, 20))
            plt.imshow(self.image)
            self.show(mask)
            plt.axis("off")
            plt.show()
        return mask

    def affine_transform_reverse(self) -> Polygon:
        r"""Return a transformed geometry using an affine transformation matrix.

        The coefficient matrix is provided as a list or tuple with 6 or 12 items
        for 2D or 3D transformations, respectively.

        For 2D affine transformations, the 6 parameter matrix is::

            [a, b, d, e, xoff, yoff]

        which represents the augmented matrix::

            [x']   / a  b xoff \ [x]
            [y'] = | d  e yoff | [y]
            [1 ]   \ 0  0   1  / [1]

        or the equations for the transformed coordinates::

            x' = a * x + b * y + xoff
            y' = d * x + e * y + yoff

        For 3D affine transformations, the 12 parameter matrix is::

            [a, b, c, d, e, f, g, h, i, xoff, yoff, zoff]

        which represents the augmented matrix::

            [x']   / a  b  c xoff \ [x]
            [y'] = | d  e  f yoff | [y]
            [z']   | g  h  i zoff | [z]
            [1 ]   \ 0  0  0   1  / [1]

        or the equations for the transformed coordinates::

            x' = a * x + b * y + c * z + xoff
            y' = d * x + e * y + f * z + yoff
            z' = g * x + h * y + i * z + zoff
        """
        if len(self.matrix) == TWO_DIMENSION_MATRIX:
            ndim = TWO_DIMENSION
            a, b, d, e, xoff, yoff = self.matrix
            if self.geom.has_z:
                ndim = THREE_DIMENSION
                i = 1.0
                c = f = g = h = zoff = 0.0
        elif len(self.matrix) == THREE_DIMENSION_MATRIX:
            ndim = THREE_DIMENSION
            a, b, c, d, e, f, g, h, i, xoff, yoff, zoff = self.matrix
            if not self.geom.has_z:
                ndim = TWO_DIMENSION
        else:
            raise ValueError("'matrix' expects either 6 or 12 coefficients")
        if ndim == TWO_DIMENSION:
            A = np.array([[a, b], [d, e]], dtype=float)
            off = np.array([xoff, yoff], dtype=float)
        else:
            A = np.array([[a, b, c], [d, e, f], [g, h, i]], dtype=float)
            off = np.array([xoff, yoff, zoff], dtype=float)

        A_inv = np.linalg.inv(A)

        def _affine_coords(coords: np.ndarray) -> np.ndarray:
            return np.matmul(A_inv, (coords - off).T).T

        return shapely.transform(
            self.geom, _affine_coords, include_z=ndim == THREE_DIMENSION
        )

    def show(self, mask: np.ndarray) -> None:
        """Show the mask."""
        ax = plt.gca()
        ax.set_autoscale_on(False)
        color_mask = np.concatenate([np.array([1, 0, 0]), [0.7]])

        if mask is not None:
            img = np.ones(
                (
                    mask[0].shape[0],
                    mask[0].shape[1],
                    4,
                )
            )
            img[:, :, 3] = 0
            for m in mask:
                img[m] = color_mask
            ax.imshow(img)

    def reverse_polygons(self) -> np.ndarray:
        """Reverse polygon to binary mask."""
        polygon = self.affine_transform_reverse()
        coords = get_coordinates(polygon).round().astype(int)

        mask = np.zeros((self.height, self.width), dtype="uint8")
        cv2.fillPoly(mask[:, :], [coords], 1)
        mask = mask.astype(bool)

        return mask
