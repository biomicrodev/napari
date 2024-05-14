import math
from collections import namedtuple

import numpy as np
import numpy.typing as npt

from napari.layers.shapes._shapes_models.shape import Shape
from napari.layers.shapes._shapes_utils import create_box
from napari.utils.translations import trans


def _wrap_angle(f: float | npt.NDArray) -> float | npt.NDArray:
    return (f + math.pi) % (2 * math.pi) - math.pi


def _make_bow(
    head: npt.NDArray, tail: npt.NDArray, *, n_seg=100, span=90.0
) -> npt.NDArray:
    hspan = math.radians(span) / 2

    delta = head - tail
    length = math.hypot(delta[1], delta[0])
    angle = math.atan2(delta[1], delta[0])

    lead_angles = np.linspace(0, hspan, n_seg // 2)
    lead_angles += angle
    lead_angles = _wrap_angle(lead_angles)
    lead_x = length * np.cos(lead_angles) + tail[0]
    lead_y = length * np.sin(lead_angles) + tail[1]
    lead_arc = np.stack([lead_x, lead_y]).T

    trail_angles = np.linspace(-hspan, 0, n_seg // 2)
    trail_angles += angle
    trail_angles = _wrap_angle(trail_angles)
    trail_x = length * np.cos(trail_angles) + tail[0]
    trail_y = length * np.sin(trail_angles) + tail[1]
    trail_arc = np.stack([trail_x, trail_y]).T

    # TODO: right now this doesn't work so well because the path of the bow has to be
    #  one continuous path and has sharp angles
    tail = tail[np.newaxis, ...]
    points = np.concatenate([trail_arc, tail, lead_arc], axis=0)
    return points


Point2D = namedtuple("Point2D", ("x", "y"))


class Bow(Shape):
    """Class for a single bow
    Dev notes: this part is taken from line, but we can't exactly duplicate the logic,
    so we subclass `Shape` instead and make sure we are up-to-date with it.

    Parameters
    ----------
    data : (2, 2) array
        A (2, 2) array specifying the head and tail vertices.
        stored as (2, D), where D  2
    edge_width : float
        thickness of lines and edges
    z_index : int
        Specifier of z order priority. Shapes with higher z order are displayed on top
        of others.
    dims_order : (D,) list
        Order that the dimensions are to be rendered in

    Bow-specific properties
    ----------
    span : float
        Angle, in degrees, of how big to render bow; must be between 0 and 360.
    """

    def __init__(
        self,
        data,
        *,
        edge_width=1,
        z_index=0,
        dims_order=None,
        ndisplay=2,
    ):
        if dims_order is not None:
            pass
            # hide the below warning for now... appears every time a bow is drawn
            # warnings.warn(f"The Bow shape is only supported in 2D for now")
        dims_order = list(range(2))

        super().__init__(
            edge_width=edge_width,
            z_index=z_index,
            dims_order=dims_order,
            ndisplay=ndisplay,
        )

        self._filled = False
        self._closed = False
        self._use_face_vertices = False

        self.data = data
        self.name = "bow"

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        data = np.array(data).astype(float)

        if len(self.dims_order) != data.shape[1]:
            self._dims_order = list(range(data.shape[1]))

        if len(data) != 2:
            raise ValueError(
                trans._(
                    "Data shape does not match a bow. Bow expects two corner vertices, {number} provided.",
                    deferred=True,
                    number=len(data),
                )
            )

        self._data = data
        self._update_displayed_data()

    def _update_displayed_data(self):
        """Update the data that is to be displayed."""
        # Build boundary vertices with num_segments

        head = self.data_displayed[0, :]
        tail = self.data_displayed[1, :]
        bow = _make_bow(head, tail, n_seg=100)

        self._set_meshes(bow, face=False, closed=False)
        self._box = create_box(self.data_displayed)

        data_not_displayed = self.data[:, self.dims_not_displayed]
        self.slice_key = np.round(
            [
                np.min(data_not_displayed, axis=0),
                np.max(data_not_displayed, axis=0),
            ]
        ).astype("int")

    def shift(self, shift):
        """
        Not sure why `shift` and `transform` need to trigger `update_displayed_data`
        like this... otherwise bow behaves unexpectedly
        """
        super().shift(shift)
        self._update_displayed_data()

    def transform(self, transform):
        super().transform(transform)
        self._update_displayed_data()

    @property
    def head(self) -> Point2D:
        head = self.data[0, :]
        return Point2D(x=head[0], y=head[1])

    @property
    def tail(self) -> Point2D:
        tail = self.data[1, :]
        return Point2D(x=tail[0], y=tail[1])
