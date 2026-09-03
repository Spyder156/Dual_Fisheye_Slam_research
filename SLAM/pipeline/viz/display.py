"""Single source of truth for how frames are DISPLAYED.

The Hilti sensor is mounted inverted: world-up projects onto image +y (down) on
both lenses, measured at +0.99. The calibration encodes that, so all GEOMETRY
runs on raw pixels and is correct. Only humans need the image turned round.

Every visualisation in this repo must go through here. This exists because the
rotation was fixed in one viz script and then silently missing from the next
two, which is exactly the sort of thing that wastes someone's time twice.
"""
import cv2

ROT180 = True          # true for the Hilti rig; set per-dataset if that changes


def show(img):
    """Raw sensor image -> image as a human should see it."""
    return cv2.rotate(img, cv2.ROTATE_180) if ROT180 else img


def pt(x, y, w, h):
    """Raw pixel coords -> display coords, matching show()."""
    return (w - 1 - x, h - 1 - y) if ROT180 else (x, y)
