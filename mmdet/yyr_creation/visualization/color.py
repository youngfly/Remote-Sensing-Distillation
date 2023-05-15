# Copyright (c) Open-MMLab. All rights reserved.
from enum import Enum

import numpy as np

from mmcv.utils import is_str


class Color(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)
    deepskyblue = (0, 191, 255)
    gold = (255, 215, 0)
    indianred = (205, 92, 92)
    orange = (255, 165, 0)
    hotpink = (255, 105, 180)
    purple = (160, 32, 240)
    seagrean = (46, 139, 87)
    aquamarine1 = (127, 255, 212)
    peru = (205, 133, 63)
    grey21 = (54, 54, 54)
    darkred = (139, 0, 0)
    violetRed4 = (139, 34, 82)



def color_val(color):
    """Convert various input to color tuples.

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if is_str(color):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')
