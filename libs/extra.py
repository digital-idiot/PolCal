import numpy as np
import pandas as pd
from rasterio.windows import Window
from ..errors import InconsistentException
from typing import Sequence, Tuple, Union
from pydantic.types import PositiveInt, NonNegativeInt
from pydantic_core.core_schema import FieldValidationInfo
from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = ["forward_shape", "backward_shape", "generate_seamless_tiles"]


class ValidateArgsBackwardShape(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    kernel_size: Union[
        PositiveInt, Sequence[PositiveInt]
    ] = (3, 3),
    dilation: Union[
        PositiveInt, Sequence[PositiveInt]
    ] = (1, 1),
    stride: Union[
        PositiveInt, Sequence[PositiveInt]
    ] = (1, 1),
    padding: Union[
        NonNegativeInt, Sequence[NonNegativeInt]
    ] = (0, 0)
    shape: Union[
        PositiveInt, Sequence[PositiveInt]
    ]

    @classmethod
    @field_validator(
        'kernel_size',
        'dilation',
        'stride',
        'padding',
        mode="after",
        check_fields=True
    )
    def check_sequence(
            cls,
            v: Union[int, Sequence[int]],
    ):
        return np.array(v, dtype=int)

    @classmethod
    @field_validator(
        'shape',
        mode="after",
        check_fields=True
    )
    def check_shape(cls, v: Union[PositiveInt, Sequence[PositiveInt]]):
        return np.array(v, dtype=int)


class ValidateArgsForwardShape(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    kernel_size: Union[
        PositiveInt, Sequence[PositiveInt]
    ] = (3, 3),
    dilation: Union[
        PositiveInt, Sequence[PositiveInt]
    ] = (1, 1),
    stride: Union[
        PositiveInt, Sequence[PositiveInt]
    ] = (1, 1),
    padding: Union[
        NonNegativeInt, Sequence[NonNegativeInt]
    ] = (0, 0)
    shape: Union[
        PositiveInt, Sequence[PositiveInt]
    ]

    @classmethod
    @field_validator(
        'kernel_size',
        'dilation',
        'stride',
        'padding',
        mode="after",
        check_fields=True
    )
    def check_sequence(
            cls,
            v: Union[int, Sequence[int]]
    ):
        return np.array(v, dtype=int)

    @classmethod
    @field_validator(
        'shape',
        mode="after",
        check_fields=True
    )
    def check_shape(
            cls,
            v: Union[PositiveInt, Sequence[PositiveInt]],
            info: FieldValidationInfo
    ):
        v = np.array(v, dtype=int)
        if (
            'kernel_size' in info.data
        ) and (
            'dilation' in info.data
        ) and (
            'stride' in info.data
        ) and (
            'zero_padding' in info.data
        ):
            i = np.array(v)
            k = np.array(info.data['kernel_size'])
            d = np.array(info.data['dilation'])
            p = np.array(info.data['padding'])
            s = np.array(info.data['stride'])
            delta = (d * (k - 1)) - (2 * p) - s + 1
            if np.any(delta > i):
                raise InconsistentException(
                    f"Image size {v} is " +
                    f"inconsistent with the " +
                    f"following combination of arguments:\n" +
                    f"\tkernel_size: {info.data['kernel_size']}\n" +
                    f"\tdilation: {info.data['dilation']}\n" +
                    f"\tstride: {info.data['stride']}\n" +
                    f"\tzero_padding: {info.data['padding']}"
                )


def forward_shape(
        shape: Union[
            PositiveInt, Sequence[PositiveInt]
        ],
        kernel_size: Union[
            PositiveInt, Sequence[PositiveInt]
        ] = (3, 3),
        dilation: Union[
            PositiveInt, Sequence[PositiveInt]
        ] = (1, 1),
        stride: Union[
            PositiveInt, Sequence[PositiveInt]
        ] = (1, 1),
        padding: Union[
            NonNegativeInt, Sequence[NonNegativeInt]
        ] = (0, 0)
):
    args = ValidateArgsForwardShape(
        shape=shape,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=stride,
        padding=padding
    )
    shape = args.shape
    kernel_size = args.kernel_size
    dilation = args.dilation
    stride = args.stride
    padding = args.padding
    target_shape = np.floor(
        (
            (
                shape + (2 * padding) - (
                    dilation * (kernel_size - 1)
                ) - 1
            ) / stride
        ) + 1
    ).astype(int).tolist()
    return tuple(target_shape) if isinstance(
        target_shape, Sequence
    ) else target_shape


def backward_shape(
        shape: Union[
            PositiveInt, Sequence[PositiveInt]
        ],
        kernel_size: Union[
            PositiveInt, Sequence[PositiveInt]
        ] = (3, 3),
        dilation: Union[
            PositiveInt, Sequence[PositiveInt]
        ] = (1, 1),
        stride: Union[
            PositiveInt, Sequence[PositiveInt]
        ] = (1, 1),
        padding: Union[
            NonNegativeInt, Sequence[NonNegativeInt]
        ] = (0, 0)
):
    args = ValidateArgsBackwardShape(
        shape=shape,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=stride,
        padding=padding
    )
    shape = args.shape
    kernel_size = args.kernel_size
    dilation = args.dilation
    stride = args.stride
    padding = args.padding
    src_shape = np.ceil(
        (
            (shape - 1) * stride
        ) + (2 * padding) + (
            dilation * (kernel_size - 1)
        ) + 1
    ).astype(int).tolist()
    return tuple(src_shape) if isinstance(
        src_shape, Sequence
    ) else src_shape


class ValidateArgsWindows(BaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    image_size: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt]
    ] = Field(...)
    kernel_size: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt]
    ] = (3, 3)
    dilation: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt]
    ] = (1, 1)
    tile_size: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt]
    ] = Field(...)

    @classmethod
    @field_validator(
        'image_size'
        'kernel_size',
        'dilation',
        mode="after",
        check_fields=True
    )
    def check_tuple(
            cls,
            v: Union[int, Tuple[int, int]],
    ):
        if isinstance(v, int):
            v = (v, v)
        return v

    @classmethod
    @field_validator(
        'tile_size',
        mode="after",
        check_fields=True
    )
    def check_tile(
            cls,
            v: Union[int, Tuple[int, int]],
            info: FieldValidationInfo
    ):
        if isinstance(v, int):
            v = (v, v)
        if (
            'kernel_size' in info.data
        ) and (
            'dilation' in info.data
        ) and (
            'image_size' in info.data
        ):
            t = np.array(v)
            i = np.array(info.data['image_size'])
            k = np.array(info.data['kernel_size'])
            d = np.array(info.data['dilation'])
            k = (d * (k - 1)) + 1
            if np.any(k > t):
                raise InconsistentException(
                    f"Tile size {v} is inconsistent with following " +
                    f"combination of arguments:\n" +
                    f"\tkernel_size: {info.data['kernel_size']}\n" +
                    f"\tdilation: {info.data['dilation']}"
                )
            if np.any(t > i):
                raise InconsistentException(
                    f"Tile size {v} is larger than " +
                    f"image size {info.data['image_size']}"
                )
        return v


def generate_seamless_tiles(
        image_size: Union[
            PositiveInt, Tuple[PositiveInt, PositiveInt]
        ],
        tile_size: Union[
            PositiveInt, Tuple[PositiveInt, PositiveInt]
        ],
        kernel_size: Union[
            PositiveInt, Tuple[PositiveInt, PositiveInt]
        ] = (3, 3),
        dilation: Union[
            PositiveInt, Tuple[PositiveInt, PositiveInt]
        ] = (1, 1),
):
    args = ValidateArgsWindows(
        image_size=image_size,
        tile_size=tile_size,
        kernel_size=kernel_size,
        dilation=dilation
    )

    sih, siw = args.image_size
    sth, stw = args.tile_size
    kh, kw = args.kernel_size

    tth, ttw = forward_shape(
        shape=(sth, stw),
        kernel_size=(kh, kw),
        dilation=args.dilation,
        stride=(1, 1),
        padding=(0, 0)
    )

    dh, dw = (sth - tth), (stw - ttw)
    pt, pl = (dh // 2), (dw // 2)
    pb, pr = (dh - pl), (dw - pt)

    th_markers = np.arange(0, sih, tth)
    tw_markers = np.arange(0, siw, ttw)

    th_markers[-1] = sih - tth
    tw_markers[-1] = siw - ttw

    th_markers, tw_markers = np.meshgrid(
        th_markers,
        tw_markers,
        copy=False,
        sparse=False,
        indexing='xy'
    )

    pad_left = np.zeros_like(th_markers)
    pad_right = np.zeros_like(th_markers)
    pad_top = np.zeros_like(th_markers)
    pad_bottom = np.zeros_like(th_markers)

    pad_left[:, 0] = pl
    pad_right[:, -1] = pr
    pad_top[0, :] = pt
    pad_bottom[-1, :] = pb

    th_markers = th_markers.ravel()
    tw_markers = tw_markers.ravel()
    pad_left = pad_left.ravel()
    pad_right = pad_right.ravel()
    pad_top = pad_top.ravel()
    pad_bottom = pad_bottom.ravel()

    sh_markers = th_markers - pl
    sw_markers = tw_markers - pt

    image_window = Window(row_off=0, col_off=0, height=sih, width=siw)
    make_windows = np.vectorize(
        pyfunc=lambda row_off, col_off, height, width: Window(
            row_off, col_off, height, width
        ).intersection(image_window),
        excluded={"height", "width"}
    )

    src_windows = make_windows(
        row_off=sw_markers,
        col_off=sh_markers,
        height=sth,
        width=stw
    )
    target_windows = make_windows(
        row_off=tw_markers,
        col_off=th_markers,
        height=tth,
        width=ttw
    )

    return pd.DataFrame(
        data={
            "src_windows": src_windows,
            "dst_windows": target_windows,
            "pad_left": pad_left,
            "pad_right": pad_right,
            "pad_top": pad_top,
            "pad_bottom": pad_bottom
        }
    )
