import torch
import numpy as np
from einops import rearrange, einsum
from jaxtyping import Complex, Float, Bool, Array
from pydantic_core.core_schema import FieldValidationInfo
from typing import Sequence, Tuple, Union, Optional, Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.types import PositiveInt, NonNegativeInt, PositiveFloat
from ..errors import (
    DimensionException,
    ShapeException,
    DTypeException,
    DeviceException,
    InconsistentException
)

__all__ = [
    "convolve2d_patches",
    "patched_cross_correlation2d",
    "covariance_matrix",
    "masked_mean",
    "masked_batch_inverse"
]


class ValidateArgsConv2DPatches(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    kernel_size: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt]
    ] = (5, 5)
    dilation: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt]
    ] = (1, 1)
    stride: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt]
    ] = (1, 1)
    zero_padding: Union[
        NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]
    ] = (0, 0)
    tensor: torch.Tensor = Field(...)

    @classmethod
    @field_validator('tensor', mode="after", check_fields=True)
    def check_tensor(
            cls,
            v: Union[Float[Array, "b c h w"], Complex[Array, "b c h w"]],
            info: FieldValidationInfo
    ):
        if v.ndim != 4:
            DimensionException(
                "Unsupported tensor shape !\n" +
                "\tExpected a `4D` tensor " +
                f"but got `{v.ndim}D` tensor instead."
            )
        if not (
                torch.is_complex(input=v) or torch.is_floating_point(input=v)
        ):
            raise DTypeException(
                "Unsupported tensor dtype!\n" +
                "\tExpected a `float` or `complex` valued  tensor " +
                f"but got a tenosr of dtype {v.dtype}."
            )
        image_shape = tuple(v.shape)[-2:]
        if (
                'kernel_size' in info.data
        ) and (
                'dilation' in info.data
        ) and (
                'stride' in info.data
        ) and (
                'zero_padding' in info.data
        ):
            i = np.array(image_shape)
            k = np.array(info.data['kernel_size'])
            d = np.array(info.data['dilation'])
            p = np.array(info.data['zero_padding'])
            s = np.array(info.data['stride'])
            delta = (d * (k - 1)) - (2 * p) - s + 1
            if np.any(delta > i):
                raise InconsistentException(
                    f"Image size {image_shape} is " +
                    f"inconsistent with the " +
                    f"following combination of arguments:\n" +
                    f"\tkernel_size: {info.data['kernel_size']}\n" +
                    f"\tdilation: {info.data['dilation']}\n" +
                    f"\tstride: {info.data['stride']}\n" +
                    f"\tzero_padding: {info.data['zero_padding']}"
                )
        return v

    @classmethod
    @field_validator(
        'kernel_size'
        'dilation',
        'stride',
        'zero_padding',
        mode="after",
        check_fields=True
    )
    def check_tuple(cls, v: Union[int, Tuple[int, int]]):
        if isinstance(v, int):
            v = (v, v)
        return v


def convolve2d_patches(
        tensor: Union[
            Float[Array, "#b #c #h #w"], Complex[Array, "#b #c #h #w"]
        ],
        kernel_size: Union[
            PositiveInt, Tuple[PositiveInt, PositiveInt]
        ] = (3, 3),
        dilation: Union[
            PositiveInt, Tuple[PositiveInt, PositiveInt]
        ] = (1, 1),
        stride: Union[
            PositiveInt, Tuple[PositiveInt, PositiveInt]
        ] = (1, 1),
        zero_padding: Union[
            NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]
        ] = (0, 0)
) -> Union[
    Float[Array, "#b #c #h #w #k1 #k2"],
    Complex[Array, "#b #c #h #w #k1 #k2"]
]:
    args = ValidateArgsConv2DPatches(
        tensor=tensor,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=stride,
        zero_padding=zero_padding
    )
    tensor = args.tensor
    kernel_size = args.kernel_size
    dilation = args.dilation
    stride = args.stride
    zero_padding = args.zero_padding
    unfold = torch.nn.Unfold(
        kernel_size=kernel_size,
        dilation=dilation,
        padding=zero_padding,
        stride=stride
    )
    b, c, h, w = tuple(tensor.shape)
    spatial_size = torch.tensor(
        data=(h, w),
        dtype=torch.int32,
        device=tensor.device
    )
    k1, k2 = kernel_size
    kernel_size = torch.tensor(
        data=kernel_size,
        dtype=torch.int32,
        device=tensor.device,
        requires_grad=False
    )
    dilation = torch.tensor(
        data=dilation,
        dtype=torch.int32,
        device=tensor.device,
        requires_grad=False
    )
    stride = torch.tensor(
        data=stride,
        dtype=torch.int32,
        device=tensor.device,
        requires_grad=False
    )
    zero_padding = torch.tensor(
        data=zero_padding,
        dtype=torch.int32,
        device=tensor.device,
        requires_grad=False
    )
    h, w = (
        (
            (
                spatial_size + (
                    2 * zero_padding
                ) - (
                    dilation * (kernel_size - 1)
                ) - 1
            ) / stride
        ) + 1
    ).floor().int().tolist()
    tensor = rearrange(
        tensor=unfold(tensor),
        pattern="b (c k1 k2) (h w) -> b c h w k1 k2",
        b=b,
        c=c,
        h=h,
        w=w,
        k1=k1,
        k2=k2
    )
    return tensor


class ValidateArgsCorrelation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    x: torch.Tensor = Field(...)
    y: torch.Tensor = Field(...)
    kernel_size: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt]
    ] = (5, 5),
    dilation: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt]
    ] = (1, 1),
    stride: Union[
        PositiveInt, Tuple[PositiveInt, PositiveInt]
    ] = (1, 1),
    zero_padding: Union[
        NonNegativeInt, Tuple[NonNegativeInt, NonNegativeInt]
    ] = (0, 0)

    @classmethod
    @field_validator('x', 'y', mode="after", check_fields=True)
    def check_tensor(
            cls,
            v: Union[Float[Array, "r c"], Complex[Array, "r c"]],
            info: FieldValidationInfo
    ):
        if v.ndim != 4:
            DimensionException(
                "Unsupported tensor shape !\n" +
                "\tExpected a `4D` tensor " +
                f"but got `{v.ndim}D` tensor instead."
            )
        if not (
                torch.is_complex(input=v) or torch.is_floating_point(input=v)
        ):
            raise DTypeException(
                "Unsupported tensor dtype!\n" +
                "\tExpected a `float` or `complex` valued  tensor " +
                f"but got a tenosr of dtype {v.dtype}."
            )
        if 'x' in info.data:
            x_shape = tuple(info.data['x'].shape)
            y_shape = tuple(v.shape)
            x_device = info.data['x'].device
            y_device = v.device
            if x_shape != y_shape:
                raise ShapeException(
                    "Shape of `y` does not match shape of `x`!" +
                    f"\t'x' shape: {x_shape}\n" +
                    f"\t'y' shape: {y_shape}"
                )
            if x_device != y_device:
                raise DeviceException(
                    "Tensor `x` and tensor `y` are " +
                    "not present in the same device!\n" +
                    f"\t`x` is in: {x_device}\n" +
                    f"\t`y` is in: {y_device}"
                )
        return v

    @classmethod
    @field_validator(
        'kernel_size',
        'dilation',
        'stride',
        'zero_padding',
        mode="after",
        check_fields=True
    )
    def check_tuple(
            cls,
            v: Union[int, Tuple[int, int]]
    ):
        if isinstance(v, int):
            v = (v, v)
        return v


def patched_cross_correlation2d(
        x: Union[Float[Array, "#b #c #h #w"], Complex[Array, "#b #c #h #w"]],
        y: Union[Float[Array, "#b #c #h #w"], Complex[Array, "#b #c #h #w"]],
        kernel_size: Tuple[PositiveInt, PositiveInt] = (5, 5),
        dilation: Tuple[PositiveInt, PositiveInt] = (1, 1),
        stride: Tuple[PositiveInt, PositiveInt] = (1, 1),
        zero_padding: Tuple[NonNegativeInt, NonNegativeInt] = (0, 0)
) -> Union[Float[Array, "#b #c #hx #wx"], Complex[Array, "#b #c #hx #wx"]]:
    x = convolve2d_patches(
        tensor=x,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=stride,
        zero_padding=zero_padding
    )

    y = convolve2d_patches(
        tensor=y,
        kernel_size=kernel_size,
        dilation=dilation,
        stride=stride,
        zero_padding=zero_padding
    )

    # noinspection PyArgumentList
    cov_xy = (
        (
                (
                        x - x.mean(dim=(4, 5), keepdim=True)
                ) * (
                        y - y.mean(dim=(4, 5), keepdim=True)
                ).conj()
        ).mean(dim=(4, 5), keepdim=True)
    )

    # noinspection PyArgumentList
    std_xy = (
        x.var(
            dim=(4, 5), correction=1, keepdim=True
        ) * y.var(
            dim=(4, 5), correction=1, keepdim=True
        )
    ) ** 0.5
    valid_mask = std_xy != 0
    cor = torch.full_like(input=cov_xy, fill_value=complex('inf'))
    cor[valid_mask] = cov_xy[valid_mask] / std_xy[valid_mask]
    return cor.squeeze(dim=(4, 5))


class ValidateArgsCovariance(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    tensor: torch.Tensor
    dim: NonNegativeInt = 0

    @classmethod
    @field_validator(
        'tensor',
        mode="after",
        check_fields=True
    )
    def check_tensor(
            cls,
            v: Union[
                Float[Array, "*#d"],
                Complex[Array, "*#d"]
            ]
    ):
        if v.ndim <= 0:
            DimensionException(
                "Unsupported tensor shape!\n" +
                "\tExpected a tensor with at least 1 dimension " +
                f"instead of a `{v.ndim}D` tensor."
            )
        if not (
                torch.is_complex(input=v) or torch.is_floating_point(input=v)
        ):
            raise DTypeException(
                "Unsupported tensor dtype!\n" +
                "\tExpected a `float` or `complex` valued  tensor " +
                f"instead of having dtype {v.dtype}."
            )
        return v

    @classmethod
    @field_validator(
        'dim',
        mode="after",
        check_fields=True
    )
    def check_dim(
            cls,
            v: NonNegativeInt,
            info: FieldValidationInfo
    ):
        if 'tensor' in info.data:
            n_dim = info.data['tensor'].ndim
            if v >= n_dim:
                raise IndexError(
                    f"`dim` {v} is out of bounds " +
                    "for a tensor with dimension {n_dim}."
                )
            return v


def covariance_matrix(
        tensor: Union[
            Float[Array, "*#d"],
            Complex[Array, "*#d"]
        ],
        dim: NonNegativeInt = 0
) -> Union[
    Float[Array, "*#d"],
    Complex[Array, "*#d"]
]:
    args = ValidateArgsCovariance(
        tensor=tensor,
        dim=dim
    )
    tensor = args.tensor
    dim = args.dim
    n_dim = tensor.ndim
    x_pat = [f"d{i}" for i in range(n_dim)]
    y_pat = x_pat.copy()
    x_pat[dim] = "i"
    y_pat[dim] = "j"
    z_pat = y_pat.copy()
    z_pat.insert(dim, "i")
    x_pat = " ".join(x_pat)
    y_pat = " ".join(y_pat)
    z_pat = " ".join(z_pat)
    pattern = f"{x_pat}, {y_pat} -> {z_pat}"
    return einsum(
        tensor, tensor.conj(), pattern
    )


class ValidateArgsMaskedMean(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    tensor: torch.Tensor = Field(...)
    dim: Optional[Union[int, Sequence[int]]] = None
    mask: Optional[torch.Tensor] = None
    keepdim: Optional[bool] = False

    @classmethod
    @field_validator('tensor', mode="after", check_fields=True)
    def check_tensor(cls, v: Union[Float[Array, "*_"], Complex[Array, "*_"]]):
        if not (
                torch.is_complex(input=v) or torch.is_floating_point(input=v)
        ):
            raise DTypeException(
                "Unsupported tensor dtype!\n" +
                "\tExpected a `float` or `complex` valued  tensor " +
                f"instead of dtype {v.dtype}."
            )
        return v

    @classmethod
    @field_validator('dim', mode="after", check_fields=True)
    def check_dim(
            cls,
            v: Union[int, Sequence[int]],
            info: FieldValidationInfo
    ):
        if 'tensor' in info.data:
            n_dim = info.data["tensor"].ndim
            if v is None:
                return tuple(range(n_dim))
            elif isinstance(v, int):
                return (v,)
            else:
                dims = list()
                for d in tuple(set(v)):
                    if d < 0:
                        d += n_dim
                    if 0 <= d < n_dim:
                        dims.append(d)
                    else:
                        raise DimensionException(
                            "One or more dimension out of range!\n" +
                            "Expected all dimensions to be in " +
                            f"range of [{-n_dim}, {n_dim - 1}], but got {v}."
                        )
                return tuple(dims)

    @classmethod
    @field_validator('mask', mode="after", check_fields=True)
    def check_mask(
            cls,
            v: Bool[Array, "*_"],
            info: FieldValidationInfo
    ):
        if ('tensor' in info.data) and ('dim' in info.data):
            tensor_shape = tuple(info.data["tensor"].shape)
            tensor_device = info.data["tensor"].device
            mask_shape = tuple(tensor_shape[i] for i in info.data["dim"])
            if v is None:
                return torch.full(
                    size=mask_shape,
                    fill_value=False,
                    dtype=torch.bool,
                    device=tensor_device,
                    requires_grad=False
                )
            if v.shape != mask_shape:
                raise ShapeException(
                    "Incompatible `mask` shape!\n" +
                    f"\tExpected shape: {mask_shape} got {v.shape}"
                )
            if v.dtype != torch.bool:
                raise DTypeException(
                    "Invalid `mask` dtype!\n" +
                    "\tExpected a `torch.bool` tensor " +
                    f"instead of having dtype {v.dtype}."
                )
            if v.device != tensor_device:
                raise DeviceException(
                    "`tensor` and `mask` are " +
                    "not present in the same device!\n" +
                    f"\t`tensor` is in: {tensor_device}" +
                    f"\t`mask` is in: {v.device}"
                )
            return v


def masked_mean(
        tensor: Union[Float[Array, "*_"], Complex[Array, "*_"]],
        mask: Optional[Bool[Array, "*_"]] = None,
        dim: Optional[Union[int, Sequence[int]]] = None,
        keepdim: Optional[bool] = False
) -> Union[Float[Array, "*_"], Complex[Array, "*_"]]:
    args = ValidateArgsMaskedMean(
        tensor=tensor,
        dim=dim,
        mask=mask,
        keepdim=keepdim
    )
    tensor = args.tensor
    dim = args.dim
    mask = args.mask
    keepdim = args.keepdim
    tensor_shape = tuple(tensor.shape)
    indexes = list(range(len(tensor_shape)))
    tokens = list()
    target_shape = list()
    r_tokens = list()
    c_tokens = list()
    for i in indexes:
        d = f"d{i}"
        tokens.append(d)
        if i in dim:
            r_tokens.append(d)
            if keepdim:
                target_shape.append(1)
        else:
            c_tokens.append(d)
            target_shape.append(tensor_shape[i])
    if len(target_shape) == 0:
        target_shape = [-1]
    t_pat = ' '.join(tokens)
    r_pat = ' '.join(r_tokens)
    c_pat = ' '.join(c_tokens)
    pattern = f"{t_pat} -> ({r_pat}) ({c_pat})"
    tensor = rearrange(tensor=tensor, pattern=pattern)
    return tensor[mask.ravel()].mean(
        dim=0,
        keepdim=False
    ).reshape(*target_shape)


class ValidateArgsMaskedInverse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra='forbid')

    tensor: torch.Tensor = Field(...)
    batch_mask: Optional[torch.Tensor] = None
    atol: Optional[PositiveFloat] = 1e-08
    fill_singular: Union[
        Literal['identity'],
        int,
        float,
        complex
    ] = complex('nan')
    driver: Optional[Literal["gesvd", "gesvdj", "gesvda"]] = None

    @classmethod
    @field_validator('tensor', mode="after", check_fields=True)
    def check_tensor(
            cls,
            v: Union[Float[Array, "#b #h #w"], Complex[Array, "#b #h #w"]]
    ):
        if v.ndim != 3:
            DimensionException(
                "Unsupported tensor shape !\n" +
                "\tExpected a `3D` tensor " +
                f"but got `{v.ndim}D` tensor instead."
            )
        if not (
                torch.is_complex(input=v) or torch.is_floating_point(input=v)
        ):
            raise DTypeException(
                "Unsupported tensor dtype!\n" +
                "\tExpected a `float` or `complex` valued tensor " +
                f"but got a tenosr of dtype {v.dtype}."
            )
        return v

    @classmethod
    @field_validator('atol', mode="after", check_fields=True)
    def check_atol(cls, v: PositiveFloat):
        if v >= 1:
            raise ValueError(
                "Invalid absolute precision tolerance.\n" +
                f"\tExpected a value in the range (0, 1) but got {v}"
            )
        return v

    @classmethod
    @field_validator('batch_mask', mode="after", check_fields=True)
    def check_mask(
            cls,
            v: Optional[Bool[Array, "#b"]],
            info: FieldValidationInfo
    ):
        if not (
                torch.is_complex(input=v) or torch.is_floating_point(input=v)
        ):
            raise DTypeException(
                "Unsupported tensor dtype!\n" +
                "\tExpected a `boolean` tensor " +
                f"but got a tenosr of dtype {v.dtype}."
            )
        if 'tensor' in info.data:
            n_batch = info.data['tensor'].size(0)

            # noinspection PyUnresolvedReferences
            n_mask = v.numel()
            mask_device = v.device
            tensor_device = info.data['tensor'].device
            if (v.ndim != 1) or (n_mask != n_batch):
                DimensionException(
                    "Unsupported tensor shape!\n" +
                    f"\tExpected a `1D` boolean tensor " +
                    f"with {n_batch} elements but got `{v.ndim}D` " +
                    f"tensor with {n_mask} elements instead."
                )
            if mask_device != tensor_device:
                raise DeviceException(
                    "`tensor` and `mask` are " +
                    "not present in the same device!\n" +
                    f"\t`tensor` is in: {tensor_device}" +
                    f"\t`mask` is in: {mask_device}"
                )
            return v

    @classmethod
    @field_validator('fill_singular', mode="after", check_fields=True)
    def check_fill(
            cls,
            v: Optional[PositiveFloat],
            info: FieldValidationInfo
    ):
        if 'tensor' in info.data:
            n, m = tuple(info.data['tensor'].shape)[1:]
            tensor_dtype = info.data['tensor'].dtype
            tensor_device = info.data['tensor'].device
            tensor_layout = info.data['tensor'].layout
            if isinstance(v, str) and v == "identity":
                return torch.eye(
                    n=n,
                    m=m,
                    out=None,
                    dtype=tensor_dtype,
                    layout=tensor_layout,
                    device=tensor_device
                )
            else:
                return torch.full(
                    size=(n, m),
                    fill_value=v,
                    out=None,
                    dtype=tensor_dtype,
                    layout=tensor_layout,
                    device=tensor_device
                )

    @classmethod
    @field_validator('driver', mode="after", check_fields=True)
    def check_fill(
            cls,
            v: Optional[Literal["gesvd", "gesvdj", "gesvda"]],
            info: FieldValidationInfo
    ):
        if 'tensor' in info.data:
            tensor_device = info.data['tensor'].device
            if (
                    v in {"gesvd", "gesvdj", "gesvda"}
            ) and (
                    tensor_device.type != "cuda"
            ):
                raise AssertionError(
                    f"Driver {v} is incompatible for the tensor \n" +
                    f"residining in the device {tensor_device}."
                )


def masked_batch_inverse(
        tensor: Union[
            Float[Array, "#b #h #w"],
            Complex[Array, "#b #h #w"]
        ],
        batch_mask: Optional[Bool[Array, "#b"]] = None,
        atol: PositiveFloat = 1e-08,
        fill_singular: Union[
            Literal['identity'],
            float,
            complex
        ] = complex('nan'),
        driver: Optional[Literal["gesvd", "gesvdj", "gesvda"]] = None
) -> Tuple[
    Union[
        Float[Array, "#b #h #w"],
        Complex[Array, "#b #h #w"]
    ],
    Bool[Array, "#b"]
]:
    args = ValidateArgsMaskedInverse(
        tensor=tensor,
        batch_mask=batch_mask,
        atol=atol,
        fill_singular=fill_singular,
        driver=driver
    )
    tensor = args.tensor
    batch_mask = args.batch_mask
    atol = args.atol
    fill_singular = args.fill_singular
    driver = args.driver

    valid = torch.logical_not(input=batch_mask)
    tensor = tensor[valid]
    u, s, v = torch.linalg.svd(
        A=tensor,
        full_matrices=True,
        driver=driver
    )
    singularity_mask = torch.any(
        input=(s > (1 / atol)),
        dim=1
    )
    batch_mask[valid] = singularity_mask
    invertible = torch.logical_not(input=singularity_mask)
    u = u[invertible]
    s = s[invertible]
    v = v[invertible]
    batch_inv_valid = v @ (torch.diag(1 / s) @ u.permute(1, 0))
    valid = torch.logical_not(input=batch_mask)
    batch_inv = torch.zeros_like(tensor)
    batch_inv[valid] = batch_inv_valid
    batch_inv[batch_mask] = fill_singular
    return batch_inv, batch_mask
