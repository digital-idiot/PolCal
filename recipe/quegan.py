import torch
from einops import rearrange
from typing import Optional, Union
from pydantic.types import PositiveFloat
from jaxtyping import Complex, Bool, Array


# noinspection PyUnresolvedReferences,PyArgumentList
def estimate_calibration_parameters(
        cov_tensor: Complex[Array, "*# #k1=4 #k2=4"],
        mask: Bool[Array, "*#"],
        rtol: Optional[PositiveFloat] = 1e-5,
        atol: Optional[PositiveFloat] = 1e-8,
        m: Optional[float] = 1.0,
        a_fill: Optional[Union[float, complex]] = 1.0,
        u_fill: Optional[Union[float, complex]] = 0.0,
        v_fill: Optional[Union[float, complex]] = 0.0,
        w_fill: Optional[Union[float, complex]] = 0.0,
        z_fill: Optional[Union[float, complex]] = 0.0
) -> Complex[Array, "*# #k=5"]:
    # TODO: Add pydantic check
    shape_original = tuple(cov_tensor.shape)
    cov_tensor = rearrange(
        tensor=cov_tensor, pattern="... k1 k2 -> (...) (k1 k2)"
    )
    shape_infered = tuple(cov_tensor.shape)
    mask = rearrange(tensor=mask, pattern="... -> (...)")
    valid = torch.logical_not(input=mask)

    cov_tensor = cov_tensor[valid]

    delta = (
        cov_tensor[:, 0] * cov_tensor[:, 15]
    ) - (
        cov_tensor[:, 3] * cov_tensor[:, 3].conj()
    )
    delta_mask = torch.isclose(
        input=delta,
        other=torch.zeros_like(delta),
        rtol=rtol,
        atol=atol,
        equal_nan=False
    )
    delta_valid = torch.logical_not(input=delta_mask)
    delta = delta[delta_valid]
    mask[valid] = delta_mask
    valid = torch.logical_not(input=mask)

    cov_tensor = cov_tensor[delta_valid]
    u = (
        (
            cov_tensor[:, 15] * cov_tensor[:, 4]
        ) - (
            cov_tensor[:, 12] * cov_tensor[:, 7]
        )
    ) / delta
    v = (
        (
            cov_tensor[:, 0] * cov_tensor[:, 7]
        ) - (
            cov_tensor[:, 4] * cov_tensor[:, 3]
        )
    ) / delta
    w = (
        (
            cov_tensor[:, 0] * cov_tensor[:, 11]
        ) - (
            cov_tensor[:, 8] * cov_tensor[:, 3]
        )
    ) / delta
    z = (
        (
            cov_tensor[:, 15] * cov_tensor[:, 8]
        ) - (
            cov_tensor[:, 12] * cov_tensor[:, 11]
        )
    ) / delta

    chi = cov_tensor[9] - (z * cov_tensor[1]) - (w * cov_tensor[11])
    chi_mask = torch.isclose(
        input=chi,
        other=torch.zeros_like(chi),
        rtol=rtol,
        atol=atol,
        equal_nan=False
    )
    chi_valid = torch.logical_not(input=chi_mask)
    chi = chi[chi_valid]
    u = u[chi_valid]
    v = v[chi_valid]
    w = w[chi_valid]
    z = z[chi_valid]
    mask[valid] = chi_mask
    valid = torch.logical_not(input=mask)

    cov_tensor = cov_tensor[chi_valid]
    alpha_a = (
        cov_tensor[:, 5] - (u * cov_tensor[:, 1]) - (v * cov_tensor[:, 13])
    ) / chi
    zeta = cov_tensor[:, 10] - (
        z.conj() * cov_tensor[:, 8]
    ) - (
        w.conj() * cov_tensor[:, 11]
    )
    zeta_mask = torch.isclose(
        input=zeta,
        other=torch.zeros_like(chi),
        rtol=rtol,
        atol=atol,
        equal_nan=False
    )
    zeta_valid = torch.logical_not(input=zeta_mask)
    zeta = zeta[zeta_valid]

    chi = chi[zeta_valid]
    mask[valid] = zeta_mask
    valid = torch.logical_not(input=mask)

    alpha_b = chi.conj() / zeta
    ab = (alpha_a * alpha_b).abs() - m
    alpha = (
        ab + (
            (
                (
                    ab ** 2
                ) + (
                    4 * alpha_b * alpha_b.conj()
                )
            ) ** 0.5
        )
    ) / (
        2 * alpha_b.abs()
    )
    ux = torch.full(
        size=shape_infered[:-1],
        fill_value=u_fill,
        dtype=u.dtype,
        device=u.device,
        layout=u.layout,
        requires_grad=u.requires_grad
    )
    vx = torch.full(
        size=shape_infered[:-1],
        fill_value=v_fill,
        dtype=v.dtype,
        device=v.device,
        layout=v.layout,
        requires_grad=v.requires_grad
    )
    wx = torch.full(
        size=shape_infered[:-1],
        fill_value=w_fill,
        dtype=w.dtype,
        device=w.device,
        layout=w.layout,
        requires_grad=w.requires_grad
    )
    zx = torch.full(
        size=shape_infered[:-1],
        fill_value=z_fill,
        dtype=z.dtype,
        device=z.device,
        layout=z.layout,
        requires_grad=z.requires_grad
    )
    ax = torch.full(
        size=shape_infered[:-1],
        fill_value=a_fill,
        dtype=alpha.dtype,
        device=alpha.device,
        layout=alpha.layout,
        requires_grad=alpha.requires_grad
    )
    ux[valid] = u
    vx[valid] = v
    wx[valid] = w
    zx[valid] = z
    ax[valid] = alpha
    ux = ux.reshape(shape=shape_original[:-2])
    vx = vx.reshape(shape=shape_original[:-2])
    wx = wx.reshape(shape=shape_original[:-2])
    zx = zx.reshape(shape=shape_original[:-2])
    ax = ax.reshape(shape=shape_original[:-2])
    mask = mask.reshape(shape=shape_original[:-2])
    cal_params = torch.stack(tensors=[ux, vx, wx, zx, ax], dim=-1)
    return cal_params, mask
