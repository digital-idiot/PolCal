import gdal
import numpy as np


def estimate_mne(mne_img, u_img, v_img, w_img, z_img, alpha_img=None):
    u_f = gdal.Open(u_img)
    v_f = gdal.Open(v_img)
    w_f = gdal.Open(w_img)
    z_f = gdal.Open(z_img)

    u_b = u_f.GetRasterBand(1)
    v_b = v_f.GetRasterBand(1)
    w_b = w_f.GetRasterBand(1)
    z_b = z_f.GetRasterBand(1)

    if alpha_img is not None:
        alpha_f = gdal.Open(alpha_img)
        alpha_b = alpha_f.GetRasterBand(1)
    else:
        alpha_b = None

    xoffset = u_b.XSize
    yoffset = u_b.YSize

    driver = gdal.GetDriverByName("ENVI")
    mne_f = driver.Create(mne_img, xoffset, yoffset, 1, gdal.GDT_Float32)
    mne_b = mne_f.GetRasterBand(1)

    for i in range(xoffset):
        for j in range(yoffset):

            u = u_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            v = v_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            w = w_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            z = z_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]

            if alpha_b is not None:
                alpha = alpha_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            else:
                alpha = 1

            if not (np.isnan(u) or np.isnan(v) or np.isnan(w) or np.isnan(z) or np.isnan(alpha)):

                if (u != 0) and (v != 0) and (w != 0) and (z != 0) and (alpha != 0):

                    denom = (((u * w) - 1) * ((v * z) - 1))

                    if alpha != 0 and denom != 0:

                        ra = alpha ** 0.5
                        ra_inv = 1 / ra
                        d_mat = np.array(
                            [
                                [
                                    (1+0j),
                                    (-w * ra),
                                    (-v * ra_inv),
                                    (v * w)
                                ],
                                [
                                    u,
                                    ra,
                                    (u * v * ra_inv),
                                    v
                                ],
                                [
                                    z,
                                    (w * z * ra),
                                    ra_inv,
                                    w
                                ],
                                [
                                    (u * z),
                                    (z * ra),
                                    (u * ra_inv),
                                    (1 + 0j)
                                ]
                            ]
                        )

                        identity4 = np.identity(4, dtype=np.complex)
                        delta_d = d_mat - identity4
                        delta_d_dagger = np.transpose(np.conjugate(delta_d))
                        noise_info = np.matmul(delta_d_dagger, delta_d)
                        try:
                            eig_vals = np.linalg.eigvals(noise_info)
                            mne = np.array([[(np.abs(np.max(eig_vals))) ** 0.5]])
                        except np.linalg.LinAlgError:
                            mne = np.array([[np.nan]])
                    else:
                        mne = np.nan

                    mne_b.WriteArray(mne, xoff=i, yoff=j)

            print(i, j)

    mne_b.FlushCache()
