import gdal
import gdalconst
import numpy as np


def quegan(cov_dir, crosstalk_dir, logic_img, m=1.0, x_start=0, y_start=0, xoffset=None, yoffset=None):
    c11_f = cov_dir + "C11"
    c12_f = cov_dir + "C12"
    # c13_f = cov_dir + "C13"
    c14_f = cov_dir + "C14"
    c21_f = cov_dir + "C21"
    c22_f = cov_dir + "C22"
    # c23_f = cov_dir + "C23"
    c24_f = cov_dir + "C24"
    c31_f = cov_dir + "C31"
    c32_f = cov_dir + "C32"
    c33_f = cov_dir + "C33"
    c34_f = cov_dir + "C34"
    c41_f = cov_dir + "C41"
    c42_f = cov_dir + "C42"
    # c43_f = cov_dir + "C43"
    c44_f = cov_dir + "C44"

    u_file = crosstalk_dir + "u"
    v_file = crosstalk_dir + "v"
    w_file = crosstalk_dir + "w"
    z_file = crosstalk_dir + "z"
    alpha_file = crosstalk_dir + "alpha"

    c11_f = gdal.Open(c11_f)
    c12_f = gdal.Open(c12_f)
    # c13_f = gdal.Open(c13_f)
    c14_f = gdal.Open(c14_f)
    c21_f = gdal.Open(c21_f)
    c22_f = gdal.Open(c22_f)
    # c23_f = gdal.Open(c23_f)
    c24_f = gdal.Open(c24_f)
    c31_f = gdal.Open(c31_f)
    c32_f = gdal.Open(c32_f)
    c33_f = gdal.Open(c33_f)
    c34_f = gdal.Open(c34_f)
    c41_f = gdal.Open(c41_f)
    c42_f = gdal.Open(c42_f)
    # c43_f = gdal.Open(c43_f)
    c44_f = gdal.Open(c44_f)

    c11b = c11_f.GetRasterBand(1)
    c12b = c12_f.GetRasterBand(1)
    # c13b = c13_f.GetRasterBand(1)
    c14b = c14_f.GetRasterBand(1)
    c21b = c21_f.GetRasterBand(1)
    c22b = c22_f.GetRasterBand(1)
    # c23b = c23_f.GetRasterBand(1)
    c24b = c24_f.GetRasterBand(1)
    c31b = c31_f.GetRasterBand(1)
    c32b = c32_f.GetRasterBand(1)
    c33b = c33_f.GetRasterBand(1)
    c34b = c34_f.GetRasterBand(1)
    c41b = c41_f.GetRasterBand(1)
    c42b = c42_f.GetRasterBand(1)
    # c43b = c43_f.GetRasterBand(1)
    c44b = c44_f.GetRasterBand(1)

    logic_f = gdal.Open(logic_img)
    logic_board = logic_f.GetRasterBand(1)

    if xoffset is None:
        xoffset = c11b.XSize

    if yoffset is None:
        yoffset = c11b.YSize

    driver = gdal.GetDriverByName("ENVI")
    u_f = driver.Create(u_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    u_band = u_f.GetRasterBand(1)

    v_f = driver.Create(v_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    v_band = v_f.GetRasterBand(1)

    w_f = driver.Create(w_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    w_band = w_f.GetRasterBand(1)

    z_f = driver.Create(z_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    z_band = z_f.GetRasterBand(1)

    alpha_f = driver.Create(alpha_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    alpha_band = alpha_f.GetRasterBand(1)

    for i in range(xoffset):
        for j in range(yoffset):
            r = x_start + i
            c = y_start + j
            flag = logic_board.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            if flag == 255:
                c11 = c11b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                c14 = c14b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                c21 = c21b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                c24 = c24b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                c31 = c31b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                c34 = c34b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                c41 = c41b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                c44 = c44b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]

                c32 = c32b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                c12 = c12b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                c42 = c42b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                c22 = c22b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                c33 = c33b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]

                delta = ((c11 * c44) - (np.abs(c14) ** 2))

                if delta != 0:
                    u = ((c44 * c21) - (c41 * c24)) / delta
                    v = ((c11 * c24) - (c21 * c14)) / delta
                    w = ((c11 * c34) - (c31 * c14)) / delta
                    z = ((c44 * c31) - (c41 * c34)) / delta

                    x = c32 - (z * c12) - (w * c42)
                    alpha_1 = (c22 - (u * c12) - (v * c42)) / x
                    alpha_2 = np.conj(x) / (c33 - (np.conj(z) * c31) - (np.conj(w) * c34))

                    t1 = np.abs(alpha_1 * alpha_2) - m
                    t2 = (t1 ** 2) + (4 * m * (np.abs(alpha_2) ** 2))
                    t3 = 2 * np.abs(alpha_2)
                    t4 = t1 + (t2 ** 0.5)
                    alpha = (t4 / t3) * (alpha_1 / np.abs(alpha_1))

                    u = np.array([[u]])
                    v = np.array([[v]])
                    w = np.array([[w]])
                    z = np.array([[z]])
                    alpha = np.array([[alpha]])

                else:
                    u = np.array([[np.nan]])
                    v = np.array([[np.nan]])
                    w = np.array([[np.nan]])
                    z = np.array([[np.nan]])
                    alpha = np.array([[np.nan]])
            else:
                u = np.array([[np.nan]])
                v = np.array([[np.nan]])
                w = np.array([[np.nan]])
                z = np.array([[np.nan]])
                alpha = np.array([[np.nan]])

            u_band.WriteArray(u, xoff=r, yoff=c)
            v_band.WriteArray(v, xoff=r, yoff=c)
            w_band.WriteArray(w, xoff=r, yoff=c)
            z_band.WriteArray(z, xoff=r, yoff=c)
            alpha_band.WriteArray(alpha, xoff=r, yoff=c)

            print(i, j)

    u_band.FlushCache()
    v_band.FlushCache()
    w_band.FlushCache()
    z_band.FlushCache()
    alpha_band.FlushCache()

    return 0


def ainsworth(
        cov_dir,
        crosstalk_dir,
        max_iter=16,
        epsilon=1e-8,
        x_start=0,
        y_start=0,
        xoffset=None,
        yoffset=None
):
    c11_f = cov_dir + "C11"
    c12_f = cov_dir + "C12"
    c13_f = cov_dir + "C13"
    c14_f = cov_dir + "C14"
    c21_f = cov_dir + "C21"
    c22_f = cov_dir + "C22"
    c23_f = cov_dir + "C23"
    c24_f = cov_dir + "C24"
    c31_f = cov_dir + "C31"
    c32_f = cov_dir + "C32"
    c33_f = cov_dir + "C33"
    c34_f = cov_dir + "C34"
    c41_f = cov_dir + "C41"
    c42_f = cov_dir + "C42"
    c43_f = cov_dir + "C43"
    c44_f = cov_dir + "C44"

    u_file = crosstalk_dir + "u"
    v_file = crosstalk_dir + "v"
    w_file = crosstalk_dir + "w"
    z_file = crosstalk_dir + "z"
    alpha_file = crosstalk_dir + "alpha"

    c11_f = gdal.Open(c11_f)
    c12_f = gdal.Open(c12_f)
    c13_f = gdal.Open(c13_f)
    c14_f = gdal.Open(c14_f)
    c21_f = gdal.Open(c21_f)
    c22_f = gdal.Open(c22_f)
    c23_f = gdal.Open(c23_f)
    c24_f = gdal.Open(c24_f)
    c31_f = gdal.Open(c31_f)
    c32_f = gdal.Open(c32_f)
    c33_f = gdal.Open(c33_f)
    c34_f = gdal.Open(c34_f)
    c41_f = gdal.Open(c41_f)
    c42_f = gdal.Open(c42_f)
    c43_f = gdal.Open(c43_f)
    c44_f = gdal.Open(c44_f)

    c11b = c11_f.GetRasterBand(1)
    c12b = c12_f.GetRasterBand(1)
    c13b = c13_f.GetRasterBand(1)
    c14b = c14_f.GetRasterBand(1)
    c21b = c21_f.GetRasterBand(1)
    c22b = c22_f.GetRasterBand(1)
    c23b = c23_f.GetRasterBand(1)
    c24b = c24_f.GetRasterBand(1)
    c31b = c31_f.GetRasterBand(1)
    c32b = c32_f.GetRasterBand(1)
    c33b = c33_f.GetRasterBand(1)
    c34b = c34_f.GetRasterBand(1)
    c41b = c41_f.GetRasterBand(1)
    c42b = c42_f.GetRasterBand(1)
    c43b = c43_f.GetRasterBand(1)
    c44b = c44_f.GetRasterBand(1)

    if xoffset is None:
        xoffset = c11b.XSize

    if yoffset is None:
        yoffset = c11b.YSize

    driver = gdal.GetDriverByName("ENVI")
    u_f = driver.Create(u_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    u_band = u_f.GetRasterBand(1)

    v_f = driver.Create(v_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    v_band = v_f.GetRasterBand(1)

    w_f = driver.Create(w_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    w_band = w_f.GetRasterBand(1)

    z_f = driver.Create(z_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    z_band = z_f.GetRasterBand(1)

    alpha_f = driver.Create(alpha_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    alpha_band = alpha_f.GetRasterBand(1)

    for i in range(xoffset):
        for j in range(yoffset):
            r = x_start + i
            c = y_start + j

            c11 = c11b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c12 = c12b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c13 = c13b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c14 = c14b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c21 = c21b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c22 = c22b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c23 = c23b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c24 = c24b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c31 = c31b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c32 = c32b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c33 = c33b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c34 = c34b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c41 = c41b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c42 = c42b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c43 = c43b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c44 = c44b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]

            c_cap = np.array(
                [
                    [c11, c12, c13, c14],
                    [c21, c22, c23, c24],
                    [c31, c32, c33, c34],
                    [c41, c42, c43, c44]
                ]
            )

            t1 = c23 / np.abs(c23)
            t2 = (np.abs(c22 / c33)) ** 0.5
            alpha = t1 * t2

            gamma = np.inf

            iter_count = 0

            u = v = w = z = 0j

            while gamma > epsilon and iter_count < max_iter and alpha > epsilon:

                beta1 = c_cap[1][1]
                beta2 = c_cap[2][2]

                eta1 = beta1 - c_cap[2][1]
                eta2 = beta2 - c_cap[1][2]

                indicator1 = np.inf
                indicator2 = np.inf

                if beta1 != 0j:
                    indicator1 = np.abs(eta1 / beta1)

                if beta2 != 0j:
                    indicator2 = np.abs(eta2 / beta2)

                if (indicator1 <= 1) or (indicator2 <= 1):

                    ralpha = alpha ** 0.5
                    ralpha_inv = 1 / ralpha
                    sigma = (1 / (((u * w) - 1) * ((v * z) - 1))) * np.array(
                        [
                            [
                                (1 + 0j),
                                -w,
                                -v,
                                v * w
                            ],
                            [
                                -u * ralpha_inv,
                                ralpha_inv,
                                u * v * ralpha_inv,
                                -v * ralpha_inv
                            ],
                            [
                                -z * ralpha,
                                w * z * ralpha,
                                ralpha,
                                -w * ralpha
                            ],
                            [
                                u * z,
                                -z,
                                -u,
                                (1 + 0j)
                            ]
                        ]
                    )

                    sigma_dagger = np.transpose(np.conj(sigma))
                    c_iter = np.matmul(sigma, np.matmul(c_cap, sigma_dagger))
                    c11_i = c_iter[0][0]
                    # c12_i = c_iter[0][1]
                    # c13_i = c_cap[0][2]
                    c14_i = c_iter[0][3]
                    c21_i = c_iter[1][0]
                    c22_i = c_iter[1][1]
                    c23_i = c_iter[1][2]
                    c24_i = c_iter[1][3]
                    c31_i = c_iter[2][0]
                    c32_i = c_iter[2][1]
                    c33_i = c_iter[2][2]
                    c34_i = c_iter[2][3]
                    c41_i = c_iter[3][0]
                    # c42_i = c_iter[3][1]
                    # c43_i = c_iter[3][2]
                    c44_i = c_iter[3][3]

                    a_cap = 0.5 * (c31_i + c21_i)
                    b_cap = 0.5 * (c34_i + c24_i)

                    if c23_i != 0j and c33_i != 0j:
                        t1_i = c23_i / np.abs(c23_i)
                        t2_i = (np.abs(c22_i / c33_i)) ** 0.5
                        alpha_i = t1_i * t2_i

                        zeta = np.array(
                            [
                                [0, 0, c41_i, c11_i],
                                [c11_i, c41_i, 0, 0],
                                [0, 0, c44_i, c14_i],
                                [c14_i, c44_i, 0, 0]
                            ]
                        )

                        tau = np.array(
                            [
                                [0, c33_i, c32_i, 0],
                                [0, c23_i, c22_i, 0],
                                [c33_i, 0, 0, c32_i],
                                [c23_i, 0, 0, c22_i]
                            ]
                        )

                        chi = np.array(
                            [
                                [c31_i - a_cap],
                                [c21_i - a_cap],
                                [c34_i - b_cap],
                                [c24_i - b_cap]
                            ]
                        )

                        rchi = chi.real
                        ichi = chi.imag

                        zpt = zeta + tau
                        zmt = zeta - tau

                        rzpt = zpt.real
                        izpt = zpt.imag

                        rzmt = zmt.real
                        izmt = zmt.imag

                        try:
                            rzmt_inv = np.linalg.inv(rzmt)
                            izmt_inv = np.linalg.inv(izmt)

                            t3 = np.linalg.inv(np.matmul(izmt_inv, rzpt) + np.matmul(rzmt_inv, izpt))
                        except np.linalg.linalg.LinAlgError as inv_err:
                            u = v = w = z = alpha = np.nan
                            print(inv_err)
                            break
                        t4 = np.matmul(izmt_inv, rchi) + np.matmul(rzmt_inv, ichi)

                        rdelta = np.matmul(t3, t4)
                        idelta = np.matmul(rzmt_inv, (ichi - np.matmul(izpt, rdelta)))
                        delta = rdelta + (1j * idelta)

                        u_i = delta[0][0]
                        v_i = delta[1][0]
                        w_i = delta[2][0]
                        z_i = delta[3][0]

                        u += u_i * ralpha_inv
                        v += v_i * ralpha_inv
                        w += w_i * ralpha
                        z += z_i * ralpha

                        alpha *= alpha_i

                        d_params = np.array([u_i, v_i, w_i, z_i])
                        gamma = np.max(np.abs(d_params))

                    else:
                        u = v = w = z = alpha = np.nan
                        break

                else:
                    u = v = w = z = alpha = np.nan
                    break

                iter_count += 1

            u = np.array([[u]])
            v = np.array([[v]])
            w = np.array([[w]])
            z = np.array([[z]])
            alpha = np.array([[alpha]])

            u_band.WriteArray(u, xoff=r, yoff=c)
            v_band.WriteArray(v, xoff=r, yoff=c)
            w_band.WriteArray(w, xoff=r, yoff=c)
            z_band.WriteArray(z, xoff=r, yoff=c)
            alpha_band.WriteArray(alpha, xoff=r, yoff=c)

            print(i, j)

    u_band.FlushCache()
    v_band.FlushCache()
    w_band.FlushCache()
    z_band.FlushCache()

    return 0


def ainsworth_orig(
        cov_dir,
        crosstalk_dir,
        data_quality_img,
        max_iter=15,
        epsilon=1e-8,
        x_start=0,
        y_start=0,
        xoffset=None,
        yoffset=None
):
    c11_f = cov_dir + "C11"
    c12_f = cov_dir + "C12"
    c13_f = cov_dir + "C13"
    c14_f = cov_dir + "C14"
    c21_f = cov_dir + "C21"
    c22_f = cov_dir + "C22"
    c23_f = cov_dir + "C23"
    c24_f = cov_dir + "C24"
    c31_f = cov_dir + "C31"
    c32_f = cov_dir + "C32"
    c33_f = cov_dir + "C33"
    c34_f = cov_dir + "C34"
    c41_f = cov_dir + "C41"
    c42_f = cov_dir + "C42"
    c43_f = cov_dir + "C43"
    c44_f = cov_dir + "C44"

    u_file = crosstalk_dir + "u"
    v_file = crosstalk_dir + "v"
    w_file = crosstalk_dir + "w"
    z_file = crosstalk_dir + "z"
    alpha_file = crosstalk_dir + "alpha"

    c11_f = gdal.Open(c11_f)
    c12_f = gdal.Open(c12_f)
    c13_f = gdal.Open(c13_f)
    c14_f = gdal.Open(c14_f)
    c21_f = gdal.Open(c21_f)
    c22_f = gdal.Open(c22_f)
    c23_f = gdal.Open(c23_f)
    c24_f = gdal.Open(c24_f)
    c31_f = gdal.Open(c31_f)
    c32_f = gdal.Open(c32_f)
    c33_f = gdal.Open(c33_f)
    c34_f = gdal.Open(c34_f)
    c41_f = gdal.Open(c41_f)
    c42_f = gdal.Open(c42_f)
    c43_f = gdal.Open(c43_f)
    c44_f = gdal.Open(c44_f)

    c11b = c11_f.GetRasterBand(1)
    c12b = c12_f.GetRasterBand(1)
    c13b = c13_f.GetRasterBand(1)
    c14b = c14_f.GetRasterBand(1)
    c21b = c21_f.GetRasterBand(1)
    c22b = c22_f.GetRasterBand(1)
    c23b = c23_f.GetRasterBand(1)
    c24b = c24_f.GetRasterBand(1)
    c31b = c31_f.GetRasterBand(1)
    c32b = c32_f.GetRasterBand(1)
    c33b = c33_f.GetRasterBand(1)
    c34b = c34_f.GetRasterBand(1)
    c41b = c41_f.GetRasterBand(1)
    c42b = c42_f.GetRasterBand(1)
    c43b = c43_f.GetRasterBand(1)
    c44b = c44_f.GetRasterBand(1)

    if xoffset is None:
        xoffset = c11b.XSize

    if yoffset is None:
        yoffset = c11b.YSize

    driver = gdal.GetDriverByName("ENVI")
    u_f = driver.Create(u_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    u_band = u_f.GetRasterBand(1)

    v_f = driver.Create(v_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    v_band = v_f.GetRasterBand(1)

    w_f = driver.Create(w_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    w_band = w_f.GetRasterBand(1)

    z_f = driver.Create(z_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    z_band = z_f.GetRasterBand(1)

    alpha_f = driver.Create(alpha_file, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    alpha_band = alpha_f.GetRasterBand(1)

    data_quality_f = driver.Create(data_quality_img, xoffset, yoffset, 2, gdalconst.GDT_Float32)
    dq1 = data_quality_f.GetRasterBand(1)
    dq2 = data_quality_f.GetRasterBand(2)

    for i in range(xoffset):
        for j in range(yoffset):
            r = x_start + i
            c = y_start + j

            c11 = c11b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c12 = c12b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c13 = c13b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c14 = c14b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c21 = c21b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c22 = c22b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c23 = c23b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c24 = c24b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c31 = c31b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c32 = c32b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c33 = c33b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c34 = c34b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c41 = c41b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c42 = c42b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c43 = c43b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
            c44 = c44b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]

            c_cap = np.array(
                [
                    [c11, c13, c12, c14],
                    [c31, c33, c32, c34],
                    [c21, c23, c22, c24],
                    [c41, c43, c42, c44]
                ]
            )

            tmp1 = ((np.abs(c_cap[2][2] / c_cap[1][1])) ** 0.25)
            tmp2 = (np.exp(1j * (0.5 * np.arctan(c_cap[2][1].imag / c_cap[2][1].real))))

            alpha = tmp1 * tmp2
            k = 1 + 0j

            gamma = np.inf

            iter_count = 0

            u = v = w = z = 0j

            indicator1 = None
            indicator2 = None

            while (gamma > epsilon) and (iter_count < max_iter):

                beta1 = c_cap[1][1]
                beta2 = c_cap[2][2]

                eta1 = beta1 - c_cap[2][1]
                eta2 = beta2 - c_cap[1][2]

                indicator1 = np.inf
                indicator2 = np.inf

                if beta1 != 0j:
                    indicator1 = np.abs(eta1 / beta1)

                if beta2 != 0j:
                    indicator2 = np.abs(eta2 / beta2)

                if (indicator1 <= 1) and (indicator2 <= 1):

                    c11p = c_cap[0][0] / ((np.abs(k) ** 2) * (np.abs(alpha) ** 2))
                    c12p = c_cap[0][1] * (np.conj(alpha) / (k * alpha))
                    c13p = c_cap[0][2] / (k * (np.abs(alpha) ** 2))
                    c14p = c_cap[0][3] * ((np.conj(k) * np.conj(alpha)) / (k * alpha))
                    c21p = c_cap[1][0] * (alpha / (np.conj(k) * np.conj(alpha)))
                    c22p = c_cap[1][1] * (np.abs(alpha) ** 2)
                    c23p = c_cap[1][2] * (alpha / np.conj(alpha))
                    c24p = c_cap[1][3] * (np.conj(k) * (np.abs(alpha) ** 2))
                    c31p = c_cap[2][0] / (np.conj(k) * (np.abs(alpha) ** 2))
                    c32p = c_cap[2][1] * (np.conj(alpha) / alpha)
                    c33p = c_cap[2][2] / (np.abs(alpha) ** 2)
                    c34p = c_cap[2][3] * ((np.conj(k) * np.conj(alpha)) / alpha)
                    c41p = c_cap[3][0] * ((k * alpha) / (np.conj(k) * np.conj(alpha)))
                    c42p = c_cap[3][1] * (k * (np.abs(alpha) ** 2))
                    c43p = c_cap[3][2] * ((k * alpha) / np.conj(alpha))
                    c44p = c_cap[3][3] * ((np.abs(k) ** 2) * (np.abs(alpha) ** 2))

                    sigma1 = np.array(
                        [
                            [c11p, c12p, c13p, c14p],
                            [c21p, c22p, c23p, c24p],
                            [c31p, c32p, c33p, c34p],
                            [c41p, c42p, c43p, c44p]
                        ]
                    )

                    a_cap = 0.5 * (c31p + c21p)
                    b_cap = 0.5 * (c34p + c24p)

                    zeta = np.array(
                        [
                            [0, 0, c41p, c11p],
                            [c11p, c41p, 0, 0],
                            [0, 0, c44p, c14p],
                            [c14p, c44p, 0, 0]
                        ]
                    )

                    tau = np.array(
                        [
                            [0, c22p, c23p, 0],
                            [0, c32p, c33p, 0],
                            [c22p, 0, 0, c23p],
                            [c32p, 0, 0, c33p]
                        ]
                    )

                    chi = np.array(
                        [
                            [c21p - a_cap],
                            [c31p - a_cap],
                            [c24p - b_cap],
                            [c34p - b_cap]
                        ]
                    )

                    rchi = chi.real
                    ichi = chi.imag

                    zpt = zeta + tau
                    zmt = zeta - tau

                    rzpt = zpt.real
                    izpt = zpt.imag

                    rzmt = zmt.real
                    izmt = zmt.imag

                    try:
                        rzmt_inv = np.linalg.inv(rzmt)
                        izmt_inv = np.linalg.inv(izmt)

                        t3 = np.linalg.inv(np.matmul(izmt_inv, rzpt) + np.matmul(rzmt_inv, izpt))
                    except np.linalg.linalg.LinAlgError as inv_err:
                        print(inv_err)
                        break
                    t4 = np.matmul(izmt_inv, rchi) + np.matmul(rzmt_inv, ichi)

                    rdelta = np.matmul(t3, t4)
                    idelta = np.matmul(rzmt_inv, (ichi - np.matmul(izpt, rdelta)))
                    delta = rdelta + (1j * idelta)

                    du = delta[0][0]
                    dv = delta[1][0]
                    dw = delta[2][0]
                    dz = delta[3][0]

                    u += du
                    v += dv
                    w += dw
                    z += dz

                    correction_left = (((1 - (v * z)) * (1 - (u * w))) ** -1) * np.array(
                        [
                            [1, -v, -w, v*w],
                            [-z, 1, w*z, -w],
                            [-u, u*v, 1, -v],
                            [u*z, -u, -z, 1]
                        ]
                    )

                    correction_right = np.conj(np.transpose(correction_left))

                    sigma2 = np.matmul(correction_left, np.matmul(sigma1, correction_right))

                    s_22 = sigma2[1][1]
                    s_32 = sigma2[2][1]
                    s_33 = sigma2[2][2]

                    alpha_update = ((s_33 / s_22) ** 0.25) * (np.exp(1j * (0.5 * np.arctan(s_32.imag / s_32.real))))
                    alpha *= alpha_update

                    m_t = alpha_update ** 2
                    m1 = np.diag([alpha, (alpha ** -1), alpha, (alpha ** -1)])
                    m2 = np.array(
                        [
                            [1, (v / m_t), w, ((v * w) / m_t)],
                            [(z * m_t), 1, (w * z * m_t), w],
                            [u, ((u * v) / m_t), 1, (v / m_t)],
                            [(u * z * m_t), u, (z * m_t), 1]
                        ]
                    )
                    m_cap = np.matmul(m1, m2)
                    m_cap_tranj = np.conj(np.transpose(m_cap))

                    c_cap = np.matmul(m_cap, np.matmul(sigma1, m_cap_tranj))

                    d_params = np.array([du, dv, dw, dz])
                    gamma = np.max(np.abs(d_params))

                    iter_count += 1

                else:
                    u = v = w = z = alpha = np.nan
                    break

            dq1.WriteArray(np.array([[indicator1]]), xoff=r, yoff=c)
            dq2.WriteArray(np.array([[indicator2]]), xoff=r, yoff=c)

            u = np.array([[u]])
            v = np.array([[v]])
            w = np.array([[w]])
            z = np.array([[z]])
            alpha = np.array([[alpha]])

            u_band.WriteArray(u, xoff=r, yoff=c)
            v_band.WriteArray(v, xoff=r, yoff=c)
            w_band.WriteArray(w, xoff=r, yoff=c)
            z_band.WriteArray(z, xoff=r, yoff=c)
            alpha_band.WriteArray(alpha, xoff=r, yoff=c)

            print(i, j)

    u_band.FlushCache()
    v_band.FlushCache()
    w_band.FlushCache()
    z_band.FlushCache()
    alpha_band.FlushCache()
    dq1.FlushCache()
    dq2.FlushCache()

    return 0


def range_binning(crstlk_dir, out_dir, range_direction=0):

    u_img = crstlk_dir + "u"
    v_img = crstlk_dir + "v"
    w_img = crstlk_dir + "w"
    z_img = crstlk_dir + "z"
    alpha_img = crstlk_dir + "alpha"
    u_f = gdal.Open(u_img)
    v_f = gdal.Open(v_img)
    w_f = gdal.Open(w_img)
    z_f = gdal.Open(z_img)
    alpha_f = gdal.Open(alpha_img)

    u_b = u_f.GetRasterBand(1)
    v_b = v_f.GetRasterBand(1)
    w_b = w_f.GetRasterBand(1)
    z_b = z_f.GetRasterBand(1)
    alpha_b = alpha_f.GetRasterBand(1)

    r = alpha_b.XSize
    c = alpha_b.YSize

    u_out = out_dir + "u"
    v_out = out_dir + "v"
    w_out = out_dir + "w"
    z_out = out_dir + "z"
    alpha_out = out_dir + "alpha"

    driver = gdal.GetDriverByName("ENVI")
    uout_f = driver.Create(u_out, r, c, 1, gdalconst.GDT_CFloat32)
    vout_f = driver.Create(v_out, r, c, 1, gdalconst.GDT_CFloat32)
    wout_f = driver.Create(w_out, r, c, 1, gdalconst.GDT_CFloat32)
    zout_f = driver.Create(z_out, r, c, 1, gdalconst.GDT_CFloat32)
    alphaout_f = driver.Create(alpha_out, r, c, 1, gdalconst.GDT_CFloat32)

    uout_b = uout_f.GetRasterBand(1)
    vout_b = vout_f.GetRasterBand(1)
    wout_b = wout_f.GetRasterBand(1)
    zout_b = zout_f.GetRasterBand(1)
    alphaout_b = alphaout_f.GetRasterBand(1)

    if range_direction == 0:
        u_dat = np.zeros((1, c), dtype=np.complex)
        u_n = np.zeros((1, c))
        v_dat = np.zeros((1, c), dtype=np.complex)
        v_n = np.zeros((1, c))
        w_dat = np.zeros((1, c), dtype=np.complex)
        w_n = np.zeros((1, c))
        z_dat = np.zeros((1, c), dtype=np.complex)
        z_n = np.zeros((1, c))
        alpha_dat = np.zeros((1, c), dtype=np.complex)
        alpha_n = np.zeros((1, c))

        for i in range(r):
            for j in range(c):
                u = u_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
                v = v_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
                w = w_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
                z = z_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
                alpha = alpha_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]

                if (not np.isnan(u)) and (np.abs(u) <= 1.0):
                    u_dat[0][j] += u
                    u_n[0][j] += 1
                    # print(u)

                if (not np.isnan(v)) and (np.abs(v) <= 1.0):
                    v_dat[0][j] += v
                    v_n[0][j] += 1
                    # print(v)

                if (not np.isnan(w)) and (np.abs(w) <= 1.0):
                    w_dat[0][j] += w
                    w_n[0][j] += 1
                    # print(w)

                if not (np.isnan(z)) and (np.abs(z) <= 1.0):
                    z_dat[0][j] += z
                    z_n[0][j] += 1
                    # print(z)

                if not np.isnan(alpha):
                    alpha_dat[0][j] += alpha
                    alpha_n[0][j] += 1
                    # print(alpha)

                print(i, j)

        # np.save(out_dir+'u_sum.npy', u_dat)
        # np.save(out_dir+'u_count.npy', u_n)
        # np.save(out_dir+'z_sum.npy', v_dat)
        # np.save(out_dir+'z_count.npy', v_n)
        # np.save(out_dir+'w_sum.npy', w_dat)
        # np.save(out_dir+'w_count.npy', w_n)
        # np.save(out_dir+'z_sum.npy', z_dat)
        # np.save(out_dir+'z_count.npy', z_n)
        # np.save(out_dir+'v_sum.npy', alpha_dat)
        # np.save(out_dir+'v_count.npy', alpha_n)

        u_avg = u_dat / u_n
        v_avg = v_dat / v_n
        w_avg = w_dat / w_n
        z_avg = z_dat / z_n
        alpha_avg = alpha_dat / alpha_n

        for r_id in range(r):
            uout_b.WriteArray(u_avg.T, xoff=r_id, yoff=0)
            vout_b.WriteArray(v_avg.T, xoff=r_id, yoff=0)
            wout_b.WriteArray(w_avg.T, xoff=r_id, yoff=0)
            zout_b.WriteArray(z_avg.T, xoff=r_id, yoff=0)
            alphaout_b.WriteArray(alpha_avg.T, xoff=r_id, yoff=0)

            print("Written: ", r_id)

        uout_b.FlushCache()
        vout_b.FlushCache()
        wout_b.FlushCache()
        zout_b.FlushCache()
        alphaout_b.FlushCache()

        return 0

    else:
        u_dat = np.zeros((r, 1), dtype=np.complex)
        u_n = np.zeros((r, 1))
        v_dat = np.zeros((r, 1), dtype=np.complex)
        v_n = np.zeros((r, 1))
        w_dat = np.zeros((r, 1), dtype=np.complex)
        w_n = np.zeros((r, 1))
        z_dat = np.zeros((r, 1), dtype=np.complex)
        z_n = np.zeros((r, 1))
        alpha_dat = np.zeros((r, 1), dtype=np.complex)
        alpha_n = np.zeros((r, 1))

        for i in range(r):
            for j in range(c):
                u = u_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
                v = v_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
                w = w_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
                z = z_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
                alpha = alpha_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]

                if (np.isfinite(u)) and (np.abs(u) <= 1.0):
                    u_dat[i][0] += u
                    u_n[i][0] += 1

                if (np.isfinite(v)) and (np.abs(v) <= 1.0):
                    v_dat[i][0] += v
                    v_n[i][0] += 1

                if (np.isfinite(w)) and (np.abs(w) <= 1.0):
                    w_dat[i][0] += w
                    w_n[i][0] += 1

                if (np.isfinite(z)) and (np.abs(z) <= 1.0):
                    z_dat[i][0] += z
                    z_n[i][0] += 1

                if np.isfinite(alpha):
                    alpha_dat[i][0] += alpha
                    alpha_n[i][0] += 1

                print(i, j)

        np.save(out_dir + 'u_sum.npy', u_dat)
        np.save(out_dir + 'u_count.npy', u_n)
        np.save(out_dir + 'z_sum.npy', v_dat)
        np.save(out_dir + 'z_count.npy', v_n)
        np.save(out_dir + 'w_sum.npy', w_dat)
        np.save(out_dir + 'w_count.npy', w_n)
        np.save(out_dir + 'z_sum.npy', z_dat)
        np.save(out_dir + 'z_count.npy', z_n)
        np.save(out_dir + 'v_sum.npy', alpha_dat)
        np.save(out_dir + 'v_count.npy', alpha_n)

        u_avg = u_dat / u_n
        v_avg = v_dat / v_n
        w_avg = w_dat / w_n
        z_avg = z_dat / z_n
        alpha_avg = alpha_dat / alpha_n

        for c_id in range(c):
            uout_b.WriteArray(u_avg.T, xoff=0, yoff=c_id)
            vout_b.WriteArray(v_avg.T, xoff=0, yoff=c_id)
            wout_b.WriteArray(w_avg.T, xoff=0, yoff=c_id)
            zout_b.WriteArray(z_avg.T, xoff=0, yoff=c_id)
            alphaout_b.WriteArray(alpha_avg.T, xoff=0, yoff=c_id)

            print("Written: ", c_id)

        uout_b.FlushCache()
        vout_b.FlushCache()
        wout_b.FlushCache()
        zout_b.FlushCache()
        alphaout_b.FlushCache()

        return 0
