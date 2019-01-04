import gdal
import gdalconst
import numpy as np


def apply_correction(
        hh_img,
        hv_img,
        vh_img,
        vv_img,
        params_dir,
        out_dir,
        error_map,
        xoffset=None,
        yoffset=None
):
    u_img = params_dir + "u"
    v_img = params_dir + "v"
    w_img = params_dir + "w"
    z_img = params_dir + "z"
    alpha_img = params_dir + "alpha"

    hh_cal_img = out_dir + "HH_Cal"
    hv_cal_img = out_dir + "HV_Cal"
    vh_cal_img = out_dir + "VH_Cal"
    vv_cal_img = out_dir + "VV_Cal"

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

    hh_f = gdal.Open(hh_img)
    hv_f = gdal.Open(hv_img)
    vh_f = gdal.Open(vh_img)
    vv_f = gdal.Open(vv_img)

    hh_b = hh_f.GetRasterBand(1)
    hv_b = hv_f.GetRasterBand(1)
    vh_b = vh_f.GetRasterBand(1)
    vv_b = vv_f.GetRasterBand(1)

    if xoffset is None:
        xoffset = hh_b.XSize

    if yoffset is None:
        yoffset = hh_b.YSize

    driver = gdal.GetDriverByName("ENVI")
    hh_cal_f = driver.Create(hh_cal_img, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    hv_cal_f = driver.Create(hv_cal_img, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    vh_cal_f = driver.Create(vh_cal_img, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)
    vv_cal_f = driver.Create(vv_cal_img, xoffset, yoffset, 1, gdalconst.GDT_CFloat32)

    hh_cal_b = hh_cal_f.GetRasterBand(1)
    hv_cal_b = hv_cal_f.GetRasterBand(1)
    vh_cal_b = vh_cal_f.GetRasterBand(1)
    vv_cal_b = vv_cal_f.GetRasterBand(1)

    err_f = driver.Create(error_map, xoffset, yoffset, 1, gdalconst.GDT_Byte)
    err_b = err_f.GetRasterBand(1)

    for i in range(xoffset):
        for j in range(yoffset):

            hh = hh_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            hv = hv_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            vh = vh_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            vv = vv_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]

            u = u_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            v = v_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            w = w_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            z = z_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            alpha = alpha_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]

            if not (np.isnan(u) or np.isnan(v) or np.isnan(w) or np.isnan(z) or np.isnan(alpha)):

                if (u != 0) and (v != 0) and (w != 0) and (z != 0) and (alpha != 0):

                    denom = (((u * w) - 1) * ((v * z) - 1))

                    if alpha != 0 and denom != 0:

                        constant = 1 / denom
                        c_mat = np.array(
                            [
                                [
                                    (1+0j),
                                    (-1 * w),
                                    (-1 * v),
                                    (v * w)
                                ],
                                [
                                    ((-1 * u) / (alpha ** 0.5)),
                                    (1 / (alpha ** 0.5)),
                                    ((u * v) / (alpha ** 0.5)),
                                    ((-1 * v) / (alpha ** 0.5))
                                ],
                                [
                                    ((-1 * z) / (alpha ** 0.5)),
                                    (w * z * (alpha ** 0.5)),
                                    (alpha ** 0.5),
                                    (-1 * w * (alpha ** 0.5))
                                ],
                                [
                                    (u * z),
                                    (-1 * z),
                                    (-1 * u),
                                    (1 + 0j)
                                ]
                            ]
                        )

                        correction_mat = constant * c_mat

                        observed_mat = np.array(
                            [
                                [hh],
                                [vh],
                                [hv],
                                [vv]
                            ]
                        )

                        corrected_mat = np.matmul(correction_mat, observed_mat)

                        hh_cal = corrected_mat[0:1, :]
                        vh_cal = corrected_mat[1:2, :]
                        hv_cal = corrected_mat[2:3, :]
                        vv_cal = corrected_mat[3:4, :]

                        # No Error
                        err = np.array([[0]])

                    else:
                        hh_cal = np.array([[hh]])
                        vh_cal = np.array([[vh]])
                        hv_cal = np.array([[hv]])
                        vv_cal = np.array([[vv]])

                        # Inverse Error
                        err = np.array([[255]])
                else:
                    hh_cal = np.array([[hh]])
                    vh_cal = np.array([[vh]])
                    hv_cal = np.array([[hv]])
                    vv_cal = np.array([[vv]])

                    # Delta is Zero
                    err = np.array([[127]])

            else:
                hh_cal = np.array([[hh]])
                vh_cal = np.array([[vh]])
                hv_cal = np.array([[hv]])
                vv_cal = np.array([[vv]])

                # Ignored Pixel
                err = np.array([[np.nan]])

            hh_cal_b.WriteArray(hh_cal, xoff=i, yoff=j)
            hv_cal_b.WriteArray(hv_cal, xoff=i, yoff=j)
            vh_cal_b.WriteArray(vh_cal, xoff=i, yoff=j)
            vv_cal_b.WriteArray(vv_cal, xoff=i, yoff=j)
            err_b.WriteArray(err, xoff=i, yoff=j)

            print(i, j)

    hh_cal_b.FlushCache()
    hv_cal_b.FlushCache()
    vh_cal_b.FlushCache()
    vv_cal_b.FlushCache()
    err_b.FlushCache()
