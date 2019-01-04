import pandas as pd
import numpy as np
import gdal
import gdalconst


def radiometric_cal(
        hh_img,
        hv_img,
        vh_img,
        vv_img,
        php_img,
        copol_info,
        xpol_info,
        hh_out,
        hv_out,
        vh_out,
        vv_out,
        abs_img=None,
        row_start=0,
        col_start=0,
        row_offset=None,
        col_offset=None,
):
    hh_f = gdal.Open(hh_img)
    hh_b = hh_f.GetRasterBand(1)
    hv_f = gdal.Open(hv_img)
    hv_b = hv_f.GetRasterBand(1)
    vh_f = gdal.Open(vh_img)
    vh_b = vh_f.GetRasterBand(1)
    vv_f = gdal.Open(vv_img)
    vv_b = vv_f.GetRasterBand(1)

    if abs_img is None:
        abs_b = None
    else:
        abs_f = gdal.Open(abs_img)
        abs_b = abs_f.GetRasterBand(1)

    php_f = gdal.Open(php_img)
    php_b = php_f.GetRasterBand(1)

    if row_offset is None:
        row_offset = hh_b.XSize
    if col_offset is None:
        col_offset = hh_b.YSize

    cdat = pd.read_csv(copol_info, usecols=['f'])
    xdat = pd.read_csv(xpol_info)

    f = np.mean(cdat['f'])

    if xdat.shape[0] == 1:
        g = xdat['g'][0]
        phm = xdat['phi_m'][0]

        driver = gdal.GetDriverByName("ENVI")
        out_hh = driver.Create(hh_out, row_offset, col_offset, 1, gdalconst.GDT_CFloat64)
        b_hh = out_hh.GetRasterBand(1)
        out_hv = driver.Create(hv_out, row_offset, col_offset, 1, gdalconst.GDT_CFloat64)
        b_hv = out_hv.GetRasterBand(1)
        out_vh = driver.Create(vh_out, row_offset, col_offset, 1, gdalconst.GDT_CFloat64)
        b_vh = out_vh.GetRasterBand(1)
        out_vv = driver.Create(vv_out, row_offset, col_offset, 1, gdalconst.GDT_CFloat64)
        b_vv = out_vv.GetRasterBand(1)

        for i in range(row_offset):
            for j in range(col_offset):
                r = row_start + i
                c = col_start + j
                hh = hh_b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                hv = hv_b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                vh = vh_b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                vv = vv_b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]

                if abs_b is None:
                    a = 1.0
                else:
                    a = abs_b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]

                php = php_b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]

                # little hack for now
                # php = np.deg2rad(php)

                phi_t = np.deg2rad((php + phm) / 2.0)
                phi_r = np.deg2rad((php - phm) / 2.0)
                hh_cal = np.array([[hh * a]])
                hv_cal = np.array([[hv * (a * f * g * np.exp(1j * phi_t))]])
                vh_cal = np.array([[vh * (a * (f / g) * np.exp(1j * phi_r))]])
                vv_cal = np.array([[vv * (a * f * f * np.exp(1j * (phi_t + phi_r)))]])

                b_hh.WriteArray(hh_cal, xoff=i, yoff=j)
                b_hv.WriteArray(hv_cal, xoff=i, yoff=j)
                b_vh.WriteArray(vh_cal, xoff=i, yoff=j)
                b_vv.WriteArray(vv_cal, xoff=i, yoff=j)

        b_hh.FlushCache()
        b_hv.FlushCache()
        b_vh.FlushCache()
        b_vv.FlushCache()
