import numpy as np
import pandas as pd
import gdal
import gdalconst


def rcs_trihedral_cr(cr_len, wavelength, inc_angle, tilt_angle, azimuth_angle):

    tilt_angle = np.deg2rad(tilt_angle)
    azimuth_angle = np.deg2rad(azimuth_angle)

    theta = inc_angle + tilt_angle
    t1 = (4 * np.pi * (cr_len ** 4)) / (wavelength ** 2)
    t2 = np.cos(theta) + ((np.sin(theta)) * (np.sin(azimuth_angle) + np.cos(azimuth_angle)))
    t3 = 2.0 / t2
    t4 = (t2 - t3) ** 2
    rcs = t1 * t4
    return rcs


def absolute_cal(s_hh, rcs):
    a = np.abs(s_hh) / (rcs ** 0.5)
    return a


def f_at_cr(s_hh, s_vv):
    denominator = s_hh * np.conj(s_hh)
    numerator = s_vv * np.conj(s_vv)
    f = (numerator / denominator) ** 0.25
    return f.real


def g_at_dist(hv_arr, vh_arr):
    numerator = np.mean(np.abs(hv_arr) ** 2)
    denominator = np.mean(np.abs(vh_arr) ** 2)
    g = (numerator / denominator) ** 0.25
    return numerator, denominator, g


def phi_plus_cr(s_hh, s_vv):
    added = s_vv * np.conj(s_hh)
    return np.arctan(added.imag / added.real)


def phi_minus_dist(hv_arr, vh_arr):
    diff = np.mean(hv_arr * vh_arr)
    return diff, np.arctan(diff.imag / diff.real)


def radio_copol(hh_img, vv_img, inc_img, wave_length, cr_info, out_file=None):

    hh_f = gdal.Open(hh_img)
    vv_f = gdal.Open(vv_img)
    inc_f = gdal.Open(inc_img)

    hh_band = hh_f.GetRasterBand(1)
    vv_band = vv_f.GetRasterBand(1)
    inc_band = inc_f.GetRasterBand(1)

    data = pd.read_csv(
        cr_info,
        usecols=[
            "CR_ID",
            "Azimuth_Angle",
            "Tilt_Angle",
            "Side_Length",
            "loc_row",
            "loc_col",
            "Side_Length"
        ]
    )
    out_dat = list()

    # dummy
    r1 = "/home/abhisek/Public/hh_abs"
    r2 = "/home/abhisek/Public/vv_abs"
    i = 1

    for row in data.itertuples():

        cr_id = row.CR_ID
        r = row.loc_row
        c = row.loc_col

        cr_len = row.Side_Length
        phi = np.deg2rad(row.Azimuth_Angle)
        inc_a = (inc_band.ReadAsArray(xoff=c, yoff=r, win_xsize=1, win_ysize=1))[0][0]
        tilt_a = np.deg2rad(row.Tilt_Angle)

        hh_pix = (hh_band.ReadAsArray(xoff=c, yoff=r, win_xsize=1, win_ysize=1))[0][0]
        vv_pix = (vv_band.ReadAsArray(xoff=c, yoff=r, win_xsize=1, win_ysize=1))[0][0]

        # Check
        drv = gdal.GetDriverByName("GTiff")
        f1 = drv.Create(r1 + str(i), 50, 50, 1, gdal.GDT_Float32)
        f2 = drv.Create(r2 + str(i), 50, 50, 1, gdal.GDT_Float32)
        b1 = f1.GetRasterBand(1)
        b2 = f2.GetRasterBand(1)
        hh_k = np.abs(hh_band.ReadAsArray(xoff=c - 24, yoff=r - 24, win_xsize=50, win_ysize=50))
        vv_k = np.abs(vv_band.ReadAsArray(xoff=c - 24, yoff=r - 24, win_xsize=50, win_ysize=50))
        b1.WriteArray(hh_k)
        b2.WriteArray(vv_k)
        b1.FlushCache()
        b2.FlushCache()
        i += 1

        sigma_0 = rcs_trihedral_cr(
            cr_len=cr_len,
            wavelength=wave_length,
            inc_angle=inc_a,
            tilt_angle=tilt_a,
            azimuth_angle=phi
        )

        a = absolute_cal(s_hh=hh_pix, rcs=sigma_0)
        f = f_at_cr(s_hh=hh_pix, s_vv=vv_pix)
        phi_p = phi_plus_cr(s_hh=hh_pix, s_vv=vv_pix)
        out_dat.append([cr_id, cr_len, r, c, inc_a, a, f, phi_p, hh_pix, vv_pix, sigma_0])

    df = pd.DataFrame(
        out_dat, columns=[
            'CR_ID',
            'CR_LEN',
            'ROW_ID',
            'COL_ID',
            'INC_ANGLE',
            'A',
            'f',
            'Phi_PLUS',
            "HH",
            "VV",
            "RCS"
        ]
    )
    if out_file is not None:
        df.to_csv(out_file, sep=',', encoding='utf-8', index=None)

    return df


def radio_xpol(hv_img, vh_img, row_offset=None, col_offset=None, outfile=None, row_start=0, col_start=0):

    hv_f = gdal.Open(hv_img)
    vh_f = gdal.Open(vh_img)

    hv_band = hv_f.GetRasterBand(1)
    vh_band = vh_f.GetRasterBand(1)

    if row_offset is None:
        row_offset = hv_band.YSize

    if col_offset is None:
        col_offset = hv_band.XSize

    hv_homogeneous = hv_band.ReadAsArray(xoff=col_start, yoff=row_start, win_xsize=col_offset, win_ysize=row_offset)
    vh_homogeneous = vh_band.ReadAsArray(xoff=col_start, yoff=row_start, win_xsize=col_offset, win_ysize=row_offset)

    n, d, g = g_at_dist(hv_arr=hv_homogeneous, vh_arr=vh_homogeneous)
    dif, phi_m = phi_minus_dist(hv_arr=hv_homogeneous, vh_arr=vh_homogeneous)
    dat = [[n, d, dif, g, phi_m]]
    df = pd.DataFrame(dat, columns=['numerator', 'denominator', 'diff', 'g', 'phi_m'])
    if outfile is not None:
        df.to_csv(outfile, sep=',', encoding='utf-8', index=None)
    return df


def fit_radio_params(copol_params, a_deg=1, phi_deg=1, abs_file=None, php_file=None):
    data = pd.read_csv(
        copol_params,
        usecols=["CR_ID", "INC_ANGLE", "A", "Phi_PLUS"]
    )

    inc_arr = list()
    abs_arr = list()
    php_arr = list()

    for row in data.itertuples():
        inc = row.INC_ANGLE
        a = row.A
        phi_p = row.Phi_PLUS

        inc_arr.append(inc)
        abs_arr.append(a)
        php_arr.append(phi_p)

    abs_arr = np.array(abs_arr)
    php_arr = np.array(php_arr)
    inc_arr = np.array(inc_arr) - np.deg2rad(45)

    abs_coeffs = [np.polyfit(x=inc_arr, y=abs_arr, deg=a_deg)]
    php_coeffs = [np.polyfit(x=inc_arr, y=php_arr, deg=phi_deg)]
    abs_sub = np.arange(a_deg + 1).astype(str)
    php_sub = np.arange(phi_deg + 1).astype(str)

    abs_hdr = list(reversed(np.core.defchararray.add("A_", abs_sub).tolist()))
    php_hdr = list(reversed(np.core.defchararray.add("a_", php_sub).tolist()))

    df1 = pd.DataFrame(abs_coeffs, columns=abs_hdr)
    df2 = pd.DataFrame(php_coeffs, columns=php_hdr)

    if abs_file is not None:
        df1.to_csv(abs_file, sep=',', encoding='utf-8', index=None)

    if php_file is not None:
        df2.to_csv(php_file, sep=',', encoding='utf-8', index=None)

    return df1, df2


def write_copol_params(
        abs_coeffs,
        phi_coeffs,
        inc_img,
        abs_img_out,
        php_img_out,
        coeff_deg=False,
        row_start=0,
        col_start=0,
        row_offset=None,
        col_offset=None
):
    inc_file = gdal.Open(inc_img)
    inc_band = inc_file.GetRasterBand(1)
    if row_offset is None:
        row_offset = inc_band.XSize
    if col_offset is None:
        col_offset = inc_band.YSize
    dat1 = pd.read_csv(abs_coeffs)
    dat2 = pd.read_csv(phi_coeffs)

    i_r, i_c = dat1.shape
    p_r, p_c = dat2.shape

    if i_r == 1 and p_r == 1:
        i_coeffs = dat1.iloc[0].tolist()
        p_coeffs = dat2.iloc[0].tolist()
        driver = gdal.GetDriverByName("ENVI")
        outfile_i = driver.Create(abs_img_out, row_offset, col_offset, 1, gdalconst.GDT_Float32)
        band_i = outfile_i.GetRasterBand(1)
        outfile_p = driver.Create(php_img_out, row_offset, col_offset, 1, gdalconst.GDT_Float32)
        band_p = outfile_p.GetRasterBand(1)

        for i in range(row_offset):
            for j in range(col_offset):
                r = row_start + i
                c = col_start + j
                inc_pix = inc_band.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)[0][0]
                inc = inc_pix - (np.pi / 4.0)
                if coeff_deg:
                    inc = np.rad2deg(inc)

                abs_pix = np.polyval(i_coeffs, inc)
                abs_pix = np.array([[abs_pix]])
                band_i.WriteArray(abs_pix, xoff=i, yoff=j)

                phip_pix = np.polyval(p_coeffs, inc)
                phip_pix = np.array([[phip_pix]])

                band_p.WriteArray(phip_pix, xoff=i, yoff=j)
        if band_i is not None:
            band_i.FlushCache()
        band_p.FlushCache()
