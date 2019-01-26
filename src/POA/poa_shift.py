import gdal
import numpy as np


def estimate_poa_shift(
        hh_img,
        hv_img,
        vv_img,
        orientation_img,
        x_start=0,
        y_start=0,
        x_offset=None,
        y_offset=None,
        winxsize=2,
        winysize=2,
):

    hh_f = gdal.Open(hh_img)
    hv_f = gdal.Open(hv_img)
    vv_f = gdal.Open(vv_img)

    hh_b = hh_f.GetRasterBand(1)
    hv_b = hv_f.GetRasterBand(1)
    vv_b = vv_f.GetRasterBand(1)

    max_x = hh_b.XSize
    max_y = hh_b.YSize

    if x_offset is None:
        x_offset = max_x

    if y_offset is None:
        y_offset = max_y

    driver = gdal.GetDriverByName("ENVI")
    orientation_f = driver.Create(orientation_img, x_offset, y_offset, 1, gdal.GDT_Float32)
    orientation_b = orientation_f.GetRasterBand(1)

    for i in range(x_offset):
        for j in range(y_offset):
            r = x_start + i
            c = y_start + j
            win_startx = r - winxsize
            win_starty = c - winysize

            winx = (2 * winxsize) + 1
            winy = (2 * winysize) + 1

            if win_startx < 0:
                win_startx = 0

            if win_starty < 0:
                win_starty = 0

            if win_startx + winx > max_x:
                winx = max_x - win_startx

            if win_starty + winy > max_y:
                winy = max_y - win_starty

            hh_win = hh_b.ReadAsArray(
                xoff=win_startx,
                yoff=win_starty,
                win_xsize=winx,
                win_ysize=winy
            )

            hv_win = hv_b.ReadAsArray(
                xoff=win_startx,
                yoff=win_starty,
                win_xsize=winx,
                win_ysize=winy
            )

            vv_win = vv_b.ReadAsArray(
                xoff=win_startx,
                yoff=win_starty,
                win_xsize=winx,
                win_ysize=winy
            )

            co_diff = hh_win - vv_win
            hv_conj = np.conj(hv_win)
            hv_mod2 = (hv_win * hv_conj).real
            co_diff_mod2 = (co_diff * np.conj(co_diff)).real

            numerator = -4.0 * (np.nanmean(co_diff * hv_conj)).real
            denominator = (4.0 * np.nanmean(hv_mod2)) - (np.nanmean(co_diff_mod2))
            eta = 0.25 * (np.pi + np.arctan2(numerator, denominator))

            theta = eta
            if eta > (np.pi / 4):
                theta = eta - (np.pi / 2)

            orientation_b.WriteArray(np.array([[theta]]), xoff=i, yoff=j)
            print(i, j)

    orientation_b.FlushCache()
    return 0


def poa_shift_correction(hh_img, hv_img, vh_img, vv_img, poa_img, out_dir, deg=False):

    hh_f = gdal.Open(hh_img)
    hv_f = gdal.Open(hv_img)
    vh_f = gdal.Open(vh_img)
    vv_f = gdal.Open(vv_img)
    poa_f = gdal.Open(poa_img)

    hh_b = hh_f.GetRasterBand(1)
    hv_b = hv_f.GetRasterBand(1)
    vh_b = vh_f.GetRasterBand(1)
    vv_b = vv_f.GetRasterBand(1)
    poa_b = poa_f.GetRasterBand(1)

    xoffset = poa_b.XSize
    yoffset = poa_b.YSize

    driver = gdal.GetDriverByName("ENVI")

    hh_out_f = driver.Create(out_dir + "HH_P", xoffset, yoffset, 1, gdal.GDT_CFloat32)
    hv_out_f = driver.Create(out_dir + "HV_P", xoffset, yoffset, 1, gdal.GDT_CFloat32)
    vh_out_f = driver.Create(out_dir + "VH_P", xoffset, yoffset, 1, gdal.GDT_CFloat32)
    vv_out_f = driver.Create(out_dir + "VV_P", xoffset, yoffset, 1, gdal.GDT_CFloat32)

    hh_out_b = hh_out_f.GetRasterBand(1)
    hv_out_b = hv_out_f.GetRasterBand(1)
    vh_out_b = vh_out_f.GetRasterBand(1)
    vv_out_b = vv_out_f.GetRasterBand(1)

    for i in range(xoffset):
        for j in range(yoffset):

            hh_in = hh_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            hv_in = hv_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            vh_in = vh_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            vv_in = vv_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            poa = poa_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]

            if deg:
                poa = np.rad2deg(poa)

            scattering_matrix = np.array(
                [
                    [hh_in, hv_in],
                    [vh_in, vv_in],
                ]
            )

            correction_left = np.array(
                [
                    [np.cos(poa), np.sin(poa)],
                    [-np.sin(poa), np.cos(poa)]
                ]
            )

            correction_right = np.array(
                [
                    [np.cos(poa), -np.sin(poa)],
                    [np.sin(poa), np.cos(poa)]
                ]
            )

            corrected_scattering_marix = np.matmul(
                correction_left,
                np.matmul(
                    scattering_matrix,
                    correction_right
                )
            )

            hh_out = np.array(
                [
                    [
                        corrected_scattering_marix[0, 0]
                    ]
                ]
            )

            hv_out = np.array(
                [
                    [
                        corrected_scattering_marix[0, 1]
                    ]
                ]
            )

            vh_out = np.array(
                [
                    [
                        corrected_scattering_marix[1, 0]
                    ]
                ]
            )

            vv_out = np.array(
                [
                    [
                        corrected_scattering_marix[1, 1]
                    ]
                ]
            )

            hh_out_b.WriteArray(hh_out, xoff=i, yoff=j)
            hv_out_b.WriteArray(hv_out, xoff=i, yoff=j)
            vh_out_b.WriteArray(vh_out, xoff=i, yoff=j)
            vv_out_b.WriteArray(vv_out, xoff=i, yoff=j)

            print(i, j)

    hh_out_b.FlushCache()
    hv_out_b.FlushCache()
    vh_out_b.FlushCache()
    vv_out_b.FlushCache()

    return 0
