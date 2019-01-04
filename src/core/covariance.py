import gdal
import gdalconst
import numpy as np


def compute_covar(
        hh_img,
        hv_img,
        vh_img,
        vv_img,
        out_dir,
        x_start=0,
        y_start=0,
        x_offset=None,
        y_offset=None,
        winxsize=2,
        winysize=2,
):

    hh_f = gdal.Open(hh_img)
    hv_f = gdal.Open(hv_img)
    vh_f = gdal.Open(vh_img)
    vv_f = gdal.Open(vv_img)

    hh_b = hh_f.GetRasterBand(1)
    hv_b = hv_f.GetRasterBand(1)
    vh_b = vh_f.GetRasterBand(1)
    vv_b = vv_f.GetRasterBand(1)

    max_x = hh_b.XSize
    max_y = hh_b.YSize

    if x_offset is None:
        x_offset = max_x

    if y_offset is None:
        y_offset = max_y

    c11_f = out_dir + "C11"
    c12_f = out_dir + "C12"
    c13_f = out_dir + "C13"
    c14_f = out_dir + "C14"
    c21_f = out_dir + "C21"
    c22_f = out_dir + "C22"
    c23_f = out_dir + "C23"
    c24_f = out_dir + "C24"
    c31_f = out_dir + "C31"
    c32_f = out_dir + "C32"
    c33_f = out_dir + "C33"
    c34_f = out_dir + "C34"
    c41_f = out_dir + "C41"
    c42_f = out_dir + "C42"
    c43_f = out_dir + "C43"
    c44_f = out_dir + "C44"

    driver = gdal.GetDriverByName("ENVI")
    c11_img = driver.Create(c11_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c12_img = driver.Create(c12_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c13_img = driver.Create(c13_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c14_img = driver.Create(c14_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c21_img = driver.Create(c21_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c22_img = driver.Create(c22_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c23_img = driver.Create(c23_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c24_img = driver.Create(c24_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c31_img = driver.Create(c31_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c32_img = driver.Create(c32_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c33_img = driver.Create(c33_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c34_img = driver.Create(c34_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c41_img = driver.Create(c41_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c42_img = driver.Create(c42_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c43_img = driver.Create(c43_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)
    c44_img = driver.Create(c44_f, x_offset, y_offset, 1, gdalconst.GDT_CFloat32)

    c11b = c11_img.GetRasterBand(1)
    c12b = c12_img.GetRasterBand(1)
    c13b = c13_img.GetRasterBand(1)
    c14b = c14_img.GetRasterBand(1)
    c21b = c21_img.GetRasterBand(1)
    c22b = c22_img.GetRasterBand(1)
    c23b = c23_img.GetRasterBand(1)
    c24b = c24_img.GetRasterBand(1)
    c31b = c31_img.GetRasterBand(1)
    c32b = c32_img.GetRasterBand(1)
    c33b = c33_img.GetRasterBand(1)
    c34b = c34_img.GetRasterBand(1)
    c41b = c41_img.GetRasterBand(1)
    c42b = c42_img.GetRasterBand(1)
    c43b = c43_img.GetRasterBand(1)
    c44b = c44_img.GetRasterBand(1)

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

            hh_win = hh_b.ReadAsArray(xoff=win_startx, yoff=win_starty, win_xsize=winx, win_ysize=winy)
            hv_win = hv_b.ReadAsArray(xoff=win_startx, yoff=win_starty, win_xsize=winx, win_ysize=winy)
            vh_win = vh_b.ReadAsArray(xoff=win_startx, yoff=win_starty, win_xsize=winx, win_ysize=winy)
            vv_win = vv_b.ReadAsArray(xoff=win_startx, yoff=win_starty, win_xsize=winx, win_ysize=winy)

            scatt_dump = np.dstack([hh_win, vh_win, hv_win, vv_win])

            winr, winc, winh = scatt_dump.shape
            cov_sum = None
            for m in range(winr):
                for n in range(winc):
                    scatt_mat = scatt_dump[m][n]
                    scatt_mat = scatt_mat.reshape((-1, scatt_mat.size))
                    co_var = np.matmul(np.transpose(scatt_mat), np.conj(scatt_mat))
                    if cov_sum is None:
                        cov_sum = co_var
                    else:
                        cov_sum = cov_sum + co_var

            avg = cov_sum / (winr * winc)
            c11b.WriteArray(avg[0:1, 0:1], xoff=i, yoff=j)
            # c11b.FlushCache()
            c12b.WriteArray(avg[0:1, 1:2], xoff=i, yoff=j)
            # c12b.FlushCache()
            c13b.WriteArray(avg[0:1, 2:3], xoff=i, yoff=j)
            # c13b.FlushCache()
            c14b.WriteArray(avg[0:1, 3:4], xoff=i, yoff=j)
            # c14b.FlushCache()
            c21b.WriteArray(avg[1:2, 0:1], xoff=i, yoff=j)
            # c21b.FlushCache()
            c22b.WriteArray(avg[1:2, 1:2], xoff=i, yoff=j)
            # c22b.FlushCache()
            c23b.WriteArray(avg[1:2, 2:3], xoff=i, yoff=j)
            # c23b.FlushCache()
            c24b.WriteArray(avg[1:2, 3:4], xoff=i, yoff=j)
            # c24b.FlushCache()
            c31b.WriteArray(avg[2:3, 0:1], xoff=i, yoff=j)
            # c31b.FlushCache()
            c32b.WriteArray(avg[2:3, 1:2], xoff=i, yoff=j)
            # c32b.FlushCache()
            c33b.WriteArray(avg[2:3, 2:3], xoff=i, yoff=j)
            # c33b.FlushCache()
            c34b.WriteArray(avg[2:3, 3:4], xoff=i, yoff=j)
            # c34b.FlushCache()
            c41b.WriteArray(avg[3:4, 0:1], xoff=i, yoff=j)
            # c41b.FlushCache()
            c42b.WriteArray(avg[3:4, 1:2], xoff=i, yoff=j)
            # c42b.FlushCache()
            c43b.WriteArray(avg[3:4, 2:3], xoff=i, yoff=j)
            # c43b.FlushCache()
            c44b.WriteArray(avg[3:4, 3:4], xoff=i, yoff=j)
            # c44b.FlushCache()
            print(i, j)

    c11b.FlushCache()
    c12b.FlushCache()
    c13b.FlushCache()
    c14b.FlushCache()
    c21b.FlushCache()
    c22b.FlushCache()
    c23b.FlushCache()
    c24b.FlushCache()
    c31b.FlushCache()
    c32b.FlushCache()
    c33b.FlushCache()
    c34b.FlushCache()
    c41b.FlushCache()
    c42b.FlushCache()
    c43b.FlushCache()
    c44b.FlushCache()
