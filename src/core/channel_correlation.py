import gdal
import gdalconst
import numpy as np


def logicboard(cov_dir, out_img, threshold=0.2, x_start=0, y_start=0, xoffset=None, yoffset=None):

    c11_f = gdal.Open(cov_dir + "C11")
    c12_f = gdal.Open(cov_dir + "C12")
    # c13_f = gdal.Open(cov_dir + "C13")
    # c14_f = gdal.Open(cov_dir + "C14")
    # c21_f = gdal.Open(cov_dir + "C21")
    c22_f = gdal.Open(cov_dir + "C22")
    # c23_f = gdal.Open(cov_dir + "C23")
    # c24_f = gdal.Open(cov_dir + "C24")
    # c31_f = gdal.Open(cov_dir + "C31")
    # c32_f = gdal.Open(cov_dir + "C32")
    c33_f = gdal.Open(cov_dir + "C33")
    # c34_f = gdal.Open(cov_dir + "C34")
    # c41_f = gdal.Open(cov_dir + "C41")
    # c42_f = gdal.Open(cov_dir + "C42")
    c43_f = gdal.Open(cov_dir + "C43")
    c44_f = gdal.Open(cov_dir + "C44")

    c11b = c11_f.GetRasterBand(1)
    c12b = c12_f.GetRasterBand(1)
    # c13b = c13_f.GetRasterBand(1)
    # c14b = c14_f.GetRasterBand(1)
    # c21b = c21_f.GetRasterBand(1)
    c22b = c22_f.GetRasterBand(1)
    # c23b = c23_f.GetRasterBand(1)
    # c24b = c24_f.GetRasterBand(1)
    # c31b = c31_f.GetRasterBand(1)
    # c32b = c32_f.GetRasterBand(1)
    c33b = c33_f.GetRasterBand(1)
    # c34b = c34_f.GetRasterBand(1)
    # c41b = c41_f.GetRasterBand(1)
    # c42b = c42_f.GetRasterBand(1)
    c43b = c43_f.GetRasterBand(1)
    c44b = c44_f.GetRasterBand(1)

    if xoffset is None:
        xoffset = c11b.XSize
    if yoffset is None:
        yoffset = c11b.YSize

    driver = gdal.GetDriverByName("ENVI")
    logic_img = driver.Create(out_img, xoffset, yoffset, 1, gdalconst.GDT_Byte)
    logic_band = logic_img.GetRasterBand(1)

    for i in range(xoffset):
        for j in range(yoffset):
            r = x_start + i
            c = y_start + j

            c11 = c11b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            c12 = c12b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            # c13 = c13b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            # c14 = c14b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            # c21 = c21b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            c22 = c22b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            # c23 = c23b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            # c24 = c24b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            # c31 = c31b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            # c32 = c32b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            c33 = c33b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            # c34 = c34b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            # c41 = c41b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            # c42 = c42b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            c43 = c43b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)
            c44 = c44b.ReadAsArray(xoff=r, yoff=c, win_xsize=1, win_ysize=1)

            r_hhvh = c12 / ((c11 * c22) ** 0.5)
            r_vvhv = c43 / ((c44 * c33) ** 0.5)

            if (r_hhvh < threshold) or (r_vvhv < threshold):
                logic_band.WriteArray(np.array([[255]]), xoff=i, yoff=j)
            else:
                logic_band.WriteArray(np.array([[0]]), xoff=i, yoff=j)

            print(i, j)

    logic_band.FlushCache()

    return 0
