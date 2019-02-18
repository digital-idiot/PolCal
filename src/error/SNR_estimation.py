import gdal
import numpy as np


def estimate_snr(c1_img, c2_img, kernel_size=3, ratio=3, x_start=0, y_start=0, x_offset=None, y_offset=None,
                 range_direction=1):
    c1_f = gdal.Open(c1_img)
    c2_f = gdal.Open(c2_img)

    c1_b = c1_f.GetRasterBand(1)
    c2_b = c2_f.GetRasterBand(1)

    max_x = c1_f.RasterXSize
    max_y = c2_f.RasterYSize

    if x_offset is None:
        x_offset = max_x

    if y_offset is None:
        y_offset = max_y

    if range_direction == 0:
        winxsize = kernel_size * ratio
        winysize = kernel_size
    else:
        winxsize = kernel_size
        winysize = kernel_size * ratio

    snr_dat = np.zeros((x_offset, y_offset))

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

            c1_win = c1_b.ReadAsArray(xoff=win_startx, yoff=win_starty, win_xsize=winx, win_ysize=winy)
            c2_win = c2_b.ReadAsArray(xoff=win_startx, yoff=win_starty, win_xsize=winx, win_ysize=winy)

            u1 = np.mean(c1_win)
            u2 = np.mean(c2_win)

            num = 2 * (np.sum(u2 * np.conjugate(u2))).real
            denom = np.sum((np.abs(u1 - u2)) ** 2)
            snr = num / denom
            snr_dat[i, j] = snr
