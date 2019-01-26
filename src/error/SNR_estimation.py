import gdal
import numpy as np


def estimate_SNR(c1_img, c2_img):

    c1_f = gdal.Open(c1_img)
    c2_f = gdal.Open(c2_img)

    c1_b = c1_f.GetRasterBand(1)
    c2_b = c2_f.GetRasterBand(1)

    xoffset = c1_b.XSize
    yoffset = c2_b.YSize

    numerator1 = 0.0
    denominator = 0.0
    numerator2 = 0.0
    count = 0
    for i in range(xoffset):
        for j in range(yoffset):
            c1 = c1_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]
            c2 = c2_b.ReadAsArray(xoff=i, yoff=j, win_xsize=1, win_ysize=1)[0][0]

            n1 = (c2 * np.conjugate(c1)).real
            d = (np.abs(c1 - c2)) ** 2
            n2 = (c1 * np.conj(c2)).real

            numerator1 += n1
            numerator2 += n2
            denominator += d
            count += 1

            print(i, j)

    snr1 = (2 * numerator1) / denominator
    snr2 = (2 * numerator2) / denominator
    snr = (snr1 + snr2) / 2
    sd_noise = (denominator / (2 * count)) ** 0.5
    print(snr1, snr2, sd_noise)
    return {"SNR": snr, "Sigma_noise": sd_noise}
