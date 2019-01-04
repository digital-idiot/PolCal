import src.core.covariance as cov
import src.core.channel_correlation as correl
import src.core.crosstalk.crosstalk_estimation as crstlk_est
import src.core.radiometry.radiometric_estimation as radio_est
import src.core.radiometry.radiometric_correction as rcal
import src.core.crosstalk.crosstalk_correction as crstlk_cor
import testing.dump as td

# data_dir = "/media/abhisek/Store/Storage/UAVSAR_PBand_ENVI/"
data_dir = "/media/abhisek/Store/Storage/UAVSAR_LBand/L-Band_2_Uncalibrated/"

# hh_f = data_dir + "HH"
hh_f = data_dir + "hh.slc"
# hv_f = data_dir + "HV"
hv_f = data_dir + "hv.slc"
# vh_f = data_dir + "VH"
vh_f = data_dir + "vh.slc"
# vv_f = data_dir + "VV"
vv_f = data_dir + "vv.slc"
# inc_f = data_dir + "INC_ANGLE"
inc_f = data_dir + "Inc_Angle"
cr_f = data_dir + "CR_Info.csv"
# out_f1 = data_dir + "Out/copol_cal.csv"
out_f1 = data_dir + "Processed/Out/copol_cal.csv"
# out_f2 = data_dir + "Out/xpol_cal.csv"
out_f2 = data_dir + "Processed/Out/xpol_cal.csv"
abs_f = data_dir + "Processed/Out/abs_coeffs.csv"
php_f = data_dir + "Processed/Out/php_coeffs.csv"
abs_out = data_dir + "Processed/Out/A"
php_img = data_dir + "Processed/Out/Phi_Plus"

inc_image = data_dir + "Processed/Subset/Incidence_Angle_Radian"

cov_out = data_dir + "Processed/Out/COVARIANCE/"

hh_out = data_dir + "Processed/Out/Radio_Cal/HH"
hv_out = data_dir + "Processed/Out/Radio_Cal/HV"
vh_out = data_dir + "Processed/Out/Radio_Cal/VH"
vv_out = data_dir + "Processed/Out/Radio_Cal/VV"

logic_out = data_dir + "Processed/Out/Logic_Board/Logic_Board"
ainsworth_quality = data_dir + "Processed/Out/Logic_Board/Data_Quality"
crstlk_dir_q = data_dir + "Processed/Out/Crosstalk_Q/"
crstlk_dir_a = data_dir + "Processed/Out/Crosstalk_A/"


crosscal_out_q = data_dir + "Processed/Out/Cal_Out_Q/"
crosscal_out_a = data_dir + "Processed/Out/Cal_Out_A/"

err_img = data_dir + "Processed/Out/Error/Error_Map"

range_bin_out_q = data_dir + "Processed/Out/Range_Corrected_Q/"
range_bin_out_a = data_dir + "Processed/Out/Range_Corrected_A/"

# radio_est.radio_copol(
#     hh_img=hh_f,
#     vv_img=vv_f,
#     inc_img=inc_f,
#     wave_length=0.2379,
#     cr_info=cr_f,
#     out_file=out_f1
# )

# radio_est.radio_xpol(
#     hv_img="/media/abhisek/Store/Storage/UAVSAR_LBand/L-Band_2_Uncalibrated/Processed/Xpol_Subset/HV_X",
#     vh_img="/media/abhisek/Store/Storage/UAVSAR_LBand/L-Band_2_Uncalibrated/Processed/Xpol_Subset/VH_X",
#     outfile=out_f2
# )

# radio_est.fit_radio_params(out_f1, abs_file=abs_f, php_file=php_f, a_deg=1, phi_deg=2)

# radio_est.write_copol_params(
#     abs_coeffs=abs_f,
#     abs_img_out=abs_out,
#     inc_img=inc_image,
#     phi_coeffs=php_f,
#     php_img_out=php_img,
# )

root_sub = data_dir + "Processed/Subset/"
hh_sub = root_sub + "HH"
hv_sub = root_sub + "HV"
vh_sub = root_sub + "VH"
vv_sub = root_sub + "VV"
a_cap_img = data_dir + "Processed/Out/A"

# rcal.radiometric_cal(
#     hh_img=hh_sub,
#     hv_img=hv_sub,
#     vh_img=vh_sub,
#     vv_img=vv_sub,
#     php_img=php_img,
#     copol_info=out_f1,
#     xpol_info=out_f2,
#     hh_out=hh_out,
#     hv_out=hv_out,
#     vh_out=vh_out,
#     vv_out=vv_out,
#     abs_img=a_cap_img
# )


# cov.compute_covar(
#     hh_img=hh_out,
#     hv_img=hv_out,
#     vh_img=vh_out,
#     vv_img=vv_out,
#     out_dir=cov_out,
#     winxsize=3,
#     winysize=3
# )

# correl.logicboard(cov_dir=cov_out, out_img=logic_out, threshold=0.3)

# crstlk_est.quegan(cov_dir=cov_out, crosstalk_dir=crstlk_dir_q, logic_img=logic_out)
#
# crstlk_est.range_binning(crstlk_dir=crstlk_dir_q, out_dir=range_bin_out_q, range_direction=1)

# crstlk_cor.apply_correction(
#     hh_img=hh_out,
#     hv_img=hv_out,
#     vh_img=vh_out,
#     vv_img=vv_out,
#     params_dir=range_bin_out,
#     out_dir=crosscal_out_q,
#     error_map=err_img
# )

# td.dump_data()

crstlk_est.ainsworth(cov_dir=cov_out, crosstalk_dir=crstlk_dir_a)

# crstlk_est.ainsworth_mod(cov_dir=cov_out, crosstalk_dir=crstlk_dir_a, data_quality_img=ainsworth_quality)
crstlk_est.range_binning(crstlk_dir=crstlk_dir_a, out_dir=range_bin_out_a, range_direction=1)

crstlk_cor.apply_correction(
    hh_img=hh_out,
    hv_img=hv_out,
    vh_img=vh_out,
    vv_img=vv_out,
    params_dir=range_bin_out_a,
    out_dir=crosscal_out_a,
    error_map=err_img
)
