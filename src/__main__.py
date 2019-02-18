import src.core.covariance as cov
import src.core.channel_correlation as correl
import src.core.crosstalk.crosstalk_estimation as crstlk_est
import src.core.radiometry.radiometric_estimation as radio_est
import src.core.radiometry.radiometric_correction as rcal
import src.core.crosstalk.crosstalk_correction as crstlk_cor
import src.error.MNE_estimation as me
import src.error.SNR_estimation as snrc
import testing.dump as td
import src.POA.poa_shift as poa


root_dir = "/media/Cellar/Thesis_Output/"
main_input = root_dir + "/Subset_Main/"
xpol_input = root_dir + "Subset_Xpol/"

radio_dir = root_dir + "Radio_Cal/"

cov_dir_b = root_dir + "Covariance.Before/"

crstlk_q_b = root_dir + "Crosstalk_Q.Before/"
crstlk_q_a = root_dir + "Crosstalk_Q.After/"
crstlk_a_b = root_dir + "Crosstalk_A.Before/"
crstlk_a_a = root_dir + "Crosstalk_A.After/"
crstlk_m_b = root_dir + "Crosstalk_M.Before/"
crstlk_m_a = root_dir + "Crosstalk_M.After/"

logic_out_qb = crstlk_q_b + "Logic_Board"
logic_out_qa = crstlk_q_a + "Logic_Board"

crstlk_qb_rc = root_dir + "Crosstalk_Q_RC.Before/"
crstlk_qa_rc = root_dir + "Crosstalk_Q_RC.After/"
crstlk_ab_rc = root_dir + "Crosstalk_A_RC.Before/"
crstlk_aa_rc = root_dir + "Crosstalk_A_RC.After/"
crstlk_mb_rc = root_dir + "Crosstalk_M_RC.Before/"
crstlk_ma_rc = root_dir + "Crosstalk_M_RC.After/"

out_q = root_dir + "CalOut_Q/"
out_a = root_dir + "CalOut_A/"
out_m = root_dir + "CalOut_M/"

err_q = out_q + "Error"
err_a = out_a + "Error"
err_m = out_m + "Error"

hh_radio = radio_dir + "HH"
hv_radio = radio_dir + "HV"
vh_radio = radio_dir + "VH"
vv_radio = radio_dir + "VV"

cov_dir_q = root_dir + "Covariance_Q/"
cov_dir_a = root_dir + "Covariance_A/"

hh_qcal = out_q + "HH_Cal"
hv_qcal = out_q + "HV_Cal"
vh_qcal = out_q + "VH_Cal"
vv_qcal = out_q + "VV_Cal"

hh_acal = out_a + "HH_Cal"
hv_acal = out_a + "HV_Cal"
vh_acal = out_a + "VH_Cal"
vv_acal = out_a + "VV_Cal"

hh_mcal = out_m + "HH_Cal"
hv_mcal = out_m + "HV_Cal"
vh_mcal = out_m + "VH_Cal"
vv_mcal = out_m + "VV_Cal"

u_qb = crstlk_qb_rc + "u"
v_qb = crstlk_qb_rc + "v"
w_qb = crstlk_qb_rc + "w"
z_qb = crstlk_qb_rc + "z"

u_qa = crstlk_qa_rc + "u"
v_qa = crstlk_qa_rc + "v"
w_qa = crstlk_qa_rc + "w"
z_qa = crstlk_qa_rc + "z"

u_ab = crstlk_ab_rc + "u"
v_ab = crstlk_ab_rc + "v"
w_ab = crstlk_ab_rc + "w"
z_ab = crstlk_ab_rc + "z"

u_aa = crstlk_aa_rc + "u"
v_aa = crstlk_aa_rc + "v"
w_aa = crstlk_aa_rc + "w"
z_aa = crstlk_aa_rc + "z"

u_mb = crstlk_mb_rc + "u"
v_mb = crstlk_mb_rc + "v"
w_mb = crstlk_mb_rc + "w"
z_mb = crstlk_mb_rc + "z"

u_ma = crstlk_ma_rc + "u"
v_ma = crstlk_ma_rc + "v"
w_ma = crstlk_ma_rc + "w"
z_ma = crstlk_ma_rc + "z"

# correl.logicboard(cov_dir=cov_dir_b, out_img=logic_out_qb, threshold=0.3)
# crstlk_est.quegan(cov_dir=cov_dir_b, crosstalk_dir=crstlk_q_b, logic_img=logic_out_qb)
# crstlk_est.range_binning(crstlk_dir=crstlk_q_b, out_dir=crstlk_qb_rc, range_direction=1)

# crstlk_cor.apply_correction(
#     hh_img=hh_radio,
#     hv_img=hv_radio,
#     vh_img=vh_radio,
#     vv_img=vv_radio,
#     params_dir=crstlk_qb_rc,
#     out_dir=out_q,
#     error_map=err_q
# )

# crstlk_est.ainsworth(cov_dir=cov_dir_b, crosstalk_dir=crstlk_a_b)
# crstlk_est.range_binning(crstlk_dir=crstlk_a_b, out_dir=crstlk_ab_rc, range_direction=1)

# crstlk_cor.apply_correction(
#     hh_img=hh_radio,
#     hv_img=hv_radio,
#     vh_img=vh_radio,
#     vv_img=vv_radio,
#     params_dir=crstlk_ab_rc,
#     out_dir=out_a,
#     error_map=err_a
# )

# cov.compute_covar(
#     hh_img=hh_qcal,
#     hv_img=hv_qcal,
#     vh_img=vh_qcal,
#     vv_img=vv_qcal,
#     out_dir=cov_dir_q,
#     winxsize=3,
#     winysize=3
# )

# correl.logicboard(cov_dir=cov_dir_q, out_img=logic_out_qa, threshold=0.3)
# crstlk_est.quegan(cov_dir=cov_dir_q, crosstalk_dir=crstlk_q_a, logic_img=logic_out_qa)
# crstlk_est.range_binning(crstlk_dir=crstlk_q_a, out_dir=crstlk_qa_rc, range_direction=1)

# cov.compute_covar(
#     hh_img=hh_acal,
#     hv_img=hv_acal,
#     vh_img=vh_acal,
#     vv_img=vv_acal,
#     out_dir=cov_dir_a,
#     winxsize=3,
#     winysize=3
# )

# crstlk_est.ainsworth(cov_dir=cov_dir_a, crosstalk_dir=crstlk_a_a)
# crstlk_est.range_binning(crstlk_dir=crstlk_a_a, out_dir=crstlk_aa_rc, range_direction=1)

# me.estimate_mne(
#     u_img=u_qb,
#     v_img=v_qb,
#     w_img=w_qb,
#     z_img=z_qb,
#     mne_img=crstlk_qb_rc + "MNE"
# )

# me.estimate_mne(
#     u_img=u_ab,
#     v_img=v_ab,
#     w_img=w_ab,
#     z_img=z_ab,
#     mne_img=crstlk_ab_rc + "MNE"
# )

# me.estimate_mne(
#     u_img=u_qa,
#     v_img=v_qa,
#     w_img=w_qa,
#     z_img=z_qa,
#     mne_img=crstlk_qa_rc + "MNE"
# )

# me.estimate_mne(
#     u_img=u_aa,
#     v_img=v_aa,
#     w_img=w_aa,
#     z_img=z_aa,
#     mne_img=crstlk_aa_rc + "MNE"
# )

# crstlk_est.ainsworth_mod(cov_dir=cov_dir_b, crosstalk_dir=crstlk_m_b, xoffset=1000, yoffset=1000)
# crstlk_est.range_binning(crstlk_dir=crstlk_m_b, out_dir=crstlk_mb_rc, range_direction=1)
#
# crstlk_cor.apply_correction(
#     hh_img=hh_radio,
#     hv_img=hv_radio,
#     vh_img=vh_radio,
#     vv_img=vv_radio,
#     params_dir=crstlk_mb_rc,
#     out_dir=out_m,
#     error_map=err_m
# )

snrc.estimate_SNR(
    c1_img=vh_radio,
    c2_img=hh_radio,
)
