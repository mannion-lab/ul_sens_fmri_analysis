
import os

import fmri_tools

import ul_sens_fmri.config


class ConfigContainer(object):
    pass


def get_conf():

    exp_conf = ul_sens_fmri.config.get_conf()

    conf = ConfigContainer()

    conf.tr_s = 2.0

    conf.st_vols_to_ignore = 16

    conf.slice_timing_path = os.path.join(
        "/sci/study/ul_sens/code/",
        "ul_sens_analysis/ul_sens_analysis",
        "ul_sens_slice_timing_pattern.1D"
    )

    conf.base_dir = "/sci/study/ul_sens"
    conf.base_subj_dir = os.path.join(conf.base_dir, "subj_data")

    conf.loc_glm_dir = "loc_analysis"

    conf.n_to_censor = int(
        exp_conf.exp.n_pre_trials *
        exp_conf.exp.trial_len_s /
        conf.tr_s
    ) - 1

    conf.censor_str = "*:0-{n:d}".format(n=conf.n_to_censor)

    conf.hrf_model = "SPMG1({n:.0f})".format(n=exp_conf.exp.stim_on_s)

    conf.glm_dir = "analysis"

    conf.roi_names = ["V1", "V2", "V3"]
    conf.roi_numbers = [
        str(fmri_tools.utils.get_visual_area_lut()[roi_name])
        for roi_name in conf.roi_names
    ]
    conf.roi_dir = "/sci/anat/db_ver1/"

    conf.loc_glm_brick = "2"
    conf.loc_glm_thresh = "2.58"

    return conf
