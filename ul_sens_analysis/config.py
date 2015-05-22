
import os

import fmri_tools.utils

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
    conf.base_group_dir = os.path.join(conf.base_dir, "group_data")
    conf.base_post_group_dir = os.path.join(conf.base_dir, "post_group_data")
    conf.base_fig_dir = os.path.join(conf.base_dir, "figures")

    conf.loc_glm_dir = "loc_analysis"
    conf.post_dir = "post_analysis"

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

    conf.source_colours = [
        [241 / 255.0, 163 / 255.0, 64 / 255.0],  # above
        [153 / 255.0, 142 / 255.0, 195 / 255.0]  # below
    ]

    conf.n_mvpa_steps = 100

    conf.subj_info = [
        ["p1001", "20140714"],
        ["p1003", "20140922"],
        ["p1006", "20140728"],
        ["p1008", "20140714"],
        ["p1004", "20140929"],
        ["p1010", "20141103"],
        ["p1011", "20141020"]
    ]

    conf.subj_wedge_sess = {
        "p1001": "20082009",
        "p1003": "20110413",
        "p1006": "20090901",
        "p1008": "20140602",
        "p1004": "20140616a",
        "p1010": "20140616a",
        "p1011": "20140922a"
    }

    return conf
