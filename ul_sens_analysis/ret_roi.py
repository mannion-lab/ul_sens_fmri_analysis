import os
import logging

import numpy as np

import fmri_tools.utils

import ul_sens_analysis.config
import ul_sens_analysis.glm
import ul_sens_fmri.config
import ul_sens_analysis.glm_prep
import runcmd


def run(subj_id, acq_date):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    log_dir = os.path.join(subj_dir, "logs")
    log_path = os.path.join(
        log_dir,
        "{s:s}-post-log.txt".format(s=inf_str)
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmri_tools.utils.set_logger("screen")
    fmri_tools.utils.set_logger(log_path)

    ana_dir = os.path.join(subj_dir, "analysis")

    post_dir = os.path.join(subj_dir, conf.ana.post_dir)
    ret_roi_dir = os.path.join(post_dir, "ret_roi")

    if not os.path.isdir(post_dir):
        os.mkdir(post_dir)

    if not os.path.isdir(ret_roi_dir):
        os.mkdir(ret_roi_dir)

    os.chdir(ret_roi_dir)

    # phase ranges for the different ret roi specs
    phases = {
        "lh": {
            "upper": [0, 90],
            "lower": [270, 360]
        },
        "rh": {
            "upper": [90, 180],
            "lower": [180, 270]
        }
    }

    mask_paths = {}

    # first, calculate the masks based on the ret phases
    for hemi in ["lh", "rh"]:

        if subj_id == "p1003":
            vis_loc = "vis_loc_ver1"
        else:
            vis_loc = "vis_loc"

        wedge_path = os.path.join(
            "/sci/vis_loc/db_ver1",
            subj_id,
            conf.ana.subj_wedge_sess[subj_id],
            "dt/wedge",
            "{s:s}_{vl:s}_{a:s}-wedge-angle-{h:s}_nf.niml.dset[0]".format(
                s=subj_id, a=conf.ana.subj_wedge_sess[subj_id], h=hemi,
                vl=vis_loc
            )
        )

        # subject's ROI definitions for this hemisphere
        roi_path = os.path.join(
            conf.ana.roi_dir,
            subj_id,
            "rois",
            "{s:s}_vis_loc_--rois-{h:s}_nf.niml.dset".format(
                s=subj_id, h=hemi
            )
        )

        for pres in ["upper", "lower"]:

            ret_roi_path = "{s:s}-{v:s}-ret_roi_-{h:s}_nf.niml.dset".format(
                s=inf_str, v=pres, h=hemi
            )

            # we want the roi file to be 'amongst' the identifiers for V1-V3
            roi_test = "(amongst(b," + ",".join(conf.ana.roi_numbers) + ")*b)"

            ret_roi_test = "within(a,{l:d}, {u:d})".format(
                l=phases[hemi][pres][0], u=phases[hemi][pres][1]
            )

            cmd = [
                "3dcalc",
                "-a", wedge_path,
                "-b", roi_path,
                "-expr", "'" + ret_roi_test + "*" + roi_test + "'",
                "-prefix", ret_roi_path,
                "-overwrite"
            ]

            runcmd.run_cmd(" ".join(cmd))

            mask_paths[(pres, hemi)] = ret_roi_path

    ul_sens_analysis.glm_prep._extract_data(
        subj_id,
        acq_date,
        conf,
        mask_paths,
        loc_mask=False
    )

    ul_sens_analysis.glm._run_glm(
        subj_id,
        acq_date,
        conf,
        log_dir,
        loc_mask=False
    )
