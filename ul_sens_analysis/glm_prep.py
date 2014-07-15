
import os
import itertools
import logging

import numpy as np

import fmri_tools.analysis
import fmri_tools.utils

import ul_sens_analysis.config
import ul_sens_fmri.config
import runcmd


def run(subj_id, acq_date):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    glm_dir = os.path.join(subj_dir, conf.ana.glm_dir)

    log_dir = os.path.join(subj_dir, "logs")
    log_path = os.path.join(
        log_dir,
        "{s:s}-glm_prep-log.txt".format(s=inf_str)
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmri_tools.utils.set_logger("screen")
    fmri_tools.utils.set_logger(log_path)

    os.chdir(glm_dir)

    mask_paths = _loc_to_mask(subj_id, acq_date, conf)

    _extract_data(subj_id, acq_date, conf, mask_paths)


def _loc_to_mask(subj_id, acq_date, conf):

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    loc_glm_dir = os.path.join(subj_dir, conf.ana.loc_glm_dir)

    os.chdir(loc_glm_dir)

    inf_str = subj_id + "_ul_sens_" + acq_date

    mask_paths = {}

    for (vf, hemi) in itertools.product(("above", "below"), ("lh", "rh")):

        loc_t_path = "{s:s}-loc_{v:s}-glm-{h:s}_nf.niml.dset".format(
            s=inf_str, v=vf, h=hemi
        ) + "[" + conf.ana.loc_glm_brick + "]"

        roi_path = os.path.join(
            conf.ana.roi_dir,
            subj_id,
            "rois",
            "{s:s}_vis_loc_--rois-{h:s}_nf.niml.dset".format(
                s=subj_id, h=hemi
            )
        )

        mask_path = "{s:s}-loc_{v:s}-mask-{h:s}_nf.niml.dset".format(
            s=inf_str, v=vf, h=hemi
        )

        roi_test = "amongst(a," + ",".join(conf.ana.roi_numbers) + ")"

        loc_test = "step(b-" + conf.ana.loc_glm_thresh + ")"

        expr = "'a*and(" + roi_test + "," + loc_test + ")'"

        cmd = [
            "3dcalc",
            "-overwrite",
            "-a", roi_path,
            "-b", loc_t_path,
            "-expr", expr,
            "-prefix", mask_path
        ]

        runcmd.run_cmd(" ".join(cmd))

        mask_paths[(vf, hemi)] = mask_path

    return mask_paths


def _extract_data(subj_id, acq_date, conf, mask_paths):

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    mask_dir = os.path.join(subj_dir, "loc_analysis")

    analysis_dir = os.path.join(subj_dir, "analysis")

    data = np.empty(
        (
            len(conf.ana.roi_names),
            conf.exp.n_runs,
            int(conf.exp.run_len_s / conf.ana.tr_s),
            2
        )
    )
    data.fill(np.NAN)

    for (i_vf, vf) in enumerate(("above", "below")):

        for run_num in range(1, conf.exp.n_runs + 1):

            run_dir = os.path.join(
                subj_dir,
                "func",
                "run_{n:02d}".format(n=run_num)
            )

            os.chdir(run_dir)

            hemi_data = np.empty((data.shape[0], data.shape[2], 2))
            hemi_data.fill(np.NAN)

            for (i_hemi, hemi) in enumerate(("lh", "rh")):

                run_path = "{s:s}-run_{n:02d}-uw-{h:s}_nf.niml.dset".format(
                    s=inf_str, n=run_num, h=hemi
                )

                cmd = [
                    "3dROIstats",
                    "-mask", os.path.join(mask_dir, mask_paths[(vf, hemi)]),
                    "-1Dformat",
                    run_path
                ]

                cmd_out = runcmd.run_cmd(" ".join(cmd), log_stdout=False)

                run_data = cmd_out.std_out.splitlines()[3::2]

                assert len(run_data) == data.shape[2]

                for (i_vol, vol_data) in enumerate(run_data):

                    vol_data = vol_data.split("\t")[-data.shape[0]:]

                    hemi_data[:, i_vol, i_hemi] = vol_data

            assert np.sum(np.isnan(hemi_data)) == 0

            # average over hemispheres
            hemi_data = np.mean(hemi_data, axis=-1)

            run_path = "{s:s}-run_{n:02d}-uw-{vf:s}_data.txt".format(
                s=inf_str, n=run_num, vf=vf
            )

            np.savetxt(run_path, hemi_data)

            run_nodes_path = "{s:s}-run_{n:02d}-uw-nodes.txt".format(
                s=inf_str, n=run_num
            )

            np.savetxt(run_nodes_path, map(int, conf.ana.roi_numbers))

            run_path_niml = "{s:s}-run_{n:02d}-uw-{vf:s}_data.niml.dset".format(
                s=inf_str, n=run_num, vf=vf
            )

            cmd = [
                "ConvertDset",
                "-i_1D",
                "-input", run_path,
                "-node_index_1D", run_nodes_path,
                "-o_niml",
                "-prefix", run_path_niml,
                "-overwrite"
            ]

            runcmd.run_cmd(" ".join(cmd))

            data[:, run_num - 1, :, i_vf] = hemi_data

    assert np.sum(np.isnan(data)) == 0

    os.chdir(analysis_dir)

    data_path = "{s:s}-data.npy".format(s=inf_str)

    np.save(data_path, data)
