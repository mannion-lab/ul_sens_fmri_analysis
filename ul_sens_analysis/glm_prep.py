
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

    # convert the localiser GLM to masks in V1, V2, and V3
    mask_paths = _loc_to_mask(subj_id, acq_date, conf)

    _extract_data(subj_id, acq_date, conf, mask_paths)


def _loc_to_mask(subj_id, acq_date, conf):

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    loc_glm_dir = os.path.join(subj_dir, conf.ana.loc_glm_dir)

    os.chdir(loc_glm_dir)

    inf_str = subj_id + "_ul_sens_" + acq_date

    mask_paths = {}

    # go through combinations of visual field position and hemisphere
    for (vf, hemi) in itertools.product(("above", "below"), ("lh", "rh")):

        # this is the localiser GLM subbrick with the t-statistic for this
        # visual field location
        loc_t_path = "{s:s}-loc_{v:s}-glm-{h:s}_nf.niml.dset".format(
            s=inf_str, v=vf, h=hemi
        ) + "[" + conf.ana.loc_glm_brick + "]"

        # check it is correct
        assert fmri_tools.utils.get_dset_label(loc_t_path)[0] == vf + "#0_Tstat"

        # subject's ROI definitions for this hemisphere
        roi_path = os.path.join(
            conf.ana.roi_dir,
            subj_id,
            "rois",
            "{s:s}_vis_loc_--rois-{h:s}_nf.niml.dset".format(
                s=subj_id, h=hemi
            )
        )

        # this is the mask file to write
        mask_path = "{s:s}-loc_{v:s}-mask-{h:s}_nf.niml.dset".format(
            s=inf_str, v=vf, h=hemi
        )

        # we want the roi file to be 'amongst' the identifiers for V1-V3
        roi_test = "amongst(a," + ",".join(conf.ana.roi_numbers) + ")"

        # we also want the t-value to be above a certain threshold
        loc_test = "step(b-" + conf.ana.loc_glm_thresh + ")"

        # so it is an 'and' operation, and we want it to be labelled with the
        # ROI identified value so we multiply it by the outcome
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

        # store the mask path to make it easier to access in the next step
        mask_paths[(vf, hemi)] = mask_path

    return mask_paths


def _extract_data(subj_id, acq_date, conf, mask_paths):

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    mask_dir = os.path.join(subj_dir, "loc_analysis")

    analysis_dir = os.path.join(subj_dir, "analysis")

    n_rois = len(conf.ana.roi_names)
    n_vols_per_run = int(conf.exp.run_len_s / conf.ana.tr_s)

    # initialise our data container; rois x runs x volumes x vf
    data = np.empty(
        (
            n_rois,
            conf.exp.n_runs,
            n_vols_per_run,
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

            # note that the last index here is hemisphere
            hemi_data = np.empty((n_rois, n_vols_per_run, 2))
            hemi_data.fill(np.NAN)

            for (i_hemi, hemi) in enumerate(("lh", "rh")):

                run_path = "{s:s}-run_{n:02d}-uw-{h:s}_nf.niml.dset".format(
                    s=inf_str, n=run_num, h=hemi
                )

                # average across all nodes in each ROI and dump the timecourse
                # to standard out
                cmd = [
                    "3dROIstats",
                    "-mask", os.path.join(mask_dir, mask_paths[(vf, hemi)]),
                    "-1Dformat",
                    run_path
                ]

                # ... which we don't want to log!
                cmd_out = runcmd.run_cmd(" ".join(cmd), log_stdout=False)

                # check the header for correctness
                roi_header = cmd_out.std_out.splitlines()[1].split("\t")[-3:]

                # make sure that the columns are what I think they are
                for (roi_head, roi_index) in zip(
                    roi_header,
                    conf.ana.roi_numbers
                ):
                    assert roi_head.strip() == "Mean_" + roi_index

                # we want to clip out the header and the info lines
                run_data = cmd_out.std_out.splitlines()[3::2]

                # check we've done this correctly
                assert len(run_data) == n_vols_per_run

                for (i_vol, vol_data) in enumerate(run_data):

                    # so this is just one line of data, tab-separated
                    # we want to pull out our three ROIs, which will be the
                    # last in the file
                    vol_data = vol_data.split("\t")[-n_rois:]

                    # store, for each of the ROIs
                    hemi_data[:, i_vol, i_hemi] = vol_data

            # check that we've filled up the array as expected
            assert np.sum(np.isnan(hemi_data)) == 0

            # average over hemispheres
            hemi_data = np.mean(hemi_data, axis=-1)

            run_path = "{s:s}-run_{n:02d}-uw-{vf:s}_data.txt".format(
                s=inf_str, n=run_num, vf=vf
            )

            # save it out as a text file for this run; rois x vols
            np.savetxt(run_path, hemi_data)

            # we also want to save what 'nodes' in this data corresponds to; ie
            # ROI identifiers
            run_nodes_path = "{s:s}-run_{n:02d}-uw-nodes.txt".format(
                s=inf_str, n=run_num
            )

            np.savetxt(run_nodes_path, map(int, conf.ana.roi_numbers), "%d")

            # now we want to make it into an AFNI dataset so we can run the GLM
            # using their software
            run_path_niml = "{s:s}-run_{n:02d}-uw-{v:s}_data.niml.dset".format(
                s=inf_str, n=run_num, v=vf
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

    # we save the data here so we can access it independent of AFNI
    np.save(data_path, data)
