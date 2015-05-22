import os

import numpy as np

import fmri_tools.analysis
import fmri_tools.utils

import ul_sens_analysis.config
import ul_sens_fmri.config
import runcmd


def get_all_mvpa_beta_data():
    """This is useful for double-checking with the group PSC data"""

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    n_subj = len(conf.ana.subj_info)

    beta_data = np.empty(
        (
            n_subj,
            conf.exp.n_img,
            2,  # pres loc
            conf.exp.n_src_locs
        )
    )
    beta_data.fill(np.NAN)

    for (i_subj, subj_info) in enumerate(conf.ana.subj_info):

        for (i_vf, vf) in enumerate(("upper", "lower")):

            subj_data = get_mvpa_data(
                subj_id=subj_info[0],
                acq_date=subj_info[1],
                vf=vf
            )[0]["V1"]

            # average over nodes
            subj_data = np.mean(subj_data, axis=-1)
            # and runs
            subj_data = np.mean(subj_data, axis=-1)

            beta_data[i_subj, :, i_vf, :] = subj_data

    return beta_data


def get_mvpa_data(subj_id, acq_date, vf):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    glm_dir = os.path.join(
        subj_dir,
        conf.ana.post_dir,
        "mvpa_glm"
    )

    beta = {}
    loc_t = {}

    raw_beta = []
    raw_loc_t = []

    for hemi in ("lh", "rh"):

        # the localiser mask
        mask_path = os.path.join(
            subj_dir,
            conf.ana.loc_glm_dir,
            "{s:s}-loc_{vf:s}-mask-{h:s}_nf.niml.dset".format(
                s=inf_str, h=hemi, vf=vf
            )
        )

        # GLM betas
        beta_filename = os.path.join(
            glm_dir,
            "{s:s}-{v:s}-mvpa_beta-{h:s}_nf.niml.dset".format(
                s=inf_str, v=vf, h=hemi
            )
        )

        # let's start with the betas
        cmd = [
            "3dmaskdump",
            "-noijk",
            "-mask", mask_path,
            mask_path,  # this holds the ROI indices, too
            beta_filename
        ]

        cmd_out = runcmd.run_cmd(" ".join(cmd), log_stdout=False)
        raw_out = cmd_out.std_out.splitlines()

        # keep track of the number of nodes - they should be the same for the
        # localiser t extraction. We'll check to make sure.
        n_beta_nodes = len(raw_out)

        # just extend the list, for now - we'll parse it later on
        raw_beta.extend(raw_out)

        # the localiser T
        loc_t_path = os.path.join(
            subj_dir,
            conf.ana.loc_glm_dir,
            "{s:s}-loc_{vf:s}-glm-{h:s}_nf.niml.dset".format(
                s=inf_str, h=hemi, vf=vf
            )
        )

        # now for the localiser T values
        cmd = [
            "3dmaskdump",
            "-noijk",
            "-mask", mask_path,
            mask_path,
            loc_t_path
        ]

        cmd_out = runcmd.run_cmd(" ".join(cmd), log_stdout=False)
        raw_out = cmd_out.std_out.splitlines()

        # keep track of the number of nodes - they should be the same for the
        # localiser t extraction. We'll check to make sure.
        n_loc_t_nodes = len(raw_out)

        # just extend the list, for now - we'll parse it later on
        raw_loc_t.extend(raw_out)

        assert n_loc_t_nodes == n_beta_nodes

    # convert to numpy arrays; n_nodes x dumped vals
    beta_data = np.array(
        [
            map(float, raw_beta_node.split(" "))
            for raw_beta_node in raw_beta
        ]
    )

    loc_t_data = np.array(
        [
            map(float, raw_loc_t_node.split(" "))
            for raw_loc_t_node in raw_loc_t
        ]
    )

    for (roi_num, roi_name) in zip(conf.ana.roi_numbers, conf.ana.roi_names):

        # find the nodes in the ROI
        in_roi = (beta_data[:, 0].astype("int") == int(roi_num))

        # check that the localiser agrees
        assert np.all(
            in_roi == (loc_t_data[:, 0].astype("int") == int(roi_num))
        )

        n_roi_nodes = np.sum(in_roi)

        roi_beta_data = np.empty(
            (
                conf.exp.n_img,
                conf.exp.n_src_locs,
                conf.exp.n_runs,
                n_roi_nodes
            )
        )
        roi_beta_data.fill(np.NAN)

        # need to farm out the beta data appropriately

        # we can use beta_filename because the hemisphere doesn't matter
        dset_labels = fmri_tools.utils.get_dset_label(beta_filename)

        # the -1 is because we also dumped the ROI index
        assert len(dset_labels) == (beta_data.shape[1] - 1)

        for (i_col, dset_label) in enumerate(dset_labels):

            # if it's one of the noise regressors, move along
            if dset_label[:3] == "Run":

                assert ("Pol" in dset_label)
                continue

            dset_params = dset_label.split("#")[0].split("_")

            (curr_vf, curr_sl, curr_id, curr_run) = dset_params

            # make sure we're looking at the correct file
            assert curr_vf == vf

            if curr_sl == "above":
                i_sl = 0
            elif curr_sl == "below":
                i_sl = 1
            else:
                raise ValueError()

            i_id = list(conf.exp.img_ids).index(int(curr_id))

            i_run = int(curr_run) - 1

            # the +1 is because of the first index being the ROI
            roi_beta_data[i_id, i_sl, i_run, :] = beta_data[in_roi, i_col + 1]

        # check that we've filled up the array
        assert np.sum(np.isnan(roi_beta_data)) == 0

        # now for the localiser - more straightforward
        roi_loc_t_data = loc_t_data[in_roi, -1]

        assert roi_loc_t_data.shape[0] == roi_beta_data.shape[-1]

        beta[roi_name] = roi_beta_data
        loc_t[roi_name] = roi_loc_t_data

    return (beta, loc_t)
