
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
        "{s:s}-glm-log.txt".format(s=inf_str)
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmri_tools.utils.set_logger("screen")
    fmri_tools.utils.set_logger(log_path)

    os.chdir(glm_dir)

    cond_details = _prep_conds(subj_id, acq_date, conf)

    _run_glm(subj_id, acq_date, conf, cond_details)


def _run_glm(subj_id, acq_date, conf, cond_details):

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    glm_dir = os.path.join(subj_dir, "analysis")

    os.chdir(glm_dir)

    for (i_vf, vf) in enumerate(("above", "below")):

        run_paths = [
            os.path.join(
                subj_dir,
                "func",
                "run_{n:02d}".format(n=run_num),
                "{s:s}-run_{n:02d}-uw-{vf:s}_data.niml.dset".format(
                    s=inf_str, n=run_num, vf=vf
                )
            )
            for run_num in range(1, conf.exp.n_runs + 1)
        ]


        glm_filename = "{s:s}-{v:s}-glm-.niml.dset".format(
            s=inf_str, v=vf
        )

        beta_filename = "{s:s}-{v:s}-beta-.niml.dset".format(
            s=inf_str, v=vf
        )

        fmri_tools.analysis.glm(
            run_paths=run_paths,
            output_dir=glm_dir,
            glm_filename=glm_filename,
            beta_filename=beta_filename,
            tr_s=conf.ana.tr_s,
            cond_details=cond_details[vf],
            contrast_details=[],
            censor_str=conf.ana.censor_str,
            matrix_filename="exp_design_" + vf
        )

        data_filename = "{s:s}-{v:s}-data-amp.txt".format(
            s=inf_str, v=vf
        )

        if os.path.exists(data_filename):
            os.remove(data_filename)

        cmd = [
            "3dmaskdump",
            "-noijk",
            "-o", data_filename,
            glm_filename + "[1..$(2)]"
        ]

        runcmd.run_cmd(" ".join(cmd))

        b_filename = "{s:s}-{v:s}-beta-amp.txt".format(
            s=inf_str, v=vf
        )

        if os.path.exists(b_filename):
            os.remove(b_filename)

        cmd = [
            "3dmaskdump",
            "-noijk",
            "-o", b_filename,
            beta_filename
        ]

        runcmd.run_cmd(" ".join(cmd))


def _prep_conds(subj_id, acq_date, conf):

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    details = {}

    for (i_vf, vf) in enumerate(("above", "below")):

        cond_details = [{}, {}]

        for (i_source, source) in enumerate(("upper", "lower")):

            cond_details[i_source]["name"] = source

            onsets_path = "{s:s}-{v:s}_{sl:s}_onsets.txt".format(
                s=inf_str, v=vf, sl=source
            )

            cond_details[i_source]["onsets_path"] = onsets_path

            cond_details[i_source]["model"] = conf.ana.hrf_model

            with open(onsets_path, "w") as onsets_file:

                for run_num in xrange(1, conf.exp.n_runs + 1):

                    run_onsets = []

                    run_seq = np.load(
                        os.path.join(
                            subj_dir,
                            "logs",
                            "{s:s}_ul_sens_fmri_run_{n:02d}_seq.npy".format(
                                s=subj_id, n=run_num
                            )
                        )
                    )

                    run_seq = run_seq[i_vf, ...]

                    n_trials = run_seq.shape[0]

                    for i_trial in xrange(n_trials):

                        trial_ok = (run_seq[i_trial, 1] == (i_source + 1))

                        if trial_ok:
                            run_onsets.append(run_seq[i_trial, 0])

                    run_str = ["{n:.0f}".format(n=n) for n in run_onsets]

                    onsets_file.write(" ".join(run_str) + "\n")

        details[vf] = cond_details

    return details


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

            run_path = "{s:s}-run_{n:02d}-uw-data.txt".format(
                s=inf_str, n=run_num
            )

            np.savetxt(run_path, hemi_data)

            run_nodes_path = "{s:s}-run_{n:02d}-uw-nodes.txt".format(
                s=inf_str, n=run_num
            )

            np.savetxt(run_nodes_path, map(int, conf.ana.roi_numbers))

            run_path_niml = "{s:s}-run_{n:02d}-uw-data.niml.dset".format(
                s=inf_str, n=run_num
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
