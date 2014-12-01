import os
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

    for vf in ("above", "below"):

        # these files have three nodes, one for each visual area
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

        # to write
        glm_filename = "{s:s}-{v:s}-glm-.niml.dset".format(
            s=inf_str, v=vf
        )

        # to write
        beta_filename = "{s:s}-{v:s}-beta-.niml.dset".format(
            s=inf_str, v=vf
        )

        # run the GLM on this visual field location
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

        # now to convert the beta weights to percent signal change

        # baseline timecourse
        bltc_filename = "{s:s}-{v:s}-bltc-.niml.dset".format(
            s=inf_str, v=vf
        )

        # baseline
        bl_filename = "{s:s}-{v:s}-bltc-.niml.dset".format(
            s=inf_str, v=vf
        )

        # psc
        psc_filename = "{s:s}-{v:s}-psc-.niml.dset".format(
            s=inf_str, v=vf
        )

        beta_bricks = "[40,41]"

        # check the beta bricks are as expected
        assert (
            fmri_tools.utils.get_dset_label(beta_filename + beta_bricks) ==
            [vf + "_upper#0", vf + "_lower#0"]
        )

        # run the PSC conversion
        fmri_tools.utils.beta_to_psc(
            beta_path=beta_filename,
            beta_bricks=beta_bricks,
            design_path="exp_design_" + vf + ".xmat.1D",
            bltc_path=bltc_filename,
            bl_path=bl_filename,
            psc_path=psc_filename,
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
            psc_filename
        ]

        runcmd.run_cmd(" ".join(cmd))

        # save the betas as text file also, for exploration / checking
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

    analysis_dir = os.path.join(subj_dir, "analysis")

    details = {}

    for (i_vf, vf) in enumerate(("above", "below")):

        cond_details = []

        # now we're also interested in the image source location
        for (i_source, source) in enumerate(("upper", "lower")):

            onsets_path = os.path.join(
                analysis_dir,
                "{s:s}-{v:s}_{sl:s}_onsets.txt".format(
                    s=inf_str, v=vf, sl=source
                )
            )

            cond_details.append(
                {
                    "name": vf + "_" + source,
                    "onsets_path": onsets_path,
                    "model": conf.ana.hrf_model
                }
            )

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

                        # this checks that the source location (index 1)
                        # matches the current source location under
                        # consideration. The latter is 0-based, so need to
                        # increment by 1
                        trial_ok = (run_seq[i_trial, 1] == (i_source + 1))

                        if trial_ok:

                            # since the trial is 'ok', then it should have a
                            # valid image ID. Let's check
                            assert run_seq[i_trial, 2] > 0.5

                            # append the onset time
                            run_onsets.append(run_seq[i_trial, 0])

                    run_str = ["{n:.0f}".format(n=n) for n in run_onsets]

                    onsets_file.write(" ".join(run_str) + "\n")

        details[vf] = cond_details

    return details


