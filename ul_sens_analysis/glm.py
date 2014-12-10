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

    _run_glm(subj_id, acq_date, conf, log_dir)


def _run_glm(subj_id, acq_date, conf, log_dir):

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    glm_dir = os.path.join(subj_dir, "analysis")

    os.chdir(glm_dir)

    for vf in ("above", "below"):

        cond_details = _write_onsets(
            subj_id=subj_id,
            acq_date=acq_date,
            conf=conf,
            vf=vf,
            runs_type="all",
            log_dir=log_dir
        )

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
            cond_details=cond_details,
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

        beta_bricks = "[40..$]"

        # check the beta bricks are as expected
        dset_labels = fmri_tools.utils.get_dset_label(
            beta_filename + beta_bricks
        )

        desired_labels = []

        for src_loc in ["upper", "lower"]:
            for img_id in conf.exp.img_ids:
                desired_labels.append(
                    vf + "_" + src_loc + "_" + str(img_id) + "#0"
                )

        assert dset_labels == desired_labels

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


def _write_onsets(subj_id, acq_date, conf, vf, runs_type, log_dir):
    """Write the onsets for a particular visual field location and runs type.

    Parameters
    ----------
    subj_id: string
        Subject ID
    acq_date: string
        Date subject was scanned, in YYYYMMDD
    conf: ConfigContainer
        ul_sens_fmri config
    vf: string, {"above", "below"}
        Presentation location
    runs_type: string, {"odd", "even", "all"}
        Whether to include the odd, even, or all runs
    log_dir: string
        Location of the subject logfiles

    Returns
    -------
    details: list of dicts
        Each item is a dictionary for a particular image ID and source location
        combination. It includes keys for the HRF model, the condition name,
        and the onsets filename.

    """

    if vf == "above":
        i_vf = 0
    elif vf == "below":
        i_vf = 1
    else:
        raise ValueError("Unknown vf")

    if runs_type == "odd":
        run_num_range = range(1, conf.exp.n_runs + 1, 2)
    elif runs_type == "even":
        run_num_range = range(2, conf.exp.n_runs + 1, 2)
    elif runs_type == "all":
        run_num_range = range(1, conf.exp.n_runs + 1)
    else:
        raise ValueError("Unknown runs_type")

    run_seqs = [
        np.load(
            os.path.join(
                log_dir,
                subj_id + "_ul_sens_fmri_run_{n:02d}_seq.npy".format(
                    s=subj_id,
                    n=run_num
                )
            )
        )
        for run_num in run_num_range
    ]

    inf_str = subj_id + "_ul_sens_" + acq_date

    details = []

    # image ID
    for img_id in conf.exp.img_ids:

        # source location
        for (i_sl, sl) in enumerate(("upper", "lower")):

            # this is the file to write
            onset_path = "{s:s}-{v:s}_{sl:s}_{i:d}_onsets.txt".format(
                s=inf_str,
                v=vf,
                sl=sl,
                i=img_id
            )

            # save the details for this condition
            details.append(
                {
                    "name": "{v:s}_{sl:s}_{i:d}".format(
                        v=vf,
                        sl=sl,
                        i=img_id
                    ),
                    "onsets_path": onset_path,
                    "model": conf.ana.hrf_model
                }
            )

            with open(onset_path, "w") as onset_file:

                for run_seq in run_seqs:

                    run_onsets = []

                    # run seq is (pres loc, trial number, trial info)
                    # where trial info is:
                    #   0: time, in seconds, when it starts
                    #   1: source location 1 for upper, 2 for lower, 0 for null
                    #   2: image id
                    #   3: whether it is in the 'pre' events
                    #   4: been prepped

                    # pull out this visual field location - either above or
                    # below
                    curr_run_seq = run_seq[i_vf, ...]

                    # axis 0 is now trials
                    n_trials = curr_run_seq.shape[0]

                    for i_trial in xrange(n_trials):

                        # a valid trial if it has the image id that we're after
                        trial_ok = np.logical_and(
                            int(curr_run_seq[i_trial, 2]) == img_id,
                            int(curr_run_seq[i_trial, 1]) == (i_sl + 1)
                        )

                        if trial_ok:
                            run_onsets.append(curr_run_seq[i_trial, 0])

                    # still inside the run loop - format the onsets as integers
                    # (they will always be multiples of 4, which is the bin
                    # length)
                    run_str = ["{n:.0f}".format(n=n) for n in run_onsets]

                    # each run is a row in the onsets file, so write it out and
                    # then move on to the next run
                    onset_file.write(" ".join(run_str) + "\n")

    return details
