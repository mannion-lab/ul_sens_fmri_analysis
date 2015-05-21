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

    glm_dir = os.path.join(
        subj_dir,
        conf.ana.post_dir,
        "mvpa_glm"
    )

    if not os.path.exists(glm_dir):
        os.mkdir(glm_dir)

    log_dir = os.path.join(subj_dir, "logs")
    log_path = os.path.join(
        log_dir,
        "{s:s}-mvpa_glm-log.txt".format(s=inf_str)
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmri_tools.utils.set_logger("screen")
    fmri_tools.utils.set_logger(log_path)

    _run_glm(subj_id, acq_date, conf, log_dir, glm_dir)


def _run_glm(subj_id, acq_date, conf, log_dir, glm_dir):

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    cwd = os.getcwd()

    os.chdir(glm_dir)

    for vf in ("upper", "lower"):

        cond_details = _write_onsets(
            subj_id=subj_id,
            acq_date=acq_date,
            conf=conf,
            vf=vf,
            log_dir=log_dir
        )

        for hemi in ("lh", "rh"):

            run_paths = [
                os.path.join(
                    subj_dir,
                    "func",
                    "run_{n:02d}".format(n=run_num),
                    "{s:s}-run_{n:02d}-uw-{h:s}_nf.niml.dset".format(
                        s=inf_str, n=run_num, h=hemi
                    )
                )
                for run_num in range(1, conf.exp.n_runs + 1)
            ]

            # the localiser mask
            mask_path = os.path.join(
                subj_dir,
                conf.ana.loc_glm_dir,
                "{s:s}-loc_{vf:s}-mask-{h:s}_nf.niml.dset".format(
                    s=inf_str, h=hemi, vf=vf
                )
            )

            # to write
            glm_filename = "{s:s}-{v:s}-mvpa_glm-{h:s}_nf.niml.dset".format(
                s=inf_str, v=vf, h=hemi
            )

            # to write
            beta_filename = "{s:s}-{v:s}-mvpa_beta-{h:s}_nf.niml.dset".format(
                s=inf_str, v=vf, h=hemi
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
                matrix_filename="exp_design_" + vf + "_" + hemi,
                mask_filename=mask_path
            )

    os.chdir(cwd)


def _write_onsets(subj_id, acq_date, conf, vf, log_dir):

    if vf == "upper":
        i_vf = 0
    elif vf == "lower":
        i_vf = 1
    else:
        raise ValueError("Unknown vf")

    run_num_range = range(1, conf.exp.n_runs + 1)

    # load the run sequence info
    run_seqs = [
        np.load(
            os.path.join(
                log_dir,
                subj_id + "_ul_sens_fmri_run_{n:02d}_seq.npy".format(
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
        for (i_sl, sl) in enumerate(("above", "below")):

            # run number
            for (i_run, run_seq) in enumerate(run_seqs):

                # this is the file to write
                name = "{s:s}-{v:s}_{sl:s}_{i:d}_{r:d}".format(
                    s=inf_str,
                    v=vf,
                    sl=sl,
                    i=img_id,
                    r=i_run + 1
                )

                onset_path = name + "_onsets.txt"

                # save the details for this condition
                details.append(
                    {
                        "name": name,
                        "onsets_path": onset_path,
                        "model": conf.ana.hrf_model
                    }
                )

                with open(onset_path, "w") as onset_file:

                    # loop through runs again, so we can indicate that other
                    # runs aren't the one that we're interested in
                    for i_onsets_run in xrange(conf.exp.n_runs):

                        # if it isn't the current run, indicate this with an
                        # asterisk and move on
                        if i_run != i_onsets_run:
                            onset_file.write("*\n")
                            continue

                        # otherwise, need to work out the timings
                        run_onsets = []

                        # run seq is (pres loc, trial number, trial info)
                        # where trial info is:
                        #   0: time, in seconds, when it starts
                        #   1: source location 1 for above, 2 for below, 0 for
                        #   null
                        #   2: image id
                        #   3: whether it is in the 'pre' events
                        #   4: been prepped

                        # pull out this visual field location - either upper or
                        # lower
                        curr_run_seq = run_seq[i_vf, ...]

                        # axis 0 is now trials
                        n_trials = curr_run_seq.shape[0]

                        for i_trial in xrange(n_trials):

                            # a valid trial if it has the image id that we're
                            # after, and the source location is correct
                            trial_ok = np.logical_and(
                                int(curr_run_seq[i_trial, 2]) == img_id,
                                int(curr_run_seq[i_trial, 1]) == (i_sl + 1)
                            )

                            if trial_ok:
                                run_onsets.append(curr_run_seq[i_trial, 0])

                        # still inside the run loop - format the onsets as
                        # integers (they will always be multiples of 4, which
                        # is the bin length)
                        run_str = ["{n:.0f}".format(n=n) for n in run_onsets]

                        # each run is a row in the onsets file, so write it out
                        # and then move on to the next run
                        onset_file.write(" ".join(run_str) + "\n")

    return details
