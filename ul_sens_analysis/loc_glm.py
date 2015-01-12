import os
import logging

import numpy as np

import fmri_tools.analysis
import fmri_tools.utils

import ul_sens_analysis.config
import ul_sens_fmri.config


def run(subj_id, acq_date):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    loc_glm_dir = os.path.join(subj_dir, conf.ana.loc_glm_dir)

    log_dir = os.path.join(subj_dir, "logs")

    log_path = os.path.join(
        log_dir,
        "{s:s}-loc_glm-log.txt".format(s=inf_str)
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmri_tools.utils.set_logger("screen")
    fmri_tools.utils.set_logger(log_path)

    os.chdir(loc_glm_dir)

    cond_details = {}

    # first, write the condition files
    # here, i_vf of 0 is upper, 1 is lower
    for (i_vf, vf) in enumerate(("upper", "lower")):

        # this is the file to write
        onset_path = "{s:s}-{v:s}_onsets.txt".format(s=inf_str, v=vf)

        cond_details[vf] = {
            "name": vf,
            "onsets_path": onset_path,
            "model": conf.ana.hrf_model
        }

        with open(onset_path, "w") as onset_file:

            for run_num in xrange(1, conf.exp.n_runs + 1):

                run_onsets = []

                # run seq is (pres loc, trial number, trial info)
                # where trial info is:
                #   0: time, in seconds, when it starts
                #   1: source location 1 for above, 2 for below, 0 for null
                #   2: image id
                #   3: whether it is in the 'pre' events
                #   4: been prepped
                run_seq = np.load(
                    os.path.join(
                        subj_dir,
                        "logs",
                        "{s:s}_ul_sens_fmri_run_{n:02d}_seq.npy".format(
                            s=subj_id, n=run_num
                        )
                    )
                )

                # pull out this visual field location - either upper or lower
                run_seq = run_seq[i_vf, ...]

                # axis 0 is now trials
                n_trials = run_seq.shape[0]

                for i_trial in xrange(n_trials):

                    # this check is for the image id, that it is greater than 0
                    # (in a floating point safe way)
                    # image IDs of 0 are null trials
                    if run_seq[i_trial, 2] > 0.5:
                        # append the onset time to the run onsets for this
                        # visual field location (we don't care about the source
                        # at the moment)
                        run_onsets.append(run_seq[i_trial, 0])

                    # just check that this is indeed a null trial
                    else:
                        assert run_seq[i_trial, 1] == 0

                # still inside the run loop - format the onsets as integers
                # (they will always be multiples of 4, which is the bin length)
                run_str = ["{n:.0f}".format(n=n) for n in run_onsets]

                # each run is a row in the onsets file, so write it out and
                # then move on to the next run
                onset_file.write(" ".join(run_str) + "\n")

    # now for the actual GLM, where we want to loop over hemispheres
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

        # run the above and below GLMs separately
        for vf in ("upper", "lower"):

            # output GLM
            glm_filename = "{s:s}-loc_{v:s}-glm-{h:s}_nf.niml.dset".format(
                s=inf_str, h=hemi, v=vf
            )

            # output betas
            beta_filename = "{s:s}-loc_{v:s}-beta-{h:s}_nf.niml.dset".format(
                s=inf_str, h=hemi, v=vf
            )

            fmri_tools.analysis.glm(
                run_paths=run_paths,
                output_dir=loc_glm_dir,
                glm_filename=glm_filename,
                beta_filename=beta_filename,
                tr_s=conf.ana.tr_s,
                cond_details=[cond_details[vf]],
                contrast_details=[],  # no contrasts necessary
                censor_str=conf.ana.censor_str
            )
