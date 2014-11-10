import os
import logging

import numpy as np

import fmri_tools.analysis
import fmri_tools.utils

import runcmd

import ul_sens_analysis.config
import ul_sens_fmri.config


def run(subj_id, acq_date):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    rsa_dir = os.path.join(subj_dir, conf.ana.rsa_dir)

    log_dir = os.path.join(subj_dir, "logs")

    log_path = os.path.join(
        log_dir,
        "{s:s}-rsa-log.txt".format(s=inf_str)
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmri_tools.utils.set_logger("screen")
    fmri_tools.utils.set_logger(log_path)

    os.chdir(rsa_dir)

    details = {}

    # first, write the condition files
    # here, i_vf of 0 is above, 1 is below
    for (i_vf, vf) in enumerate(("above", "below")):

        vf_details = []

        # iterate through the source locations
        for (i_sl, sl) in enumerate(("upper", "lower")):

            # iterate through the image IDs
            for (i_img, img_id) in enumerate(conf.exp.img_ids):

                # this is the file to write
                onset_path = "{s:s}-{v:s}_{sl:s}_{i:d}_onsets.txt".format(
                    s=inf_str,
                    v=vf,
                    sl=sl,
                    i=img_id
                )

                # save the details for this condition
                cond_details = {
                    "name": "{v:s}_{sl:s}_{i:d}".format(
                        v=vf,
                        sl=sl,
                        i=img_id
                    ),
                    "onsets_path": onset_path,
                    "model": conf.ana.hrf_model
                }

                # and append to our accumulator
                vf_details.append(cond_details)

                # now to write out the associated onsets
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

                        # pull out this visual field location - either above or below
                        run_seq = run_seq[i_vf, ...]

                        # axis 0 is now trials
                        n_trials = run_seq.shape[0]

                        for i_trial in xrange(n_trials):

                            # a valid trial if it has the image id that we're
                            # after
                            trial_ok = np.logical_and(
                                int(run_seq[i_trial, 2]) == img_id,
                                int(run_seq[i_trial, 1]) == (i_sl + 1)
                            )

                            if trial_ok:
                                run_onsets.append(run_seq[i_trial, 0])


                        # still inside the run loop - format the onsets as integers
                        # (they will always be multiples of 4, which is the bin length)
                        run_str = ["{n:.0f}".format(n=n) for n in run_onsets]

                        # each run is a row in the onsets file, so write it out and
                        # then move on to the next run
                        onset_file.write(" ".join(run_str) + "\n")

        details[vf] = vf_details

    data = {}

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
        for vf in ("above", "below"):

            # output GLM
            glm_filename = "{s:s}-rsa_{v:s}-glm-{h:s}_nf.niml.dset".format(
                s=inf_str, h=hemi, v=vf
            )

            # output betas
            beta_filename = "{s:s}-rsa_{v:s}-beta-{h:s}_nf.niml.dset".format(
                s=inf_str, h=hemi, v=vf
            )

            # might as well use the mask to make it run quicker
            mask_path = os.path.join(
                conf.ana.base_subj_dir,
                subj_id,
                conf.ana.loc_glm_dir,
                "{s:s}-loc_{v:s}-mask-{h:s}_nf.niml.dset".format(
                    s=inf_str, v=vf, h=hemi
                )
            )

            extra_args = ["-mask", mask_path]

            fmri_tools.analysis.glm(
                run_paths=run_paths,
                output_dir=rsa_dir,
                glm_filename=glm_filename,
                beta_filename=beta_filename,
                tr_s=conf.ana.tr_s,
                cond_details=details[vf],
                contrast_details=[],  # no contrasts necessary
                censor_str=conf.ana.censor_str,
                extra_args=extra_args,
                extra_reml_args=extra_args
            )

            data_filename = "{s:s}-rsa_{v:s}-data-{h:s}.txt".format(
                s=inf_str, h=hemi, v=vf
            )

            if os.path.exists(data_filename):
                os.remove(data_filename)

            # want the t-value for each image
            cmd = [
                "3dmaskdump",
                "-noijk",
                "-mask", mask_path,
                "-o", data_filename,
                mask_path,
                glm_filename + "[2..$(2)]"
            ]

            runcmd.run_cmd(" ".join(cmd))

            vf_data = np.loadtxt(data_filename)

            data[(hemi, vf)] = vf_data

    # ok, now to do the correlations
    n_corr = conf.exp.n_img * 2
    n_rois = len(conf.ana.roi_numbers)
    corr_mat = np.empty((2, n_rois, n_corr, n_corr))
    corr_mat.fill(np.NAN)

    for (i_vf, vf) in enumerate(("above", "below")):

        # combine over hemispheres
        vf_data = np.vstack((data[("lh", vf)], data[("rh", vf)]))

        # plus one is for the ROI id
        assert vf_data.shape[1] == n_corr + 1

        for (i_roi, roi_id) in enumerate(conf.ana.roi_numbers):

            roi_data = vf_data[vf_data[:, 0].astype("int") == int(roi_id), 1:]

            for i_a in xrange(n_corr):
                for i_b in xrange(n_corr):

                    corr_mat[i_vf, i_roi, i_a, i_b] = np.corrcoef(
                        roi_data[:, i_a], roi_data[:, i_b]
                    )[1, 0]

    rdm = 1 - corr_mat

    rdm_filename = "{s:s}-rsa-rdm-.npy".format(s=inf_str)

    np.save(rdm_filename, rdm)
