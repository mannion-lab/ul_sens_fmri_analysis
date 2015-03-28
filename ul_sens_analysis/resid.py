import os
import logging

import numpy as np

import fmri_tools.utils

import ul_sens_analysis.config
import ul_sens_analysis.glm
import ul_sens_fmri.config
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
    resid_dir = os.path.join(post_dir, "resid")

    if not os.path.isdir(post_dir):
        os.mkdir(post_dir)

    if not os.path.isdir(resid_dir):
        os.mkdir(resid_dir)

    os.chdir(resid_dir)

    n_window = 12

    traces = np.zeros(
        (
            3,  # ROI
            2,  # vf
            2,  # src
            n_window  # time
        )
    )

    for (i_vf, vf) in enumerate(["upper", "lower"]):

        # in - residuals from the GLM analysis
        resid_path = os.path.join(
            ana_dir,
            "{s:s}-{v:s}-resid-.niml.dset".format(
                s=inf_str, v=vf
            )
        )

        # get the residuals into an array by dumping from the dataset
        cmd = [
            "3dmaskdump",
            "-noijk",
            resid_path
        ]
        cmd_out = runcmd.run_cmd(" ".join(cmd), log_stdout=False)

        # this converts the output string to ROIs x time
        resid_flat = np.array(
            [
                map(float, roi_resid.split(" "))
                for roi_resid in cmd_out.std_out.splitlines()
            ]
        )

        # want to get the baseline values to normalise the residuals
        bl_path = os.path.join(
            ana_dir,
            "{s:s}-{v:s}-bltc-.niml.dset".format(
                s=inf_str, v=vf
            )
        )

        cmd = [
            "3dmaskdump",
            "-noijk",
            bl_path
        ]
        cmd_out = runcmd.run_cmd(" ".join(cmd), log_stdout=False)

        # get a baseline value
        bl = np.array(map(float, cmd_out.std_out.splitlines()))

        # convert the residuals to to PSC units
        resid_flat = 100 * (resid_flat / bl)

        # want to split it into runs rather than one flat timecourse
        # want to exclude the initial censored volumes
        vols_per_run = (
            int(resid_flat.shape[1] / conf.exp.n_runs) - (conf.ana.n_to_censor + 1)
        )
        vols_per_run_total = int(resid_flat.shape[1]) / conf.exp.n_runs

        resid = np.empty(
            (
                resid_flat.shape[0],  # rois
                conf.exp.n_runs,
                vols_per_run
            )
        )
        resid.fill(np.NAN)

        for i_run in xrange(conf.exp.n_runs):

            i_start = i_run * vols_per_run_total + conf.ana.n_to_censor + 1
            i_end = i_start + vols_per_run

            resid[:, i_run, :] = resid_flat[:, i_start:i_end]

        # convert to squared error
        resid = resid ** 2

        for i_run in xrange(conf.exp.n_runs):

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
                        s=subj_id, n=i_run + 1
                    )
                )
            )

            # pull out this visual field location - either upper or lower
            run_seq = run_seq[i_vf, ...]

            # axis 0 is now trials
            n_trials = run_seq.shape[0]

            # keep a track of how many trials we go through, just to check
            # everything is hunky dory
            trial_count = 0

            for i_trial in xrange(n_trials):

                # check if its a 'trial' that we're interested in
                trial_ok = np.all(
                    [
                        run_seq[i_trial, 3] == 0,  # not a pre event
                        run_seq[i_trial, 2] > 0.5,  # an image was shown
                        run_seq[i_trial, 1] > 0  # not a null event
                    ]
                )

                if not trial_ok:
                    continue

                onset_s = run_seq[i_trial, 0]
                onset_vol = int(onset_s / conf.ana.tr_s)
                onset_vol -= conf.ana.st_vols_to_ignore

                # trial type is 1-based
                trial_type = run_seq[i_trial, 1] - 1

                # move the residual timecourse so the first index aligns with
                # the trial onset
                shifted_resid = np.roll(
                    resid[:, i_run, :],
                    -onset_vol,
                    axis=1
                )

                traces[:, i_vf, trial_type, :] += shifted_resid[:, :n_window]

                trial_count += 1

            assert trial_count == 60

            # convert to an average
            traces[:, i_vf, ...] /= (30.0 * conf.exp.n_runs)

    # out
    traces_path = "{s:s}--traces-.npy".format(
        s=inf_str
    )

    np.save(traces_path, traces)
