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
    ana_dir = os.path.join(subj_dir, "analysis")
    post_dir = os.path.join(subj_dir, conf.ana.post_dir)
    task_dir = os.path.join(post_dir, "task")

    if not os.path.isdir(task_dir):
        os.mkdir(task_dir)

    os.chdir(task_dir)

    bin_res = 0.2

    bin_left_edges = np.arange(0.0, conf.exp.run_len_s - 32.0, bin_res)
    n_bins = len(bin_left_edges)

    responses = np.zeros((conf.exp.n_runs, n_bins))

    for i_run in xrange(conf.exp.n_runs):

        resp_path = os.path.join(
            log_dir,
            "{s:s}_ul_sens_fmri_run_{n:02d}_task.npy".format(
                s=subj_id, n=i_run + 1
            )
        )

        run_resp = np.load(resp_path)

        for (_, resp_time) in run_resp:

            if resp_time <= 32.0:
                continue

            i_bin = np.where(resp_time > bin_left_edges)[0][-1]

            responses[i_run, i_bin] = 1

    tasks_flat = np.zeros((conf.exp.n_runs, n_bins))

    for i_run in xrange(conf.exp.n_runs):

        task_path = os.path.join(
            log_dir,
            "{s:s}_ul_sens_fmri_run_{n:02d}_task_lut.npy".format(
                s=subj_id, n=i_run + 1
            )
        )

        # (time_s, digit, polarity, target)
        run_task = np.load(task_path)

        for i_task in xrange(run_task.shape[0]):

            if subj_id == "p1004" and i_run < 5:

                if (
                    (
                        run_task[i_task, 1] == 9 and
                        run_task[i_task, 2] == 1
                    ) or
                    (
                        run_task[i_task, 1] == 5 and
                        run_task[i_task, 2] == -1
                    )
                ):
                    is_target = 1
                else:
                    is_target = 0

            else:
                is_target = run_task[i_task, -1]

            onset = run_task[i_task, 0]

            if onset <= 32.0:
                continue

            if bool(is_target):

                i_bin = np.where(onset >= bin_left_edges)[0][-1]

                tasks_flat[i_run, i_bin] = 1

    tasks = np.zeros(
        (
            conf.exp.n_runs,
            2,  # pres
            2,  # src,
            n_bins
        )
    )

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
                log_dir,
                "{s:s}_ul_sens_fmri_run_{n:02d}_seq.npy".format(
                    s=subj_id, n=i_run + 1
                )
            )
        )

        for i_bin in xrange(n_bins):

            if tasks_flat[i_run, i_bin] == 0:
                continue

            # have a task, need to allocate it to a condition
            time_diff = run_seq[..., 0] - (bin_left_edges[i_bin] + 32.0)

            trial_ok = np.logical_and(
                time_diff > 0,
                time_diff < (conf.exp.trial_len_s / 2.0)
            )

            if np.sum(trial_ok) == 0:
                continue

            if np.sum(trial_ok) > 1:
                raise ValueError()

            (task_vf, i_task) = np.where(trial_ok)

            task_src = int(run_seq[task_vf, i_task, 1] - 1)

            tasks[i_run, task_vf, task_src, i_bin] = 1

    n_perf_bins = 10

    perf = np.empty((2, 2, n_perf_bins))
    perf.fill(np.NAN)

    for i_pres in xrange(2):
        for i_src in xrange(2):
            for i_roll in xrange(n_perf_bins):

                r = np.corrcoef(
                    tasks[:, i_pres, i_src, :].flat,
                    np.roll(responses, -i_roll, axis=1).flat
                )[0, 1]

                perf[i_pres, i_src, i_roll] = r

    # out
    perf_path = "{s:s}--perf-.npy".format(
        s=inf_str
    )

    np.save(perf_path, perf)
