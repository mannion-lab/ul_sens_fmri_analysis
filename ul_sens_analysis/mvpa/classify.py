
import numpy as np

import sklearn.svm

import ul_sens_fmri.config
import ul_sens_analysis.config

import ul_sens_analysis.mvpa.data


def run(subj_id, acq_date, subj_data=None):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    cm = np.zeros(
        (
            len(conf.ana.roi_names),
            2,  # pres loc (upper, lower),
            conf.exp.n_img,
            conf.exp.n_src_locs,  # (above, below)
            2  # (above, below) predicted
        )
    )

    for (i_vf, vf) in enumerate(("upper", "lower")):

        # get the data for this VF
        subj_data = ul_sens_analysis.mvpa.data.get_mvpa_data(
            subj_id,
            acq_date,
            vf
        )

        for (i_roi, roi_name) in enumerate(conf.ana.roi_names):

            beta_data = subj_data[0][roi_name]
            loc_data = subj_data[1][roi_name]

            # beta_data needs to be z-scored
            # combine the images and source locations together so we can get a
            # mean and std for each run
            temp_beta = np.concatenate(
                (
                    beta_data[:, 0, ...],
                    beta_data[:, 1, ...]
                )
            )
            run_mean = np.mean(temp_beta, axis=0)
            run_std = np.std(temp_beta, axis=0)

            # do the z-scoring
            beta_data = (
                (beta_data - run_mean[np.newaxis, np.newaxis, ...]) /
                run_std[np.newaxis, np.newaxis, ...]
            )

            node_k = len(loc_data)

            for i_img in xrange(conf.exp.n_img):

                # pull out the data where we are at now; that is, source
                # location (above, below) x runs (all)
                curr_beta = beta_data[i_img, ...]

                for i_test_run in xrange(conf.exp.n_runs):

                    # exclude the current 'test' run
                    i_train_runs = np.setdiff1d(
                        range(conf.exp.n_runs),
                        [i_test_run]
                    )

                    train_data = np.empty(
                        (
                            len(i_train_runs) * conf.exp.n_src_locs,
                            node_k
                        )
                    )
                    train_data.fill(np.NAN)

                    train_labels = np.empty(train_data.shape[0])
                    train_labels.fill(np.NAN)

                    i_flat = 0
                    for i_train_run in i_train_runs:

                        for (i_sl, sl_label) in enumerate([0, 1]):

                            train_data[i_flat, :] = curr_beta[
                                i_sl,
                                i_train_run,
                                :
                            ]

                            train_labels[i_flat] = sl_label

                            i_flat += 1

                    svm = sklearn.svm.SVC(kernel="linear")

                    svm.fit(train_data, train_labels)

                    curr_pred = svm.predict(curr_beta[:, i_test_run, :])

                    for (true_val, pred_val) in zip([0, 1], curr_pred):

                        cm[i_roi, i_vf, i_img, true_val, pred_val] += 1

    return cm


