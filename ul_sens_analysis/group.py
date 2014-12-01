import os

import numpy as np
import scipy.stats

import fmri_tools.analysis
import fmri_tools.utils

import runcmd

import ul_sens_analysis.config
import ul_sens_fmri.config


def resp_amps(conf, subj_info=None):

    if subj_info is None:
        subj_info = conf.subj_info

    data = np.empty(
        (
            len(subj_info),
            len(conf.roi_names),
            2,  # pres_loc: above, below
            2   # src_loc: upper, lower
        )
    )
    data.fill(np.NAN)

    for (i_subj, (subj_id, acq_date)) in enumerate(subj_info):

        for (i_vf, vf) in enumerate(["above", "below"]):

            data[i_subj, :, i_vf, :] = np.loadtxt(
                os.path.join(
                    conf.base_subj_dir,
                    subj_id,
                    "analysis",
                    (
                        subj_id + "_ul_sens_" + acq_date + "-" +
                        vf + "-data-amp.txt"
                    )
                )
            )

    assert np.sum(np.isnan(data)) == 0

    subj_mean = np.mean(
        np.mean(
            np.mean(data, axis=-1),
            axis=-1
        ),
        axis=-1
    )

    grand_mean = np.mean(data)

    norm_data = (
        data - subj_mean[:, np.newaxis, np.newaxis, np.newaxis]
    ) + grand_mean

    return (data, norm_data)


def rdms():

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    rdms = np.empty(
        (
            len(conf.ana.subj_info),
            len(conf.ana.roi_names),
            2,  # (above, below)
            2   # (same, different)
        )
    )
    rdms.fill(np.NAN)

    for (i_subj, (subj_id, acq_date)) in enumerate(conf.ana.subj_info):

        subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

        # roi x run type x vf x img x img
        subj_rdms = np.load(
            os.path.join(
                subj_dir,
                "rsa",
                "{s:s}-rsa-rdm-.npy".format(s=subj_id + "_ul_sens_" + acq_date)
            )
        )

        for i_roi in xrange(len(conf.ana.roi_names)):

            roi_data = subj_rdms[i_roi, ...]

            above_same = 1 - scipy.stats.spearmanr(
                roi_data[0, 0, ...].flat,  # odd, above
                roi_data[1, 0, ...].flat   # even, above
            )[0]

            below_same = 1 - scipy.stats.spearmanr(
                roi_data[0, 1, ...].flat,  # odd, below
                roi_data[1, 1, ...].flat   # even, below
            )[0]

            above_diff = 1 - scipy.stats.spearmanr(
                roi_data[0, 0, ...].flat,  # odd, above
                roi_data[1, 1, ...].flat   # even, below
            )[0]

            below_diff = 1 - scipy.stats.spearmanr(
                roi_data[0, 1, ...].flat,  # odd, below
                roi_data[1, 0, ...].flat   # even, above
            )[0]

            rdms[i_subj, i_roi, 0, 0] = above_same
            rdms[i_subj, i_roi, 0, 1] = above_diff
            rdms[i_subj, i_roi, 1, 0] = below_same
            rdms[i_subj, i_roi, 1, 1] = below_diff

    assert np.sum(np.isnan(rdms)) == 0

    return rdms
