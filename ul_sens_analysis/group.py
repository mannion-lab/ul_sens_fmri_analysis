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

    np.save(
        file=os.path.join(
            conf.base_group_dir,
            "ul_sens_group_amp_data.npy"
        ),
        arr=data
    )

    save_resp_amps_for_spss(
        data=data,
        txt_path=os.path.join(
            conf.base_group_dir,
            "ul_sens_group_amp_data_spss.txt"
        )
    )

    return (data, norm_data)


def save_resp_amps_for_spss(data, txt_path):

    (n_subj, n_rois, n_pres, n_src) = data.shape

    n_rows = n_subj * n_rois * n_pres * n_src

    with open(txt_path, "w") as txt_file:

        for i_subj in xrange(n_subj):

            txt_file.write(str(i_subj + 1) + "\t")

            for i_roi in xrange(n_rois):
                for i_pres in xrange(n_pres):
                    for i_src in xrange(n_src):

                        dv = data[i_subj, i_roi, i_pres, i_src]

                        txt_file.write(str(dv) + "\t")

            txt_file.write("\n")


def get_rdms():

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

    base_rdms = np.empty(
        (
            len(conf.ana.subj_info),
            len(conf.ana.roi_names),
            2,  # odd, even
            2,  # above, below
            60,  # images
            60
        )
    )
    base_rdms.fill(np.NAN)

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

        base_rdms[i_subj, ...] = subj_rdms

        for i_roi in xrange(len(conf.ana.roi_names)):

            # leaves (odd, even) x (above, below) x 60 x 60
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

    return (rdms, base_rdms)


def descriptives():

    conf = ul_sens_analysis.config.get_conf()

    # subjects X va X pres (A,B) X src (U, L)
    (_, amp_data) = ul_sens_analysis.group.resp_amps(conf)

    # for the interaction, average over rois
    x_data = np.mean(amp_data, axis=1)

    x_mean = np.mean(x_data, axis=0)
    x_se = np.std(x_data, axis=0, ddof=1) / np.sqrt(amp_data.shape[0])

    pres_locs = ("Above", "Below")
    src_locs = ("Upper", "Lower")

    print "Descriptives:"

    for (i_pres, pres_loc) in enumerate(pres_locs):

        (t, p) = scipy.stats.ttest_rel(
            x_data[:, i_pres, 0],
            x_data[:, i_pres, 1]
        )

        print t, p

        for (i_src, src_loc) in enumerate(src_locs):

            out_str = (
                "\t" + pres_loc + ", " + src_loc + "- " +
                "Mean: {n:.4f}".format(n=x_mean[i_pres, i_src]) +
                " SE: {n:.4f}".format(n=x_se[i_pres, i_src])
            )

            print out_str

    # for the main effect of presentation, average over rois and srcs
    pres_data = np.mean(np.mean(amp_data, axis=-1), axis=1)
    pres_mean = np.mean(pres_data, axis=0)
    pres_se = (
        np.std(pres_data, axis=0, ddof=1) /
        np.sqrt(amp_data.shape[0])
    )

    print "\n\tPres means: ", pres_mean
    print "\tPres SEs: ", pres_se

    print "\n"

    # for the interaction between ROI and src, average over pres locs
    rx_data = np.mean(amp_data, axis=-2)
    rx_mean = np.mean(rx_data, axis=0)
    rx_se = np.std(rx_data, axis=0, ddof=1) / np.sqrt(amp_data.shape[0])

    for (i_roi, roi) in enumerate(conf.roi_names):

        (t, p) = scipy.stats.ttest_rel(
            rx_data[:, i_roi, 0],
            rx_data[:, i_roi, 1]
        )

        print t, p

        for (i_src, src_loc) in enumerate(src_locs):

            out_str = (
                "\t" + roi + ", " + src_loc + "- " +
                "Mean: {n:.4f}".format(n=rx_mean[i_roi, i_src]) +
                " SE: {n:.4f}".format(n=rx_se[i_roi, i_src])
            )

            print out_str

def get_rdm_diff():

    conf = ul_sens_analysis.config.get_conf()

    rdms = get_rdms()[1]

    # average over odd/even
    rdms = np.mean(rdms, axis=2)

    # for V1, look at (above - below)
    rdm_diff = rdms[:, 0, 0, ...] - rdms[:, 0, 1, ...]

#    data = np.empty((rdm_diff.shape[0], 3))
#    data.fill(np.NAN)

    diff_mask = np.tril(np.ones((60, 60)), k=-1)

    data = []

    for i_subj in xrange(rdm_diff.shape[0]):

        subj_diff = rdm_diff[i_subj, ...]

        # mask out the upper triangle

        subj_data = [[] for _ in xrange(3)]

        for i_img_1 in xrange(60):
            for i_img_2 in xrange(60):

                if diff_mask[i_img_1, i_img_2] == 0:
                    continue

                if i_img_1 < 30:
                    if i_img_2 < 30:
                        i_data = 0
                    else:
                        i_data = 2
                else:
                    if i_img_2 < 30:
                        i_data = 2
                    else:
                        i_data = 1

                subj_data[i_data].append(
                    subj_diff[i_img_1, i_img_2]
                )

        subj_data = [np.mean(x_data) for x_data in subj_data]

        data.append(subj_data)

    return data




