import os

import numpy as np
import scipy.stats

import fmri_tools.analysis
import fmri_tools.utils

import runcmd

import ul_sens_analysis.config
import ul_sens_fmri.config


def resp_amps(conf=None, subj_info=None):

    if conf is None:
        conf = ul_sens_fmri.config.get_conf()
        conf.ana = ul_sens_analysis.config.get_conf()

    if subj_info is None:
        subj_info = conf.ana.subj_info

    data = np.empty(
        (
            len(subj_info),  # subjects
            len(conf.ana.roi_names),  # ROIs
            conf.exp.n_img,  # images
            2,  # pres_loc: above, below
            2   # src_loc: upper, lower
        )
    )
    data.fill(np.NAN)

    for (i_subj, (subj_id, acq_date)) in enumerate(subj_info):

        for (i_vf, vf) in enumerate(["above", "below"]):

            # this will be ROIs x (n_img x 2)
            # the columns are (upper, lower, upper, lower, ...)
            subj_data = np.loadtxt(
                os.path.join(
                    conf.ana.base_subj_dir,
                    subj_id,
                    "analysis",
                    (
                        subj_id + "_ul_sens_" + acq_date + "-" +
                        vf + "-data-amp.txt"
                    )
                )
            )

            # this is the upper source images; even indices
            data[i_subj, :, :, i_vf, 0] = subj_data[:, 0::2]
            # this is the lower source images; odd indices
            data[i_subj, :, :, i_vf, 1] = subj_data[:, 1::2]

    # check we've filled up 'data' correctly
    assert np.sum(np.isnan(data)) == 0

    np.save(
        file=os.path.join(
            conf.ana.base_group_dir,
            "ul_sens_group_amp_data.npy"
        ),
        arr=data
    )

    save_resp_amps_for_spss(
        data=data,
        txt_path=os.path.join(
            conf.ana.base_group_dir,
            "ul_sens_group_amp_data_spss.txt"
        )
    )


def difference_term():

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    # this is subjects, rois, images, pres_loc, src_loc
    data = np.load(
        os.path.join(
            conf.ana.base_group_dir,
            "ul_sens_group_amp_data.npy"
        )
    )

    # this will hold the difference term
    diff = np.empty(data.shape[:4])
    diff.fill(np.NAN)

    for i_subj in xrange(data.shape[0]):
        for i_roi in xrange(data.shape[1]):
            for i_img in xrange(data.shape[2]):
                for i_src_loc in xrange(data.shape[-1]):

                    resp = data[i_subj, i_roi, i_img, :, i_src_loc]

                    # above - below
                    curr_diff = resp[0] - resp[1]

                    diff[i_subj, i_roi, i_img, i_src_loc] = curr_diff

    assert np.sum(np.isnan(diff)) == 0

    diff = np.mean(diff, axis=0)

    # roi, fragment, (id, src_loc, diff_term)
    rdiff = np.empty((data.shape[1], data.shape[2] * data.shape[4], 3))
    rdiff.fill(np.NAN)

    for i_roi in xrange(rdiff.shape[0]):

        i = 0

        for i_img in xrange(diff.shape[1]):
            for i_src_loc in xrange(diff.shape[2]):

                rdiff[i_roi, i, 0] = i_img
                rdiff[i_roi, i, 1] = i_src_loc
                rdiff[i_roi, i, 2] = diff[i_roi, i_img, i_src_loc]

                i += 1

    for i_roi in xrange(rdiff.shape[0]):

        i_sort = np.argsort(rdiff[i_roi, :, -1])

        rdiff[i_roi, ...] = rdiff[i_roi, i_sort, ...]

    np.save(
        file=os.path.join(
            conf.ana.base_group_dir,
            "ul_sens_group_amp_diffs_sorted.npy"
        ),
        arr=rdiff
    )


def save_resp_amps_for_spss(data, txt_path):

    # average over images
    data = np.mean(data, axis=2)

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
