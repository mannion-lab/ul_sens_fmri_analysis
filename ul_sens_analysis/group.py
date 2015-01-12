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
            2,  # pres_loc: upper, lower
            2   # src_loc: above, below
        )
    )
    data.fill(np.NAN)

    for (i_subj, (subj_id, acq_date)) in enumerate(subj_info):

        for (i_vf, vf) in enumerate(["upper", "lower"]):

            # this will be ROIs x (n_img x 2)
            # the columns are (above, below, above, below, ...)
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

            # this is the above source images; even indices
            data[i_subj, :, :, i_vf, 0] = subj_data[:, 0::2]
            # this is the below source images; odd indices
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


def save_resp_amps_for_spss(data, txt_path):

    # average over images
    data = np.mean(data, axis=2)

    (n_subj, n_rois, n_pres, n_src) = data.shape

    n_rows = n_subj * n_rois * n_pres * n_src

    header = []

    for i_roi in xrange(n_rois):
        for i_pres in xrange(n_pres):
            for i_src in xrange(n_src):

                header.append(
                    "r{r:d}_p{p:d}_s{s:d}".format(
                        r=i_roi + 1,
                        p=i_pres + 1,
                        s=i_src + 1
                    )
                )

    with open(txt_path, "w") as txt_file:

        txt_file.write("\t".join(header) + "\n")

        for i_subj in xrange(n_subj):

            for i_roi in xrange(n_rois):
                for i_pres in xrange(n_pres):
                    for i_src in xrange(n_src):

                        dv = data[i_subj, i_roi, i_pres, i_src]

                        txt_file.write("{dv:.18e}\t".format(dv=dv))

            txt_file.write("\n")


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

    # average over ROIs
    data = np.mean(data, axis=1)

    (n_subj, n_img, n_pres, n_src) = data.shape

    # this will hold the difference term
    diff = np.empty((n_subj, n_img, n_src))
    diff.fill(np.NAN)

    for i_subj in xrange(n_subj):
        for i_img in xrange(n_img):
            for i_src_loc in xrange(n_src):

                resp = data[i_subj, i_img, :, i_src_loc]

                # above - below
                curr_diff = resp[0] - resp[1]

                diff[i_subj, i_img, i_src_loc] = curr_diff

    assert np.sum(np.isnan(diff)) == 0

    diff = np.mean(diff, axis=0)

    # roi, fragment, (id, i_id, src_loc, diff_term)
    rdiff = np.empty((n_img * n_src, 4))
    rdiff.fill(np.NAN)

    i = 0

    for i_img in xrange(n_img):
        for i_src_loc in xrange(n_src):

            rdiff[i, 0] = i_img
            rdiff[i, 1] = conf.exp.img_ids[i_img]
            rdiff[i, 2] = i_src_loc
            rdiff[i, 3] = diff[i_img, i_src_loc]

            i += 1

    i_sort = np.argsort(rdiff[:, -1])

    rdiff = rdiff[i_sort, ...]

    np.save(
        file=os.path.join(
            conf.ana.base_group_dir,
            "ul_sens_group_amp_diffs_sorted.npy"
        ),
        arr=rdiff
    )
