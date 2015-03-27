import os

import numpy as np
import scipy.stats

import ul_sens_analysis.config
import ul_sens_fmri.config

def rsqs(conf=None, subj_info=None):

    if conf is None:
        conf = ul_sens_fmri.config.get_conf()
        conf.ana = ul_sens_analysis.config.get_conf()

    if subj_info is None:
        subj_info = conf.ana.subj_info

    data = np.empty(
        (
            len(subj_info),  # subjects
            len(conf.ana.roi_names),  # ROIs
            1,
            2,  # pres_loc: upper, lower
            2   # src_loc : above, below
        )
    )
    data.fill(np.NAN)

    for (i_subj, (subj_id, acq_date)) in enumerate(subj_info):

        inf_str = subj_id + "_ul_sens_" + acq_date

        for (i_vf, vf) in enumerate(["upper", "lower"]):

            rsq_path = os.path.join(
                conf.ana.base_subj_dir,
                subj_id,
                conf.ana.post_dir,
                "rsq",
                "{s:s}-{v:s}-rsq-.txt".format(
                    s=inf_str, v=vf
                )
            )

            rsq = np.loadtxt(rsq_path)

            data[i_subj, :, 0, i_vf, :] = rsq

    # check we've filled up 'data' correctly
    assert np.sum(np.isnan(data)) == 0

    np.save(
        file=os.path.join(
            conf.ana.base_post_group_dir,
            "ul_sens_post_group_rsq_data.npy"
        ),
        arr=data
    )

    save_resp_amps_for_spss(
        data=data,
        txt_path=os.path.join(
            conf.ana.base_post_group_dir,
            "ul_sens_post_group_rsq_data_spss.txt"
        )
    )

def resids(conf=None, subj_info=None):

    if conf is None:
        conf = ul_sens_fmri.config.get_conf()
        conf.ana = ul_sens_analysis.config.get_conf()

    if subj_info is None:
        subj_info = conf.ana.subj_info

    data = np.empty(
        (
            len(subj_info),  # subjects
            len(conf.ana.roi_names),  # ROIs
            2  # pres_loc: upper, lower
        )
    )
    data.fill(np.NAN)

    for (i_subj, (subj_id, acq_date)) in enumerate(subj_info):

        inf_str = subj_id + "_ul_sens_" + acq_date

        for (i_vf, vf) in enumerate(["upper", "lower"]):

            sse_psc_path = os.path.join(
                conf.ana.base_subj_dir,
                subj_id,
                conf.ana.post_dir,
                "resid",
                "{s:s}-{v:s}-sse_psc-.1D".format(
                    s=inf_str, v=vf
                )
            )

            # this will be a ROIs long vector
            sse_psc = np.loadtxt(sse_psc_path)

            assert len(sse_psc) == len(conf.ana.roi_names)

            data[i_subj, :, i_vf] = sse_psc

    # check we've filled up 'data' correctly
    assert np.sum(np.isnan(data)) == 0

    np.save(
        file=os.path.join(
            conf.ana.base_post_group_dir,
            "ul_sens_post_group_sse_data.npy"
        ),
        arr=data
    )

#    save_resp_amps_for_spss(
#        data=data,
#        txt_path=os.path.join(
#            conf.ana.base_group_dir,
#            "ul_sens_group_amp_data_spss.txt"
#        )
#    )


def save_resp_amps_for_spss(data, txt_path):

    # average over images
    data = np.mean(data, axis=2)

    (n_subj, n_rois, n_pres, n_src) = data.shape

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

    (n_subj, n_img, _, n_src) = data.shape

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


def stats():

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    # this is subjects, rois, images, pres_loc, src_loc
    data = np.load(
        os.path.join(
            conf.ana.base_group_dir,
            "ul_sens_group_amp_data.npy"
        )
    )

    # first, let's look a the interaction between presentation and source
    # locations. we can average over rois and images
    pres_x_src = np.mean(np.mean(data, axis=2), axis=1)

    # normalise
    subj_mean = np.mean(np.mean(pres_x_src, axis=-1), axis=-1)
    grand_mean = np.mean(pres_x_src)

    pres_x_src = (
        (pres_x_src - subj_mean[:, np.newaxis, np.newaxis]) +
        grand_mean
    )

    # t-test between above and below for upper and lower
    print "T-test for pres x src (above - below):"
    for (i_pres, pres_label) in zip(xrange(2), ("upper", "lower")):

        (t, p) = scipy.stats.ttest_rel(
            pres_x_src[:, i_pres, 0],
            pres_x_src[:, i_pres, 1]
        )

        print "\t" + pres_label + ":"
        print "\t\tt = " + str(t) + "; p = " + str(p)

        src_mean = np.mean(pres_x_src[:, i_pres, :], axis=0)
        src_sem = (
            np.std(pres_x_src[:, i_pres, :], axis=0, ddof=1) /
            np.sqrt(pres_x_src.shape[0])
        )

        print "\t\tMean = " + str(src_mean) + "; SEM = " + str(src_sem)
