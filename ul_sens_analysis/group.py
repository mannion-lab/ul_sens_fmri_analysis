import os

import numpy as np
import scipy.stats

import fmri_tools.stats

import ul_sens_analysis.config
import ul_sens_fmri.config


def resp_amps(conf=None, subj_info=None, loc_mask=True):

    if conf is None:
        conf = ul_sens_fmri.config.get_conf()
        conf.ana = ul_sens_analysis.config.get_conf()

    if subj_info is None:
        subj_info = conf.ana.subj_info

    if loc_mask:
        mask_descrip = ""
    else:
        mask_descrip = "_ret_roi"

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

        if loc_mask:
            data_dir = os.path.join(
                conf.ana.base_subj_dir,
                subj_id,
                "analysis"
            )
        else:
            data_dir = os.path.join(
                conf.ana.base_subj_dir,
                subj_id,
                "post_analysis",
                "ret_roi"
            )

        for (i_vf, vf) in enumerate(["upper", "lower"]):

            # this will be ROIs x (n_img x 2)
            # the columns are (above, below, above, below, ...)
            subj_data = np.loadtxt(
                os.path.join(
                    data_dir,
                    (
                        subj_id + "_ul_sens_" + acq_date + "-" +
                        vf + mask_descrip + "-data-amp.txt"
                    )
                )
            )

            # this is the above source images; even indices
            data[i_subj, :, :, i_vf, 0] = subj_data[:, 0::2]
            # this is the below source images; odd indices
            data[i_subj, :, :, i_vf, 1] = subj_data[:, 1::2]

    # check we've filled up 'data' correctly
    assert np.sum(np.isnan(data)) == 0

    save_path = os.path.join(
        conf.ana.base_group_dir,
        "ul_sens_group_amp{m:s}_data.npy".format(m=mask_descrip)
    )

    np.save(
        file=save_path,
        arr=data
    )

    spss_save_path = os.path.join(
        conf.ana.base_group_dir,
        "ul_sens_group_amp{m:s}_data_spss.txt".format(m=mask_descrip)
    )

    save_resp_amps_for_spss(
        data=data,
        txt_path=spss_save_path
    )

    return data


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

    diff = data[:, :, 0, :] - data[:, :, 1, :]

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

    return diff


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
            2,  # pres_loc: upper, lower,
            2,  # src loc
            12  # window
        )
    )
    data.fill(np.NAN)

    for (i_subj, (subj_id, acq_date)) in enumerate(subj_info):

        inf_str = subj_id + "_ul_sens_" + acq_date

        traces_path = os.path.join(
            conf.ana.base_subj_dir,
            subj_id,
            conf.ana.post_dir,
            "resid",
            "{s:s}--traces-.npy".format(s=inf_str)
        )

        traces = np.load(traces_path)

        data[i_subj, ...] = traces

    # check we've filled up 'data' correctly
    assert np.sum(np.isnan(data)) == 0

    np.save(
        file=os.path.join(
            conf.ana.base_group_dir,
            "ul_sens_group_traces_data.npy"
        ),
        arr=data
    )


def tasks(conf=None, subj_info=None):

    if conf is None:
        conf = ul_sens_fmri.config.get_conf()
        conf.ana = ul_sens_analysis.config.get_conf()

    if subj_info is None:
        subj_info = conf.ana.subj_info

    data = np.empty(
        (
            len(subj_info),
            2,  # pres loc
            2,  # src loc
            20  # perf bins
        )
    )
    data.fill(np.NAN)

    for (i_subj, (subj_id, acq_date)) in enumerate(subj_info):

        inf_str = subj_id + "_ul_sens_" + acq_date

        perf_path = os.path.join(
            conf.ana.base_subj_dir,
            subj_id,
            conf.ana.post_dir,
            "task",
            "{s:s}--perf-.npy".format(s=inf_str)
        )

        perf = np.load(perf_path)

        data[i_subj, ...] = perf

    # check we've filled up 'data' correctly
    assert np.sum(np.isnan(data)) == 0

    np.save(
        file=os.path.join(
            conf.ana.base_group_dir,
            "ul_sens_group_perf_data.npy"
        ),
        arr=data
    )

    # run the stats
    i_avg = range(4, 10)

    # average over the window
    perf_data = np.mean(data[..., i_avg], axis=-1)

    stats = fmri_tools.stats.anova(
        data=perf_data,
        output_path="/tmp",
        factor_names=["pres", "src"]
    )

    print stats
