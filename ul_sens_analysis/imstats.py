
import os

import numpy as np
import scipy.misc
import cv2

import psychopy.monitors
import psychopy.misc
import psychopy.filters

import monitors.conversions

import ul_sens_fmri.config
import ul_sens_analysis.figures


def get_fragments():
    """Extracts and saves the patches in DKL space"""

    exp_conf = ul_sens_fmri.config.get_conf()

    monitor = psychopy.monitors.Monitor(exp_conf.exp.monitor_name)

    ap_pos_pix = {
        h_side: psychopy.misc.deg2pix(
            exp_conf.stim.ap_pos_deg["a" + h_side],
            monitor
        )
        for h_side in ["l", "r"]
    }

    ap_size_pix = psychopy.misc.deg2pix(
        exp_conf.stim.ap_size_deg,
        monitor
    )

    # rows, columns
    cap_size_pix = np.array([1200, 1920])

    cap_centre_pix = cap_size_pix / 2.0

    sshot_dir = "/sci/study/ul_sens/sshots"
    sshot_files = os.listdir(sshot_dir)

    os.chdir(sshot_dir)

    rgb_frag = np.empty(
        (
            len(exp_conf.exp.img_ids),
            2,  # source (a,b)
            2,  # l/r
            ap_size_pix, ap_size_pix, 3
        )
    )
    rgb_frag.fill(np.NAN)

    dkl_frag = np.empty(rgb_frag.shape)
    dkl_frag.fill(np.NAN)

    for (i_img, img_id) in enumerate(exp_conf.exp.img_ids):
        for (i_src, _) in enumerate(["above", "below"]):

            sshot_found = False

            # find the relevant screenshot file
            for sshot_file in sshot_files:

                if str(img_id) not in sshot_file:
                    continue

                src_str = "src_loc_{n:.1f}".format(n=i_src + 1)
                if src_str not in sshot_file:
                    continue

                if "id_{n:.1f}".format(n=img_id) not in sshot_file:
                    continue

                pres_str = "pres_loc_a"
                if pres_str not in sshot_file:
                    continue

                if "crop" in sshot_file:
                    continue

                if sshot_found:
                    continue

                sshot_found = True

                # should be OK
                img = scipy.misc.imread(sshot_file)

                # rows, columns - not x, y
                assert img.shape == (1200, 1920, 3)

                for (i_horiz, horiz) in enumerate(["l", "r"]):

                    i_row_start = int(
                        cap_centre_pix[0] -
                        ap_pos_pix[horiz][1] -
                        ap_size_pix / 2.0
                    )

                    i_col_start = int(
                        cap_centre_pix[1] +
                        ap_pos_pix[horiz][0] -
                        ap_size_pix / 2.0
                    )

                    frag = img[
                        i_row_start:(i_row_start + int(ap_size_pix)),
                        i_col_start:(i_col_start + int(ap_size_pix)),
                        :
                    ]

                    frag = frag.astype("float") / 255.0

                    frag_in_dkl = monitors.conversions.rgb_image_to_dkl(
                        frag,
                        monitor.currentCalib["cie_raw"]
                    )

                    rgb_frag[i_img, i_src, i_horiz, ...] = frag
                    dkl_frag[i_img, i_src, i_horiz, ...] = frag_in_dkl

            if not sshot_found:
                raise ValueError()

    assert np.sum(np.isnan(rgb_frag)) == 0
    assert np.sum(np.isnan(dkl_frag)) == 0

    frags = [rgb_frag, dkl_frag]

    frag_dir = "/sci/study/ul_sens/imstats"

    for (frag, img_type) in zip(frags, ["rgb", "dkl"]):
        np.save(
            os.path.join(
                frag_dir,
                "ul_sens_img_frags_{t:s}.npy".format(t=img_type)
            ),
            frag
        )

    return (rgb_frag, dkl_frag)


def load_dkl_frags():
    """Returns the saved DKL patches"""

    frag_dir = "/sci/study/ul_sens/imstats"

    dkl_frags = np.load(os.path.join(frag_dir, "ul_sens_img_frags_dkl.npy"))

    return dkl_frags


def run_filter():
    """Runs the filter-based analysis of the DKL patches"""

    dkl = load_dkl_frags()
    bank = _get_filter_bank()

    ap_mask = psychopy.filters.makeMask(
        matrixSize=dkl.shape[-2],
        shape="circle",
        radius=0.9,
        range=[0, 1]
    )

    (n_img, n_src, n_horiz, _, _, n_chan) = dkl.shape

    n_sf = len(bank)
    n_ori = bank[0].shape[0]

    # this is rather ugly
    filt_out = np.empty(
        (
            n_img,
            n_src,
            n_horiz,
            n_chan,
            n_sf,
            n_ori
        )
    )
    filt_out.fill(np.NAN)

    for i_img in xrange(n_img):
        for i_src in xrange(n_src):
            for i_horiz in xrange(n_horiz):
                for i_chan in xrange(n_chan):

                    # extract the dkl image and mask out the edges
                    dkl_for_filt = dkl[i_img, i_src, i_horiz, ..., i_chan]
                    dkl_for_filt *= ap_mask

                    # loop through each spatial frequency
                    for i_sf in xrange(n_sf):

                        # apply the filter bank (different oris)
                        out = _apply_filter_bank(
                            dkl_for_filt,
                            bank[i_sf]
                        )

                        # square, sum over space, square root
                        out = np.sqrt(
                            np.sum(np.sum(out ** 2, axis=-1), axis=-1)
                        )

                        filt_out[i_img, i_src, i_horiz, i_chan, i_sf, :] = out

    out_dir = "/sci/study/ul_sens/imstats"

    np.save(
        os.path.join(
            out_dir,
            "ul_sens_img_filter_output.npy"
        ),
        filt_out
    )

    return filt_out


def _apply_filter_bank(img, bank):
    """Applies a filter bank (different oris) to an image"""

    n_ori = bank.shape[0]

    filt = np.empty(
        (
            n_ori,
            img.shape[0],
            img.shape[1]
        )
    )
    filt.fill(np.NAN)

    for i_ori in xrange(bank.shape[0]):

        # filter using openCV
        out = cv2.filter2D(
            src=img,
            kernel=bank[i_ori, ...],
            ddepth=cv2.CV_64F
        )

        filt[i_ori, ...] = out

    return filt


def _get_filter_bank():
    """Returns the filter banks"""

    # convert to cpd, for info
    monitor = psychopy.monitors.Monitor("BOLDscreen")
    one_deg_in_pix = psychopy.misc.deg2pix(1, monitor)

    sfs = [1.0, 2.0, 4.0, 8.0, 16.0]

    lambdas = []

    for sf in sfs:
        lambdas.append(
            one_deg_in_pix / sf
        )

    # oris
    thetas = np.radians([0, 45, 90, 135])
    # aspect
    gamma = 1
    # phases
    psi = np.pi / 2

    filter_bank = []

    for wl in lambdas:

        # set sigma so that bandwidth is 1 octave
        sigma = (
            wl * 1.0 / np.pi * np.sqrt(np.log(2) / 2.0) *
            (
                ((2 ** 1) + 1) / ((2 ** 1) - 1)
            )
        )

        k_size = int(sigma * 6)

        # if the size is even, add one because opencv needs odd
        if np.mod(k_size, 2) == 0:
            k_size += 1

        wl_filt = np.empty((len(thetas), k_size, k_size))
        wl_filt.fill(np.NAN)

        for (i_ori, ori) in enumerate(thetas):

            # change the orientation convention to be 0 at horizontal and
            # increasing angles moving counter clockwise
            filt_ori = np.mod(-ori - np.pi / 2.0, 2 * np.pi)

            filt = cv2.getGaborKernel(
                ksize=(k_size, k_size),
                sigma=sigma,
                theta=filt_ori,
                lambd=wl,
                gamma=gamma,
                psi=psi
            )

            wl_filt[i_ori, ...] = filt

        filter_bank.append(wl_filt)

    return filter_bank


def load_filter_output():
    """Returns the saved filter output"""

    filt_path = os.path.join(
        "/sci/study/ul_sens/imstats",
        "ul_sens_img_filter_output.npy"
    )

    return np.load(filt_path)


def test_radial_bias():
    """Compares the 45 and 135 degree orientations for different power"""

    filt_out = load_filter_output()

    # average over SF
    filt_out = np.mean(filt_out, axis=-2)

    (_, n_src, n_horiz, n_chan, _) = filt_out.shape

    for i_chan in xrange(n_chan):
        print "Channel: " + str(i_chan)

        for i_src in xrange(n_src):
            print "\tSource: " + str(i_src)

            for i_horiz in xrange(n_horiz):
                print "\t\tHoriz: " + str(i_horiz)

                data = filt_out[:, i_src, i_horiz, i_chan, :]

                # could use wilcoxon for non-parameteric, but then how to run
                # other ANOVA based stats is unclear
                test_meth = scipy.stats.ttest_rel
                # test_meth = scipy.stats.wilcoxon

                (t, p) = test_meth(data[:, 1], data[:, 3])

                print "\t" * 3 + str(t) + ", " + str(p)
                print "\t" * 3 + "{n:.3f}".format(
                    n=np.median(data[:, 1] - data[:, 3])
                )
                print "\t" * 3 + "{n:.3f}".format(
                    n=np.mean(data[:, 1] - data[:, 3])
                )


def run_hist():
    """Calculates the luminance and contrast stats from the DKL frags"""

    # (image, ab, lr, 353, 353, 3)
    dkl = load_dkl_frags()

    (n_img, n_src, n_horiz, _, _, n_chan) = dkl.shape

    hist = np.empty(
        (
            2,  # mean, std
            n_img,
            n_src,
            n_horiz,  # lr
            n_chan  # dkl
        )
    )
    hist.fill(np.NAN)

    ap_mask = psychopy.filters.makeMask(
        matrixSize=dkl.shape[-2],
        shape="circle",
        radius=0.8,
        range=[0, 1]
    )

    for i_img in xrange(n_img):
        for i_src in xrange(n_src):
            for i_h in xrange(n_horiz):
                for i_chan in xrange(n_chan):

                    frag = dkl[i_img, i_src, i_h, :, :, i_chan]

                    # get the values from just inside the aperture
                    frag_vals = frag.flat[ap_mask.flat > 0]

                    # do some calculations

                    # mean
                    frag_mean = np.mean(frag_vals)
                    # std
                    frag_std = np.std(frag_vals)

                    hist[:, i_img, i_src, i_h, i_chan] = [
                        frag_mean,
                        frag_std
                    ]

    hist_path = os.path.join(
        "/sci/study/ul_sens/imstats",
        "ul_sens_img_hist_output.npy"
    )

    np.save(hist_path, hist)

    return hist


def load_hist_output():
    """Loads the histogram-based output"""

    hist_path = os.path.join(
        "/sci/study/ul_sens/imstats",
        "ul_sens_img_hist_output.npy"
    )

    return np.load(hist_path)


def compare_filt_with_data():
    """Correlates filter output with upper-lower"""

    filt = load_filter_output()

    filt = np.mean(filt, axis=2)

    (_, _, n_chan, n_sf, n_ori) = filt.shape

    conf = ul_sens_analysis.config.get_conf()

    data_path = os.path.join(
        conf.base_group_dir,
        "ul_sens_group_amp{m:s}_data.npy".format(m="")
    )

    # (subject, ROI, img, pres, src)
    data = np.load(data_path)

    # average over subjects and ROIs
    data = np.mean(np.mean(data, axis=0), axis=0)

    # upper - lower
    data_diff = data[:, 0, :] - data[:, 1, :]

    for i_chan in xrange(n_chan):

        print "Chan: ", i_chan

        sf = np.mean(filt[:, :, i_chan, :, :], axis=-1)

        print "\t\tSF:"
        for i_sf in xrange(n_sf):

            curr_filt = sf[..., i_sf].flat

            (r, p) = scipy.stats.pearsonr(
                data_diff.flat,
                curr_filt
            )

            print "\t\t\t", i_sf, ", ", r, ", ", p

        ori = np.mean(filt[:, :, i_chan, :, :], axis=-2)

        print "\t\tOri:"
        for i_ori in xrange(n_ori):

            curr_filt = ori[..., i_ori].flat

            (r, p) = scipy.stats.spearmanr(
                data_diff.flat,
                curr_filt
            )

            print "\t\t\t", i_ori, ", ", r, ", ", p

    return (data_diff, filt)


def compare_hist_with_data():
    """Correlates histogram output with upper-lower"""

    conf = ul_sens_analysis.config.get_conf()

    hist = load_hist_output()

    # average hist over lr
    hist = np.mean(hist, axis=-2)

    data_path = os.path.join(
        conf.base_group_dir,
        "ul_sens_group_amp{m:s}_data.npy".format(m="")
    )

    # (subject, ROI, img, pres, src)
    data = np.load(data_path)

    # average over subjects and ROIs
    data = np.mean(np.mean(data, axis=0), axis=0)

    # upper - lower
    data_diff = data[:, 0, :] - data[:, 1, :]

    for i_chan in xrange(3):
        print "Chan: ", i_chan

        for i_stat in xrange(2):

            print "\tStat: ", i_stat

            curr_hist = hist[i_stat, :, :, i_chan].flat

            (r, p) = scipy.stats.spearmanr(
                data_diff.flat,
                curr_hist
            )

            print "\t\tr: ", r
            print "\t\tp: ", p
