
import os
import itertools

import numpy as np
import scipy.misc
import cv2

import psychopy.monitors
import psychopy.misc
import psychopy.filters

import monitors.conversions

import stimuli.utils

import ul_sens_fmri.config
import ul_sens_analysis.figures


def get_fragments():

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
        for (i_src, src) in enumerate(["above", "below"]):

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

    frag_dir = "/sci/study/ul_sens/imstats"

    dkl_frags = np.load(os.path.join(frag_dir, "ul_sens_img_frags_dkl.npy"))

    return dkl_frags


def run_filter():

    dkl = load_dkl_frags()
    bank = _get_filter_bank()

    ap_mask = psychopy.filters.makeMask(
        matrixSize=dkl.shape[-2],
        shape="circle",
        radius=0.9,
        range=[0,1]
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

                    break

                    # extract the dkl image and mask out the edges
                    dkl_for_filt = dkl[i_img, i_src, i_horiz, ..., i_chan]
                    dkl_for_filt *= ap_mask

                    # loop through each spatial frequency (ie wavelength)
                    for i_wl in xrange(n_sf):

                        # apply the filter bank (different oris)
                        out = _apply_filter_bank(
                            dkl_for_filt,
                            bank[i_wl]
                        )

                        # square, sum over space, square root
                        out = np.sqrt(
                            np.sum(np.sum(out ** 2, axis=-1), axis=-1)
                        )

                        filt_out[i_img, i_src, i_horiz, i_chan, i_wl, :] = out

    out_dir = "/sci/study/ul_sens/imstats"

#    np.save(
#        os.path.join(
#            out_dir,
#            "ul_sens_img_filter_output.npy"
#        ),
#        filt_out
#    )

    filt_out = np.load(out_dir + "/ul_sens_img_filter_output.npy")

    # save for SPSS
    spss_path = os.path.join(out_dir, "ul_sens_img_filter_output_spss.txt")

    header = []

    # header first - a monster loop
    for (i_src, i_horiz, i_chan, i_sf, i_ori) in itertools.product(
        range(n_src), range(n_horiz), range(n_chan), range(n_sf), range(n_ori)
    ):

        header.append(
            "_".join(
                [
                    str(i_src + 1),
                    str(i_horiz + 1),
                    str(i_chan + 1),
                    str(i_sf + 1),
                    str(i_ori + 1)
                ]
            )
        )

    with open(spss_path, "w") as spss_file:

        spss_file.write("\t".join(header))

        for i_img in range(n_img):

            spss_file.write("\n")

            for (i_src, i_horiz, i_chan, i_sf, i_ori) in itertools.product(
                range(n_src), range(n_horiz), range(n_chan), range(n_sf),
                range(n_ori)
            ):

                data = filt_out[i_img, i_src, i_horiz, i_chan, i_sf, i_ori]

                spss_file.write("{dv:.10f}\t".format(dv=data))

    return filt_out


def _apply_filter_bank(img, bank):

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

    # wavelengths
    lambdas = np.array([5, 10, 20, 40, 80])
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

            # change the orientation convention to be 0 at horizontal and increasing angles moving
            # counter clockwise
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


def calc_stats():

    conf = ul_sens_fmri.config.get_conf()

    # (image, ab, lr, 353, 353, 3)
    dkl = load_dkl_frags()

    stats = np.empty(
        (
            3,  # mean, std, slope
            dkl.shape[0],
            dkl.shape[1],
            2,  # lr
            3  # dkl
        )
    )
    stats.fill(np.NAN)

    ap_mask = psychopy.filters.makeMask(
        matrixSize=dkl.shape[-2],
        shape="circle",
        radius=0.8,
        range=[0,1]
    )

    for i_img in xrange(stats.shape[1]):
        for i_src in xrange(stats.shape[2]):
            for i_h in xrange(stats.shape[3]):
                for i_chan in xrange(stats.shape[-1]):

                    frag = dkl[i_img, i_src, i_h, :, :, i_chan]

                    # get the values from just inside the aperture
                    frag_vals = frag.flat[ap_mask.flat > 0]

                    # do some calculations

                    # mean
                    frag_mean = np.mean(frag_vals)
                    # std
                    frag_std = np.std(frag_vals)

                    # slope
                    amp_spec = stimuli.utils.get_amp_spectrum(frag, True)
                    (_, slope) = np.polyfit(
                        x=np.log(amp_spec[:, 0]),
                        y=np.log(amp_spec[:, 1]),
                        deg=1
                    )

                    stats[:, i_img, i_src, i_h, i_chan] = [
                        frag_mean,
                        frag_std,
                        slope
                    ]

    return stats


def compare_with_data(stats=None):

    conf = ul_sens_analysis.config.get_conf()

    if stats is None:
        stats = calc_stats()

    # average stats over lr
    stats = np.mean(stats, axis=-2)

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
        for i_stat in xrange(3):

            curr_stats = stats[i_stat, :, :, i_chan].flat

            (r, p) = scipy.stats.pearsonr(
                data_diff.flat,
                curr_stats
            )

            print r, p


