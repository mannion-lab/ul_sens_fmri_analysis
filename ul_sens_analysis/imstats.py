
import os

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


def filter():

    dkl = load_dkl_frags()
    bank = _get_filter_bank()

    filt_out = np.empty((30, 2, 2, 3, 5, 4))
    filt_out.fill(np.NAN)

    for i_img in xrange(30):
        for i_src in xrange(2):
            for i_horiz in xrange(2):
                for i_chan in xrange(3):

                    dkl_for_filt = dkl[i_img, i_src, i_horiz, ..., i_chan]

                    for i_wl in xrange(5):
                        out = _apply_filter_bank(
                            dkl_for_filt,
                            bank[i_wl]
                        )

                        filt_out[i_img, i_src, i_horiz, i_chan, i_wl, :] = out

    return filt_out


def _apply_filter_bank(img, bank):

    (n_ori, n_phase, _, _) = bank.shape

    filt = np.empty(
        (
            n_ori,
            n_phase,
            img.shape[0],
            img.shape[1]
        )
    )
    filt.fill(np.NAN)

    for i_ori in xrange(bank.shape[0]):
        for i_phase in xrange(bank.shape[1]):

            out = cv2.filter2D(
                src=img,
                kernel=bank[i_ori, i_phase, ...],
                ddepth=cv2.CV_64F
            )

            filt[i_ori, i_phase, ...] = out

    filt = np.sqrt(
        np.sum(filt ** 2, axis=1)
    )
    filt = np.sum(np.sum(filt, axis=-1), axis=-1)

    return filt


def _get_filter_bank():

    # wavelengths
    lambdas = np.array([5, 10, 20, 40, 80])
    # oris
    thetas = np.radians([0, 45, 90, 135])
    # aspect
    gamma = 1
    # phases
    psis = np.array([np.pi / 2])

    filter_bank = []

    for wl in lambdas:

        sigma = (
            wl * 1.0 / np.pi * np.sqrt(np.log(2) / 2.0) *
            (
                ((2 ** 1) + 1) / ((2 ** 1) - 1)
            )
        )

        k_size = int(sigma * 6)

        if np.mod(k_size, 2) == 0:
            k_size += 1

        wl_filt = np.empty((len(thetas), len(psis), k_size, k_size))
        wl_filt.fill(np.NAN)

        for (i_ori, ori) in enumerate(thetas):

            # change the orientation convention to be 0 at horizontal and increasing angles moving
            # counter clockwise
            filt_ori = np.mod(-ori - np.pi / 2.0, 2 * np.pi)

            for (i_phase, phase) in enumerate(psis):

                filt = cv2.getGaborKernel(
                    ksize=(k_size, k_size),
                    sigma=sigma,
                    theta=filt_ori,
                    lambd=wl,
                    gamma=gamma,
                    psi=phase
                )

                wl_filt[i_ori, i_phase, ...] = filt


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


