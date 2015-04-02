
import os

import numpy as np
import scipy.misc

import psychopy.monitors
import psychopy.misc


import monitors.conversions

import ul_sens_fmri.config
import ul_sens_analysis.figures


def get_fragments_dkl():

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
            for (i_horiz, horiz) in enumerate(["l", "r"]):

                # find the relevant screenshot file
                for sshot_file in sshot_files:

                    if str(img_id) not in sshot_file:
                        continue

                    src_str = "src_loc_{n:.1f}".format(n=i_src + 1)
                    if src_str not in sshot_file:
                        continue

                    if "id_{n:.1f}".format(n=img_id) not in sshot_file:
                        continue

                    if "crop" in sshot_file:
                        continue

                    # should be OK
                    img = scipy.misc.imread(sshot_file)

                    # rows, columns - not x, y
                    assert img.shape == (1200, 1920, 3)

                    i_row_start = int(
                        cap_centre_pix[0] +
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
