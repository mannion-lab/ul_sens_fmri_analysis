import os
import logging

import numpy as np
import scipy.stats

import fmri_tools.analysis
import fmri_tools.utils

import runcmd

import ul_sens_analysis.config
import ul_sens_fmri.config


def run_glm(subj_id, acq_date):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    log_dir = os.path.join(subj_dir, "logs")

    log_path = os.path.join(
        log_dir,
        "{s:s}-rsa-log.txt".format(s=inf_str)
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmri_tools.utils.set_logger("screen")
    fmri_tools.utils.set_logger(log_path)

    # run separately for VF location and run split
    for vf in ("above", "below"):
        for runs_type in ("odd", "even"):

            rsa_dir = os.path.join(
                subj_dir,
                conf.ana.rsa_dir,
                vf,
                runs_type
            )

            os.chdir(rsa_dir)

            # write out the condition details
            details = _write_onsets(
                subj_id=subj_id,
                acq_date=acq_date,
                conf=conf,
                vf=vf,
                runs_type=runs_type,
                log_dir=log_dir
            )

            if runs_type == "odd":
                run_num_range = range(1, conf.exp.n_runs + 1, 2)
            elif runs_type == "even":
                run_num_range = range(2, conf.exp.n_runs + 1, 2)

            # loop over hemispheres
            for hemi in ("lh", "rh"):

                run_paths = [
                    os.path.join(
                        subj_dir,
                        "func",
                        "run_{n:02d}".format(n=run_num),
                        "{s:s}-run_{n:02d}-uw-{h:s}_nf.niml.dset".format(
                            s=inf_str, n=run_num, h=hemi
                        )
                    )
                    for run_num in run_num_range
                ]

                file_base = "{s:s}-rsa_{v:s}_{r:s}-{t:s}-{h:s}_nf{e:s}"

                glm_filename = file_base.format(
                    s=inf_str,
                    v=vf,
                    r=runs_type,
                    h=hemi,
                    t="glm",
                    e=".niml.dset"
                )
                beta_filename = file_base.format(
                    s=inf_str,
                    v=vf,
                    r=runs_type,
                    h=hemi,
                    t="beta",
                    e=".niml.dset"
                )

                # might as well use the mask to make it run quicker
                mask_path = os.path.join(
                    conf.ana.base_subj_dir,
                    subj_id,
                    conf.ana.loc_glm_dir,
                    "{s:s}-loc_{v:s}-mask-{h:s}_nf.niml.dset".format(
                        s=inf_str, v=vf, h=hemi
                    )
                )

                extra_args = ["-mask", mask_path]

                fmri_tools.analysis.glm(
                    run_paths=run_paths,
                    output_dir=rsa_dir,
                    glm_filename=glm_filename,
                    beta_filename=beta_filename,
                    tr_s=conf.ana.tr_s,
                    cond_details=details,
                    contrast_details=[],  # no contrasts necessary
                    censor_str=conf.ana.censor_str,
                    extra_args=extra_args,
                    extra_reml_args=extra_args
                )

                data_filename = file_base.format(
                    s=inf_str,
                    v=vf,
                    r=runs_type,
                    h=hemi,
                    t="data",
                    e=".txt"
                )

                if os.path.exists(data_filename):
                    os.remove(data_filename)

                # want the t-value for each image
                cmd = [
                    "3dmaskdump",
                    "-noijk",
                    "-mask", mask_path,
                    "-o", data_filename,
                    mask_path,
                    glm_filename + "[2..$(2)]"
                ]

                runcmd.run_cmd(" ".join(cmd))


def _write_onsets(subj_id, acq_date, conf, vf, runs_type, log_dir):
    """Write the onsets for a particular visual field location and runs type.

    Parameters
    ----------
    subj_id: string
        Subject ID
    acq_date: string
        Date subject was scanned, in YYYYMMDD
    conf: ConfigContainer
        ul_sens_fmri config
    vf: string, {"above", "below"}
        Presentation location
    runs_type: string, {"odd", "even"}
        Whether to include the odd or even runs
    log_dir: string
        Location of the subject logfiles

    Returns
    -------
    details: list of dicts
        Each item is a dictionary for a particular image ID and source location
        combination. It includes keys for the HRF model, the condition name,
        and the onsets filename.

    """

    if vf == "above":
        i_vf = 0
    elif vf == "below":
        i_vf = 1
    else:
        raise ValueError("Unknown vf")

    if runs_type == "odd":
        run_num_range = range(1, conf.exp.n_runs + 1, 2)
    elif runs_type == "even":
        run_num_range = range(2, conf.exp.n_runs + 1, 2)
    else:
        raise ValueError("Unknown runs_type")

    run_seqs = [
        np.load(
            os.path.join(
                log_dir,
                subj_id + "_ul_sens_fmri_run_{n:02d}_seq.npy".format(
                    s=subj_id,
                    n=run_num
                )
            )
        )
        for run_num in run_num_range
    ]

    inf_str = subj_id + "_ul_sens_" + acq_date

    details = []

    # image ID
    for img_id in conf.exp.img_ids:

        # source location
        for (i_sl, sl) in enumerate(("upper", "lower")):

            # this is the file to write
            onset_path = "{s:s}-{v:s}_{sl:s}_{i:d}_onsets.txt".format(
                s=inf_str,
                v=vf,
                sl=sl,
                i=img_id
            )

            # save the details for this condition
            details.append(
                {
                    "name": "{v:s}_{sl:s}_{i:d}".format(
                        v=vf,
                        sl=sl,
                        i=img_id
                    ),
                    "onsets_path": onset_path,
                    "model": conf.ana.hrf_model
                }
            )

            with open(onset_path, "w") as onset_file:

                for run_seq in run_seqs:

                    run_onsets = []

                    # run seq is (pres loc, trial number, trial info)
                    # where trial info is:
                    #   0: time, in seconds, when it starts
                    #   1: source location 1 for upper, 2 for lower, 0 for null
                    #   2: image id
                    #   3: whether it is in the 'pre' events
                    #   4: been prepped

                    # pull out this visual field location - either above or
                    # below
                    curr_run_seq = run_seq[i_vf, ...]

                    # axis 0 is now trials
                    n_trials = curr_run_seq.shape[0]

                    for i_trial in xrange(n_trials):

                        # a valid trial if it has the image id that we're after
                        trial_ok = np.logical_and(
                            int(curr_run_seq[i_trial, 2]) == img_id,
                            int(curr_run_seq[i_trial, 1]) == (i_sl + 1)
                        )

                        if trial_ok:
                            run_onsets.append(curr_run_seq[i_trial, 0])

                    # still inside the run loop - format the onsets as integers
                    # (they will always be multiples of 4, which is the bin
                    # length)
                    run_str = ["{n:.0f}".format(n=n) for n in run_onsets]

                    # each run is a row in the onsets file, so write it out and
                    # then move on to the next run
                    onset_file.write(" ".join(run_str) + "\n")

    return details


def run_rdm(subj_id, acq_date):
    """Converts the estimates into a set of representational dissimilarity
    matrices. Eventual output is:
        roi (V1, V2, V3) x
        runs_type (odd, even) x
        vf location (above, below) x
        upper images, lower images x
        upper images, lower images
    """

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    n_cond = int(conf.exp.n_img * conf.exp.n_src_locs)

    rdms = np.empty(
        (
            len(conf.ana.roi_names),   # rois
            2,  # runs type (odd, even)
            2,  # pres loc  (above, below)
            n_cond,
            n_cond
        )
    )
    rdms.fill(np.NAN)

    file_base = "{s:s}-rsa_{v:s}_{r:s}-{t:s}-{h:s}_nf{e:s}"

    # run separately for VF location and run split
    for (i_run_type, runs_type) in enumerate(("odd", "even")):
        for (i_vf, vf) in enumerate(("above", "below")):

            rsa_dir = os.path.join(
                subj_dir,
                conf.ana.rsa_dir,
                vf,
                runs_type
            )

            os.chdir(rsa_dir)

            data = []

            for hemi in ("lh", "rh"):

                data_filename = file_base.format(
                    s=inf_str,
                    v=vf,
                    r=runs_type,
                    h=hemi,
                    t="data",
                    e=".txt"
                )

                vf_data = np.loadtxt(data_filename)

                data.append(vf_data)

            data = np.vstack(data)

            assert data.shape[1] == n_cond + 1

            for (i_roi, roi_id) in enumerate(conf.ana.roi_numbers):

                in_roi = data[:, 0].astype("int") == int(roi_id)

                assert np.all(data[in_roi, 0].astype("int") == int(roi_id))

                # roi_data becomes IMG_u, IMG_l, IMG_u, IMG_l ...
                roi_data = data[in_roi, 1:]

                # change it so it is IMG_u, IMG_u, ..., IMG_l, IMG_l
                i_arrange = (
                    range(0, n_cond, 2) +  # upper
                    range(1, n_cond, 2)    # lower
                )
                roi_data = roi_data[:, i_arrange]

                for i_a in xrange(n_cond):
                    for i_b in xrange(n_cond):

                        rdm = 1 - scipy.stats.pearsonr(
                            roi_data[:, i_a],
                            roi_data[:, i_b]
                        )[0]

                        rdms[i_roi, i_run_type, i_vf, i_a, i_b] = rdm

    assert np.sum(np.isnan(rdms)) == 0

    rdm_filename = os.path.join(
        subj_dir,
        conf.ana.rsa_dir,
        "{s:s}-rsa-rdm-.npy".format(s=inf_str)
    )

    np.save(rdm_filename, rdms)
