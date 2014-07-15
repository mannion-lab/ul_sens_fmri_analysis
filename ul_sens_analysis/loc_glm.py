
import os
import itertools
import logging

import numpy as np

import fmri_tools.analysis
import fmri_tools.utils

import ul_sens_analysis.config
import ul_sens_fmri.config


def run(subj_id, acq_date):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    loc_glm_dir = os.path.join(subj_dir, conf.ana.loc_glm_dir)

    log_dir = os.path.join(subj_dir, "logs")

    log_path = "{s:s}-loc_glm-log.txt".format(s=inf_str)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmri_tools.utils.set_logger("screen")
    fmri_tools.utils.set_logger(log_path)

    os.chdir(loc_glm_dir)

    cond_details = [{}, {}]

    # first, write the condition files
    for (i_vf, vf) in enumerate(("above", "below")):

        onset_path = "{s:s}-{v:s}_onsets.txt".format(s=inf_str, v=vf)

        cond_details[i_vf]["name"] = vf
        cond_details[i_vf]["onsets_path"] = onset_path
        cond_details[i_vf]["model"] = conf.ana.hrf_model

        with open(onset_path, "w") as onset_file:

            for run_num in xrange(1, conf.exp.n_runs + 1):

                run_onsets = []

                run_seq = np.load(
                    os.path.join(
                        subj_dir,
                        "logs",
                        "{s:s}_ul_sens_fmri_run_{n:02d}_seq.npy".format(
                            s=subj_id, n=run_num
                        )
                    )
                )

                run_seq = run_seq[i_vf, ...]

                n_trials = run_seq.shape[0]

                for i_trial in xrange(n_trials):

                    if run_seq[i_trial, 2] > 0.5:
                        run_onsets.append(run_seq[i_trial, 0])

                run_str = ["{n:.0f}".format(n=n) for n in run_onsets]

                onset_file.write(" ".join(run_str) + "\n")

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
            for run_num in range(1, conf.exp.n_runs + 1)
        ]

        for (i_vf, vf) in enumerate(("above", "below")):

            glm_filename = "{s:s}-loc_{v:s}-glm-{h:s}_nf.niml.dset".format(
                s=inf_str, h=hemi, v=vf
            )

            beta_filename = "{s:s}-loc_{v:s}-beta-{h:s}_nf.niml.dset".format(
                s=inf_str, h=hemi, v=vf
            )

            fmri_tools.analysis.glm(
                run_paths=run_paths,
                output_dir=loc_glm_dir,
                glm_filename=glm_filename,
                beta_filename=beta_filename,
                tr_s=conf.ana.tr_s,
                cond_details=[cond_details[i_vf]],
                contrast_details=[],
                censor_str=conf.ana.censor_str
            )
