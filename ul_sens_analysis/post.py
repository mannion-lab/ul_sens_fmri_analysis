import os
import logging

import numpy as np

import fmri_tools.utils

import ul_sens_analysis.config
import ul_sens_analysis.glm
import ul_sens_fmri.config
import runcmd


def run(subj_id, acq_date, post_type):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    inf_str = subj_id + "_ul_sens_" + acq_date

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    log_dir = os.path.join(subj_dir, "logs")
    log_path = os.path.join(
        log_dir,
        "{s:s}-post-log.txt".format(s=inf_str)
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fmri_tools.utils.set_logger("screen")
    fmri_tools.utils.set_logger(log_path)

    if post_type == "node_distances":
        node_distances(subj_id, acq_date, conf)
    elif post_type == "glm":
        glm(subj_id, acq_date, conf)
    elif post_type == "resid":
        resid(subj_id, acq_date, conf)
    elif post_type == "rsq":
        rsq(subj_id, acq_date, conf)


def node_distances(subj_id, acq_date, conf):

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    post_dir = os.path.join(subj_dir, conf.ana.post_dir)
    dist_dir = os.path.join(post_dir, "dist")
    os.chdir(dist_dir)

    inf_str = subj_id + "_ul_sens_" + acq_date


    # also hemispheres
    for hemi in ["lh", "rh"]:

        dist_paths = []

        # different ROIs for upper and lower presentation
        for vf in ["upper", "lower"]:

            # each ROI
            for (roi_num, roi_name) in zip(
                conf.ana.roi_numbers,
                conf.ana.roi_names
            ):

                spec_path = os.path.join(
                    "/sci/anat/db_ver1",
                    subj_id,
                    "SUMA",
                    subj_id + "_" + hemi + ".spec"
                )

                mask_path = os.path.join(
                    conf.ana.base_subj_dir,
                    subj_id,
                    conf.ana.loc_glm_dir,
                    "{s:s}-loc_{v:s}-mask-{h:s}_nf.niml.dset".format(
                        s=inf_str, v=vf, h=hemi
                    )
                )

                out_path = "{s:s}-post_{v:s}-{r:s}_mask-{h:s}_nf.niml.dset"
                out_path = out_path.format(
                    s=subj_id, v=vf, r=roi_name, h=hemi
                )

                cmd = [
                    "3dcalc",
                    "-a", mask_path,
                    "-expr", "equals(a,{n:s})".format(n=roi_num),
                    "-prefix", out_path,
                    "-overwrite"
                ]

                runcmd.run_cmd(" ".join(cmd))

                # now to find the centre node
                centre_node = fmri_tools.utils.get_centre_node(
                    surf_dset=out_path,
                    spec_path=spec_path
                )

                dist_path = "{s:s}-post_{v:s}-{r:s}_dist-{h:s}_nf"
                dist_path = dist_path.format(
                    s=subj_id, v=vf, r=roi_name, h=hemi
                )

                fmri_tools.utils.write_dist_to_centre(
                    centre_node=centre_node,
                    in_dset=out_path,
                    spec_path=spec_path,
                    dist_dset=dist_path,
                    pad_to="d:" + out_path,
                    inc_centre_node=True
                )

                dist_paths.append(dist_path + ".niml.dset")

        # this is under the hemi
        comb_path = "{s:s}-post-dist-{h:s}_nf.niml.dset"
        comb_path = comb_path.format(
            s=subj_id, h=hemi
        )

        comb_cmd = [
            "3dMean",
            "-non_zero",
            "-sum",
            "-prefix", comb_path,
            "-overwrite"
        ]

        comb_cmd.extend(dist_paths)

        runcmd.run_cmd(" ".join(comb_cmd))

def glm(subj_id, acq_date, conf):
    """Run the GLM at each node"""

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)

    post_dir = os.path.join(subj_dir, conf.ana.post_dir)
    dist_dir = os.path.join(post_dir, "dist")
    post_glm_dir = os.path.join(post_dir, "glm")
    os.chdir(post_glm_dir)

    inf_str = subj_id + "_ul_sens_" + acq_date

    for vf in ["upper", "lower"]:

        cond_details = ul_sens_analysis.glm._write_onsets(
            subj_id=subj_id,
            acq_date=acq_date,
            conf=conf,
            vf=vf,
            runs_type="all",
            log_dir=os.path.join(subj_dir, "logs")
        )

        contrast = []

        for curr_cond in cond_details:
            if "above" in curr_cond["name"]:
                curr_str = "+"
            else:
                curr_str = "-"

            contrast.append(curr_str + curr_cond["name"])

        contrast = " ".join(contrast)

        contrast_details = [
            {
                "label": "above_gt_below",
                "contrast": contrast
            }
        ]

        for hemi in ["lh", "rh"]:

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

            # to write
            glm_filename = "{s:s}-{v:s}-post_glm-{h:s}.niml.dset".format(
                s=inf_str, v=vf, h=hemi
            )

            # to write
            beta_filename = "{s:s}-{v:s}-post_beta-{h:s}.niml.dset".format(
                s=inf_str, v=vf, h=hemi
            )

            mask_path = os.path.join(
                conf.ana.base_subj_dir,
                subj_id,
                conf.ana.loc_glm_dir,
                "{s:s}-loc_{v:s}-mask-{h:s}_nf.niml.dset".format(
                    s=inf_str, v=vf, h=hemi
                )
            )

            # run the GLM on this visual field location
            fmri_tools.analysis.glm(
                run_paths=run_paths,
                output_dir=post_glm_dir,
                glm_filename=glm_filename,
                beta_filename=beta_filename,
                tr_s=conf.ana.tr_s,
                cond_details=cond_details,
                contrast_details=contrast_details,
                censor_str=conf.ana.censor_str,
                mask_filename=mask_path,
                matrix_filename="exp_design_" + vf + "_" + hemi
            )

            # now to convert the beta weights to percent signal change

            # baseline timecourse
            bltc_filename = "{s:s}-{v:s}-post_bltc-{h:s}.niml.dset".format(
                s=inf_str, v=vf, h=hemi
            )

            # baseline
            bl_filename = "{s:s}-{v:s}-post_bltc-{h:s}.niml.dset".format(
                s=inf_str, v=vf, h=hemi
            )

            # psc
            psc_filename = "{s:s}-{v:s}-post_psc-{h:s}.niml.dset".format(
                s=inf_str, v=vf, h=hemi
            )

            beta_bricks = "[40..$]"

            # check the beta bricks are as expected
            dset_labels = fmri_tools.utils.get_dset_label(
                beta_filename + beta_bricks
            )

            desired_labels = []

            for img_id in conf.exp.img_ids:
                for src_loc in ["above", "below"]:
                    desired_labels.append(
                        vf + "_" + src_loc + "_" + str(img_id) + "#0"
                    )

            assert dset_labels == desired_labels

            # run the PSC conversion
            fmri_tools.utils.beta_to_psc(
                beta_path=beta_filename,
                beta_bricks=beta_bricks,
                design_path="exp_design_" + vf + "_" + hemi + ".xmat.1D",
                bltc_path=bltc_filename,
                bl_path=bl_filename,
                psc_path=psc_filename,
            )

            data_filename = "{s:s}-{v:s}-data-amp.txt".format(
                s=inf_str, v=vf
            )

            if os.path.exists(data_filename):
                os.remove(data_filename)

            cmd = [
                "3dmaskdump",
                "-noijk",
                "-o", data_filename,
                psc_filename
            ]

            #runcmd.run_cmd(" ".join(cmd))

            # save the betas as text file also, for exploration / checking
            b_filename = "{s:s}-{v:s}-beta-amp.txt".format(
                s=inf_str, v=vf
            )

            if os.path.exists(b_filename):
                os.remove(b_filename)

            cmd = [
               "3dmaskdump",
                "-noijk",
                "-o", b_filename,
                beta_filename
            ]

            #runcmd.run_cmd(" ".join(cmd))

def resid(subj_id, acq_date, conf):

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)
    ana_dir = os.path.join(subj_dir, "analysis")

    post_dir = os.path.join(subj_dir, conf.ana.post_dir)
    resid_dir = os.path.join(post_dir, "resid")

    if not os.path.isdir(post_dir):
        os.mkdir(post_dir)

    if not os.path.isdir(resid_dir):
        os.mkdir(resid_dir)

    os.chdir(resid_dir)

    inf_str = subj_id + "_ul_sens_" + acq_date

    for vf in ["upper", "lower"]:

        # in
        bl_path = os.path.join(
            ana_dir,
            "{s:s}-{v:s}-bltc-.niml.dset".format(
                s=inf_str, v=vf
            )
        )

        # in
        resid_path = os.path.join(
            ana_dir,
            "{s:s}-{v:s}-resid-.niml.dset".format(
                s=inf_str, v=vf
            )
        )

        # out
        sse_path = "{s:s}-{v:s}-sse-.niml.dset".format(
            s=inf_str, v=vf
        )

        # out
        sse_psc_path = "{s:s}-{v:s}-sse_psc-.1D".format(
            s=inf_str, v=vf
        )

        # calculate sum-of-squares
        cmd = [
            "3dTstat",
            "-overwrite",
            "-prefix", sse_path,
            "-sos", resid_path
        ]

        runcmd.run_cmd(" ".join(cmd))

        # convert to percent signal change
        cmd = [
            "3dcalc",
            "-a", bl_path,
            "-b", sse_path,
            "-expr", "'100*b/a'",
            "-overwrite",
            "-prefix", sse_psc_path
        ]

        runcmd.run_cmd(" ".join(cmd))



def rsq(subj_id, acq_date, conf):

    subj_dir = os.path.join(conf.ana.base_subj_dir, subj_id)
    ana_dir = os.path.join(subj_dir, "analysis")

    post_dir = os.path.join(subj_dir, conf.ana.post_dir)
    rsq_dir = os.path.join(post_dir, "rsq")

    if not os.path.isdir(post_dir):
        os.mkdir(post_dir)

    if not os.path.isdir(rsq_dir):
        os.mkdir(rsq_dir)

    os.chdir(rsq_dir)

    inf_str = subj_id + "_ul_sens_" + acq_date

    for vf in ["upper", "lower"]:

        # in
        glm_path = os.path.join(
            ana_dir,
            "{s:s}-{v:s}-glm-.niml.dset".format(
                s=inf_str, v=vf
            )
        )

        bricks = "[184,187]"

        # check the beta bricks are as expected
        dset_labels = fmri_tools.utils.get_dset_label(
            glm_path + bricks
        )

        desired_labels = ["above_all_R^2", "below_all_R^2"]

        assert dset_labels == desired_labels

        cmd = [
            "3dmaskdump",
            "-noijk",
            glm_path + bricks
        ]

        cmd_out = runcmd.run_cmd(" ".join(cmd))

        roi_rsq = cmd_out.std_out.splitlines()

        rsq = [map(float, roi_r.split(" ")) for roi_r in roi_rsq]

        # out
        rsq_path = "{s:s}-{v:s}-rsq-.txt".format(
                s=inf_str, v=vf
        )

        np.savetxt(rsq_path, rsq)
