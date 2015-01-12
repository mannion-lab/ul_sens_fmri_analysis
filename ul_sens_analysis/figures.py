
import os
import tempfile

import figutils
import matplotlib
figutils.set_defaults()

import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
import scipy.stats
import svgutils.transform as sg

import psychopy.filters

import fmri_tools.stats
import runcmd

import ul_sens_analysis.config
import ul_sens_analysis.group
import ul_sens_fmri.config
import ul_sens_fmri.stim


def plot_resp_amp(save_path=None):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    # subjects x va x img x pres (A,B) x src (U, L)
    amp_data = np.load(
        os.path.join(
            conf.ana.base_group_dir,
            "ul_sens_group_amp_data.npy"
        )
    )

    # average over images
    amp_data = np.mean(amp_data, axis=2)
    # and ROIs
    amp_data = np.mean(amp_data, axis=1)

    plot_amps_visual_area(
        save_path + ".svg",
        amp_data
    )

    figutils.svg_to_pdf(
        svg_path=save_path + ".svg",
        pdf_path=save_path + ".pdf"
    )


def plot_resp_amp_rois(save_path=None):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    # subjects x va x img x pres (A,B) x src (U, L)
    amp_data = np.load(
        os.path.join(
            conf.ana.base_group_dir,
            "ul_sens_group_amp_data.npy"
        )
    )

    # average over images
    amp_data = np.mean(amp_data, axis=2)

    for i_va in xrange(len(conf.ana.roi_names)):

        plot_amps_visual_area(
            save_path + "_" + str(i_va) + ".svg",
            amp_data[:, i_va, ...],
            conf.ana.roi_names[i_va]
        )

    fig = sg.SVGFigure("13.7cm", "3.56cm")

    v1_fig = sg.fromfile(save_path + "_0.svg")
    v1_plot = v1_fig.getroot()
    v1_plot.moveto(0, 0, scale=1.25)

    v2_fig = sg.fromfile(save_path + "_1.svg")
    v2_plot = v2_fig.getroot()
    v2_plot.moveto(170, 0, scale=1.25)

    v3_fig = sg.fromfile(save_path + "_2.svg")
    v3_plot = v3_fig.getroot()
    v3_plot.moveto(170 * 2, 0, scale=1.25)

    fig.append([v1_plot, v2_plot, v3_plot])

    fig.save(save_path + ".svg")

    figutils.svg_to_pdf(
        svg_path=save_path + ".svg",
        pdf_path=save_path + ".pdf"
    )


def plot_amps_visual_area(save_path, amp_data, roi_name=""):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    # average and SE over subjects
    amp_mean = np.mean(amp_data, axis=0)
    amp_se = np.std(amp_data, axis=0, ddof=1) / amp_data.shape[0]

    h_off = 0.3
    v_off = 0.3
    h_max = 0.97
    v_max = 0.97
    v_lower_max = v_off + 0.1

    (fig, ax_base, ax_plt) = figutils.setup_panel(
        size=(1.6, 1.4),
        offsets=(h_off, v_off),
        scales=(h_max, v_max, v_lower_max),
        draw_dashes=True
    )

    symbols = ["o", "s"]
    labels = ["Above", "Below"]
    colours = conf.ana.source_colours
    marker_size = 20

    for i_src in xrange(2):

        ax_plt.plot(
            range(2),
            amp_mean[:, i_src],
            color=colours[i_src]
        )

        for i_pres in xrange(2):

            ax_plt.plot(
                [i_pres] * 2,
                [
                    amp_mean[i_pres, i_src] - amp_se[i_pres, i_src],
                    amp_mean[i_pres, i_src] + amp_se[i_pres, i_src]
                ],
                color=colours[i_src]
            )

        ax_plt.scatter(
            range(2),
            amp_mean[:, i_src],
            facecolor=colours[i_src],
            edgecolor=[1] * 3,
            s=marker_size,
            zorder=100,
            marker=symbols[i_src],
            label=labels[i_src]
        )

#    ax_plt.set_ylim([0.7, 0.86])
    ax_base.set_ylim([0.0, 0.1])

    ax_base.set_xlabel("Presentation location")

    ax_base.set_xlim([-0.5, 1.5])
    ax_base.set_xticks(range(2))
    ax_base.set_xticklabels(["Upper", "Lower"])

    ax_plt.set_ylabel("Response (psc)", y=0.45)

    ax_plt.text(
        x=0.05,
        y=0.9,
        s=roi_name,
        transform=ax_plt.transAxes
    )

    leg = plt.legend(
        scatterpoints=1,
        loc=(-0.05, -0.1),
        handletextpad=0
    )
    leg.draw_frame(False)

    if save_path:
        plt.savefig(save_path)

    plt.close(fig)


def get_img_fragments(save_path=None):

    conf = ul_sens_fmri.config.get_conf()

    frags = ul_sens_fmri.stim.get_img_fragments(conf)

    mask = psychopy.filters.makeMask(
        matrixSize=conf.stim.img_aperture_size_pix,
        shape="raisedCosine",
        range=[0, 1]
    )
    mask = mask[..., np.newaxis]

    data = np.empty(
        (
            conf.exp.n_img,
            conf.exp.n_src_locs,  # above, below
            2,  # left, right
            128, 128, 3
        )
    )
    data.fill(np.NAN)

    for (i_img, img_id) in enumerate(conf.exp.img_ids):
        for (i_vert, vert) in enumerate(("a", "b")):
            for (i_horiz, horiz) in enumerate(("l", "r")):

                img = frags[img_id][vert + horiz]
                img = (img + 1) / 2.0

                img = (0.5 * (1 - mask) ) + (img * mask)

                data[i_img, i_vert, i_horiz, ...] = img

                if save_path is not None:

                    fname = os.path.join(
                        save_path,
                        (
                            "ul_sens_fmri_" +
                            "img_" + str(img_id) +
                            "_" + vert +
                            "_" + horiz + ".png"
                        )
                    )

                    plt.imsave(
                        fname,
                        (img * 255).astype("uint8")
                    )


    assert np.sum(np.isnan(data)) == 0

    return data


def write_stim_library(save_path):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    sshot_dir = "/sci/study/ul_sens/sshots"
    sshot_files = os.listdir(sshot_dir)

    cwd = os.getcwd()
    os.chdir(sshot_dir)

    pdf_list = []

    for img_id in conf.exp.img_ids:
        for (i_src_loc, src_loc) in enumerate(("above", "below")):
            for (pres_loc, floc) in zip(("upper", "lower"), ("a", "b")):

                out_file = tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False
                )
                out_file.close()

                pdf_list.append(out_file.name)

                for sshot_file in sshot_files:

                    if str(img_id) not in sshot_file:
                        continue

                    src_str = "src_loc_{n:.1f}".format(n=i_src_loc + 1)
                    if src_str not in sshot_file:
                        continue

                    if "id_{n:.1f}".format(n=img_id) not in sshot_file:
                        continue

                    pres_str = "pres_loc_" + floc
                    if pres_str not in sshot_file:
                        continue

                    if "crop" not in sshot_file:
                        continue

                    header = (
                        "Image ID: " + str(img_id) +
                        "; Source location: " + src_loc +
                        "; Presentation location: " + pres_loc
                    )

                    (root, ext) = os.path.splitext(sshot_file)

                    # made it this far, must be OK
                    cmd = [
                        "convert",
                        "-append", "'label:" + header + "'",
                        sshot_file,
                        "-compress", "jpeg",
                        out_file.name
                    ]

                    runcmd.run_cmd(" ".join(cmd))

                    break

    assert len(pdf_list) == 120

    os.chdir(cwd)

    cmd = [
        "stapler",
        "cat"
    ]

    cmd.extend(pdf_list)

    cmd.append(save_path)

    runcmd.run_cmd(" ".join(cmd))

    for pdf_file in pdf_list:
        os.remove(pdf_file)


def plot_top_resp_diff(save_path=None):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    # (img * src, 4)
    diff_data = np.load(
        os.path.join(
            conf.ana.base_group_dir,
            "ul_sens_group_amp_diffs_sorted.npy"
        )
    )

    # img x src x LR x rows x cols x colours
    img_frags = get_img_fragments()

    n_to_show = 5

    i_tops = range(n_to_show)
    i_bottoms = range(-n_to_show, 0)[::-1]

    main_fig = sg.SVGFigure("13.7cm", "13.56cm")

    tmp_files = []
    figs = []

    z = 0

    for (i_rank, rank_type) in zip((i_tops, i_bottoms), ("top", "bottom")):

        for i in i_rank:

            i_img = int(diff_data[i, 0])
            i_src = int(diff_data[i, 2])

            for (i_side, side) in enumerate(("left", "right")):

                img_file = tempfile.NamedTemporaryFile(
                    prefix=rank_type + "_" + str(i),
                    delete=False
                )
                img_file.close()

                tmp_files.append(img_file.name)

                img = img_frags[i_img, i_src, i_side, ...]

                plt.imsave(
                    fname=img_file.name + ".png",
                    arr=img,
                    vmin=0.0,
                    vmax=1.0
                )

                cmd = [
                    "convert",
                    img_file.name + ".png",
                    img_file.name + ".svg"
                ]

                runcmd.run_cmd(" ".join(cmd))

                with open(img_file.name + ".svg", "r") as svg_file:
                    svg_data = svg_file.readlines()

                svg_data.insert(
                    3,
                    '<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="128" height="128">'
                )

                del svg_data[4]

                with open(img_file.name + ".svg", "w") as svg_file:
                    svg_file.writelines(svg_data)

                fig = sg.fromfile(img_file.name + ".svg")

                fig_plot = fig.getroot()
                fig_plot.moveto(0, z * 128, scale=1.)

                figs.append(fig_plot)

                z += 1

    main_fig.append(figs)

    main_fig.save("/home/damien/tst.svg")

    for tmp_file in tmp_files:
        os.remove(tmp_file)
        os.remove(tmp_file + ".png")
        os.remove(tmp_file + ".svg")
