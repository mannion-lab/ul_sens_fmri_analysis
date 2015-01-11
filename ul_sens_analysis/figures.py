
import os

import figutils
import matplotlib
figutils.set_defaults()

import matplotlib.pyplot as plt
plt.ioff()

import numpy as np
import scipy.stats
import svgutils.transform as sg

import fmri_tools.stats

import ul_sens_analysis.config
import ul_sens_analysis.group
import ul_sens_fmri.config
import ul_sens_fmri.stim


def plot_resp_amp(save_path=None):

    conf = ul_sens_analysis.config.get_conf()

    # subjects X va X pres (A,B) X src (U, L)
    (_, amp_data) = ul_sens_analysis.group.resp_amps(conf)

    # average over rois
    amp_data = np.mean(amp_data, axis=1)

    subj_mean = np.mean(amp_data, axis=0)
    subj_se = np.std(amp_data, axis=0, ddof=1) / np.sqrt(amp_data.shape[0])

    pres_locs = ("Above", "Below")
    src_locs = ("Upper", "Lower")

    print "Descriptives:"

    for (i_pres, pres_loc) in enumerate(pres_locs):
        for (i_src, src_loc) in enumerate(src_locs):

            out_str = (
                "\t" + pres_loc + ", " + src_loc + "- " +
                "Mean: {n:.4f}".format(n=subj_mean[i_pres, i_src]) +
                " SE: {n:.4f}".format(n=subj_se[i_pres, i_src])
            )

            print out_str

    pres_mean = np.mean(subj_mean, axis=-1)
    pres_se = (
        np.std(np.mean(amp_data, axis=-1), axis=0, ddof=1) /
        np.sqrt(amp_data.shape[0])
    )

    print "\n\tPres means: ", pres_mean
    print "\n\tPres SEs: ", pres_se



def plot_resp_amp_rois(save_path=None):

    conf = ul_sens_analysis.config.get_conf()

    for i_va in xrange(len(conf.roi_names)):

        plot_amps_visual_area(
            save_path + "_" + str(i_va) + ".svg",
            i_va
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


def plot_amps_visual_area(save_path, i_va):

    conf = ul_sens_fmri.config.get_conf()
    conf.ana = ul_sens_analysis.config.get_conf()

    # subjects x va x pres (A,B) x src (U, L)
    amp_data = np.load(
        os.path.join(
            conf.ana.base_group_dir,
            "ul_sens_group_amp_data.npy"
        )
    )

    amp_data = np.mean(amp_data, axis=2)

    amp_data = amp_data[:, i_va, ...]

    # might as well do stats
    stats = fmri_tools.stats.anova(
        data=amp_data,
        output_path="/tmp",
        factor_names=["pres", "src"]
    )

    print "Stats for " + str(conf.ana.roi_names[i_va]) + ":"
    print stats

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
        s=conf.ana.roi_names[i_va],
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
