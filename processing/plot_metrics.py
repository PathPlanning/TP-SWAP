import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
import numpy as np
import datetime

from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FormatStrFormatter


class OOMFormatter(ScalarFormatter):
    def __init__(self, fformat="%.1f", offset=True, mathText=True):
        self.fformat = fformat
        ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)

    def _set_format(self, vmin=None, vmax=None):
        self.format = self.fformat
        if self._useMathText:
            self.format = r"$\mathdefault{%s}$" % self.format


def plot_metrics(
    metrics,
    metric_titles,
    metric_maps,
    alg_order,
    data,
    agents_ranges, 
    colors_for_plots, 
    alg_titles,
    fig_size,
    show_title,
    title_size,
    map_name_size,
    line_width,
    marker_size,
    label_size,
    legend_size,
    tick_label_size,
    ofsetttext_size,
    suffix
):
    markers = ["o", "v", "D", "s"]
    for metric in metrics:
        metric_legend = []
        plots_leg = []

        if len(metric_maps) > 4 and len(metric_maps) % 2 == 0:
            fig, ax_plots = plt.subplots(2, len(metric_maps) // 2, figsize=fig_size)
        else:
            fig, ax_plots = plt.subplots(1, len(metric_maps), figsize=fig_size)

        if len(metric_maps) == 1:
            ax_plots = [ax_plots]

        if show_title:
            fig.suptitle(
                metric_titles[metric], fontsize=title_size, fontweight="bold", y=1.05
            )
        fonttype = {"family": "monospace"}

        for map_id, map_type in enumerate(metric_maps):

            if len(metric_maps) > 4 and len(metric_maps) % 2 == 0:
                curr_ax = ax_plots[map_id % 2, map_id // 2]
            else:
                curr_ax = ax_plots[map_id]

            curr_ax.set_title(map_type, fontsize=map_name_size, **fonttype, fontweight="bold")
            for alg_id, alg_type in enumerate(alg_order):
                trans1 = Affine2D().translate((alg_id - 1) / 3, 0.0) + curr_ax.transData
                result = data[map_type][alg_type]
                p1 = result[metric].plot(
                    ax=curr_ax,
                    lw=line_width,
                    color=colors_for_plots[alg_id],
                    linestyle="--",
                    transform=trans1,
                    marker=markers[alg_id],
                )
                p1 = p1.lines

                lower_error = result[metric + " (std)"]
                upper_error = result[metric + " (std)"]
                error = [lower_error, upper_error]

                p3 = curr_ax.errorbar(
                    agents_ranges[map_type][alg_type],
                    result[metric],
                    error,
                    fmt=markers[alg_id],
                    markersize=marker_size,
                    capsize=19,
                    capthick=7,
                    lw=line_width-5,
                    color=colors_for_plots[alg_id],
                    transform=trans1,
                )

                curr_ax.fill_between(
                    agents_ranges[map_type][alg_type],
                    result[metric] - lower_error,
                    result[metric] + upper_error,
                    alpha=0.3,
                    color=colors_for_plots[alg_id],
                    transform=trans1,
                )

                p2 = curr_ax.fill(
                    np.NaN,
                    np.NaN,
                    color=colors_for_plots[alg_id],
                    alpha=0.5,
                    linewidth=0,
                )

                curr_leg_title = alg_titles[alg_type]
                if curr_leg_title not in metric_legend:
                    metric_legend.append(curr_leg_title)
                    plots_leg.append((p1[alg_id * 4], p2[0], p3[0]))

            if len(metric_maps) > 4 and len(metric_maps) % 2 == 0:
                ax_plots[0, 0].set_ylabel(
                    metric, fontsize=label_size, fontweight="bold", labelpad=40
                )
                ax_plots[1, 0].set_ylabel(
                    metric, fontsize=label_size, fontweight="bold", labelpad=40
                )

            else:
                ax_plots[0].set_ylabel(
                    metric, fontsize=100, fontweight="bold", labelpad=40
                )

            if len(metric_maps) > 4 and len(metric_maps) % 2 == 0:
                if map_id % 2 == 1:
                    curr_ax.set_xlabel(
                        "Number of agents", fontsize=label_size, fontweight="bold"
                    )
                else:
                    curr_ax.set(xlabel=None)
            else:
                curr_ax.set_xlabel(
                    "Number of agents", fontsize=label_size, fontweight="bold", labelpad=40
                )

            if len(metric_maps) > 4 and len(metric_maps) % 2 == 0:
                ax_plots[0, 0].legend(plots_leg, metric_legend, fontsize=legend_size, loc=2)
            else:
                ax_plots[0].legend(plots_leg, metric_legend, fontsize=legend_size, loc=2)
            curr_ax.grid()

            curr_ax.yaxis.set_major_formatter(OOMFormatter("%.1f"))

            curr_ax.tick_params(axis="both", which="major", labelsize=tick_label_size)
            curr_ax.tick_params(axis="y", which="major", pad=60)
            curr_ax.tick_params(axis="x", which="major", pad=40)

            curr_ax.ticklabel_format(
                axis="y", style="sci", scilimits=(0, 0), useMathText=True
            )

            curr_ax.yaxis.offsetText.set_fontsize(ofsetttext_size)
            curr_ax.set_axisbelow(True)
        plt.show()

        if suffix is None:
            now = datetime.datetime.now()
            suffix = now.strftime("%d_%m_%y_%H_%M_%S")
        metrix_file_name = metric.lower().replace(" ", "_").replace(".", "")
        fig.savefig(
            f"./img/{metrix_file_name}_{suffix}.png",
            bbox_inches="tight",
            transparent=True,
            pad_inches=0,
            dpi=20
        )