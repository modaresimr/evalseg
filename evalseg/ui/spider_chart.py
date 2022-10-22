# Libraries

from abc import abstractmethod

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


def _radar_factory(num_vars, frame="polygon"):
    """Create a radar chart with `num_vars` axes.
    This function creates a RadarAxes projection and registers it.
    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = "radar"

        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location("N")

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == "circle":
                return Circle((0.5, 0.5), 0.5)
            elif frame == "polygon":
                return RegularPolygon(
                    (0.5, 0.5), num_vars, radius=0.5, edgecolor="k"
                )
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """Draw. If frame is polygon, make gridlines polygon-shaped"""
            if frame == "polygon":
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)

        def _gen_axes_spines(self):
            if frame == "circle":
                return super()._gen_axes_spines()
            elif frame == "polygon":
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes
                )

                return {"polar": spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        @abstractmethod
        def _as_mpl_axes():
            return RadarAxes, {}

    # register_projection(RadarAxes)
    return theta, RadarAxes


def spider_chart(df, rng=[0, 0.5, 1], title=None, **kwargs):
    # N = len(list(df))
    # theta, axis_class = _radar_factory(N, frame='polygon')
    # fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection=axis_class))
    # return _spider_chart(df, rng, title, ax=ax, theta=theta)
    return spider_chart_multi({title or "": df}, rng, **kwargs)


def _spider_chart(df, rng, title, ax, theta):
    df = df.T
    group = df.index
    # df=df.drop(['group'],axis=1)
    spoke_labels = list(df)
    case_data = df.values

    # fig.subplots_adjust(top=0.85, bottom=0.05)
    ax.set_title(
        title,
        position=(0.5, 1.2),
        horizontalalignment="center",
        verticalalignment="center",
    )
    ax.set_rgrids(rng, angle=0)
    ax.set_ylim(0, 1)
    # ax.set_xlim(0,1)
    # ax.set_title(title,  position=(0.5, 1.1), ha='center')
    cmap = plt.cm.get_cmap("Dark2")
    for i, d in enumerate(case_data):
        line = ax.plot(theta, d, color=cmap(i), label=group[i])
        ax.fill(theta, d, alpha=0.1, color=cmap(i))
    ax.set_varlabels(spoke_labels)
    ax.tick_params(pad=0)
    # ax.set_thetagrids(np.degrees(theta), spoke_labels, )
    ax.margins(x=0, y=0)
    legend = ax.legend(loc=(0.8, 0.9), labelspacing=0.1, fontsize="small")
    # ax.set_thetagrids([0, 10])
    # plt.show()


def spider_chart_multi(dic, rng, title=None, cols=5, **kwargs):

    N = len(next(iter(dic.values())).index)

    theta, axis_class = _radar_factory(N, frame="polygon")
    rows = (len(dic) - 1) // cols + 1

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(4 * cols, 4 * rows),
        subplot_kw=dict(projection=axis_class),
    )

    axes = np.reshape(axes, -1)
    for i, k in enumerate(dic):

        _spider_chart(dic[k], rng, k, ax=axes[i], theta=theta)
    for j in range(i + 1, len(axes)):
        axes[j].remove()
    plt.suptitle(title)
    # plt.tight_layout()
    if kwargs.get("dst", ""):
        plt.savefig(f"{kwargs['dst']}.png")
    if kwargs.get("show", 1):
        plt.show()
    else:
        plt.close()
    return axes


# Set data
df = pd.DataFrame(
    {
        "group": ["A", "B", "C", "D"],
        "var1": [0.4, 0.5, 0.30, 0.74],
        "var2": [0.29, 1.0, 0.9, 0.34],
        "var3": [0.8, 0.39, 0.23, 0.24],
        "var4": [0.7, 0.31, 0.33, 0.14],
        "var5": [0.28, 0.15, 0.32, 0.14],
    }
)

# plotSpiderChart(df,[0,.5])
