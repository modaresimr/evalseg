
import matplotlib.pyplot as plt
from typing_extensions import Literal
from matplotlib.pyplot import Figure, Axes


def subplots(
    nrows: int = ...,
    ncols: int = ...,
    *,
    subplot_size=(1, 1),
    squeeze: bool = False,
    sharex: Literal[False, True, "none", "all", "row", "col"] = False,
    sharey: Literal[False, True, "none", "all", "row", "col"] = False,
    subplot_kw: dict = None,
    gridspec_kw: dict = None,
    **fig_kw
):
    if 'figsize' not in fig_kw:
        fig_kw['figsize'] = ncols*subplot_size[0], nrows * subplot_size[1]

    if 'constrained_layout' not in fig_kw:
        fig_kw['constrained_layout'] = True

    fig, axes = plt.subplots(nrows, ncols, squeeze=squeeze, sharex=sharex, sharey=sharey,
                             subplot_kw=subplot_kw, gridspec_kw=gridspec_kw, **fig_kw)

    # if nrows == 1 and ncols == 1:
    #     axes = [axes]
    # if nrows == 1 or ncols == 1:
    #     axes = [axes]
    return fig, axes
