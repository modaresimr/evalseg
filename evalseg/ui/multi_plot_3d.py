import numpy as np

from .. import geometry

epsilon = 0.00001


def _get_common_region(dict_of_images):
    data = np.array(list(dict_of_images.values()))
    idx = geometry.one_roi(data, ignore=[0], threshold=10, return_index=True)
    return idx[1:4]


def multi_plot_3d(dict_of_images, spacing=None, interactive=False):
    spacing = np.array([1, 1, 1] if spacing is None else spacing)

    dict_of_images = dict_of_images.copy()
    f = {}
    idx = _get_common_region(dict_of_images)
    for k in dict_of_images:
        x = dict_of_images[k][idx].copy()
        f[k] = np.clip(x, 0, 5) / min(5, x.max() + epsilon)
    if interactive:
        import plotly.express as px

        data = np.array(list(f.values()))
        fig = px.imshow(data, animation_frame=3, facet_col=0, facet_col_wrap=5)
        itemsmap = {f"{i}": key for i, key in enumerate(f)}
        fig.for_each_annotation(
            lambda a: a.update(text=itemsmap[a.text.split("=")[1]])
        )
        # fig.write_html('a.html')
        fig.show()

    else:
        import matplotlib.pyplot as plt

        data = np.array(list(f.values()))
        fig, axes = plt.subplots(
            data.shape[3], data.shape[0], figsize=(14, 50)
        )
        if data.shape[3] == 1:
            axes = [axes]
        itemsmap = {i: key for i, key in enumerate(f)}
        for i in range(data.shape[3]):
            for j in range(data.shape[0]):
                axes[i][j].imshow(
                    data[j, :, :, i],
                    vmin=0,
                    vmax=1,
                    aspect=spacing[0] / spacing[1],
                )
                axes[i][j].set_title(itemsmap[j])
        fig.show()
