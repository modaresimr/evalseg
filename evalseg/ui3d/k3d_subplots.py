import k3d

from ipywidgets import Box, Layout, Output
from IPython.display import display

box_layout = Layout(  # overflow='scroll hidden',
    border='1px solid black',
    width='100%',
    height='',
    flex_flow='wrap',
    display='flex')


class k3d_subplots:
    def __init__(self):
        self.box = Box([], layout=box_layout)
        display(self.box)
        self.plots = []

    def add_plot(self, plot, title=None, siz=.3):
        item_layout = Layout(height='auto',
                             min_width=f'{siz*100}%',
                             width='auto',
                             border='1px solid black',
                             margin='0px')

        out = Output(layout=item_layout)
        self.box.children += (out,)
        with out:
            display(plot)
        plot.outputs.append(out)
        self.plots.append(plot)
        if title:
            plot += k3d.text2d(title, position=(0.5, 0), reference_point='ct', label_box=False, is_html=True)

    def sync_camera_view(self):
        for plot in self.plots:
            plot.camera = self.plots[0].camera
            # plot.camera_fov = self.plots[0].camera_fov*plot.grid[1]/self.plots[0].grid[1]

    def set_menu_visiblity(self, visible=True):
        for plot in self.plots:
            plot.menu_visibility = visible

    def remove_colorbar(self):
        for plot in self.plots:
            plot.colorbar_object_id = 10000

    # def take_snapshot(self):
    #     for plot in self.plots:
    #         plot.s
