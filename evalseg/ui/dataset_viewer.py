from .. import geometry

def slice_selector(data,ax,slice=None):
    axi = {'all': -1, 'x': 0, 'y': 1, 'z': 2}[ax]
    if type(slice)==int:
        data=geometry