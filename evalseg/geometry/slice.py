import numpy as np

from ..io import SegmentArray


def slice_segment(data: SegmentArray, dim, cuts) -> SegmentArray:
    if dim == -1 or dim == 'all' or dim == None:
        return data
    spacing = data.voxelsize
    if dim == 1 or dim == 'y':
        code = [2, 0, 1]
        # data = np.transpose(data, (2, 0, 1))
        # data=data[::-1,::-1,:]
    elif dim == 0 or dim == 'x':
        code = [2, 1, 0]
        # data = np.transpose(data, (2, 1, 0))
        # data=data[::-1,::-1,:]
    else:
        code = [1, 0, 2]
    data = np.transpose(data.todense(), code)
    if spacing is not None:
        spacing = spacing[code]
        # data=data[:,::-1,:]
        # data=data[::-1,:,:]

    # spacing=spacing[[2,0,1] if dim==1 else [2,1,0] if dim==0 else [1,0,2]]

    if not hasattr(cuts, "__len__") or len(cuts):
        data = data[:, :, cuts]

    return SegmentArray(data, spacing)


def slice(data, spacing, dim, cuts):
    if dim == -1 or dim == 'all' or dim == None:
        return data, spacing

    if dim == 1 or dim == 'y':
        code = [2, 0, 1]
        # data = np.transpose(data, (2, 0, 1))
        # data=data[::-1,::-1,:]
    elif dim == 0 or dim == 'x':
        code = [2, 1, 0]
        # data = np.transpose(data, (2, 1, 0))
        # data=data[::-1,::-1,:]
    else:
        code = [1, 0, 2]
    data = np.transpose(data, code)
    if spacing is not None:
        spacing = spacing[code]
        # data=data[:,::-1,:]
        # data=data[::-1,:,:]

    # spacing=spacing[[2,0,1] if dim==1 else [2,1,0] if dim==0 else [1,0,2]]

    if not hasattr(cuts, "__len__") or len(cuts):
        data = data[:, :, cuts]

    return data, spacing
