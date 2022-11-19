import numpy as np
from .. import common, geometry, ui
import cc3d

import memory_profiler


# @memory_profiler.profile
class MultiClassSegment:
    def __init__(self, dense_array: np.ndarray, voxelsize=None):
        self.voxelsize = voxelsize
        # assert arr.dtype == np.uint8, f'type is {arr.dtype} but uint8 as expected'
        dense_array = dense_array.astype(np.uint8)
        self.shape = dense_array.shape
        self.dtype = dense_array.dtype

        self.segments = {}
        self.default_segment = SingleSegment(dtype=self.dtype, shape=dense_array.shape)
        self.roi = geometry.one_roi(dense_array, return_index=True)
        dense_array = dense_array[self.roi]
        if dense_array.shape[0] == 0:
            return
        spoint_new = [self.roi[i].start for i in range(dense_array.ndim)]
        for c in range(1, dense_array.max()+1):
            self.segments[c] = Segment(dense_array == c, voxelsize=voxelsize,
                                       shape=self.shape, dtype=self.dtype, spoint=spoint_new)

        import gc
        gc.collect()

    def todense(self):
        return self[tuple([np.s_[:] for i in range(len(self.shape))])]
        # dense = np.zeros(self.shape, self.dtype)
        # for c in self.segments:
        #     dense[self.segments[c].roi] += self.segments[c].todense()[self.segments[c].roi]
        # return dense

    def __getitem__(self, index):
        res = self.default_segment[index]
        for c in self.segments:
            res += (self.segments[c][index]*c).astype(self.dtype)
        return res

    def __repr__(self):
        return f'{self.dtype}-voxelsize={self.voxelsize}-{roi_str(self.roi)}-\nsegments={self.segments!r}'

    def __str__(self):
        return f'{self.dtype}-voxelsize={self.voxelsize}-{roi_str(self.roi)}-segments={len(self.segments)}'


class Segment:
    def __init__(self, dense_array: np.ndarray, voxelsize=None, dtype=None, shape=None, spoint=None):
        assert (dense_array is None and shape is not None) or (dense_array is not None and (
            (shape is None and spoint is None) or (shape is not None and spoint is not None)))

        self.dtype = dtype if dtype else dense_array.dtype
        self.shape = shape if shape else dense_array.shape
        self.voxelsize = voxelsize

        self.default_segment = SingleSegment(dtype=self.dtype, shape=self.shape)

        if dense_array is None:
            self.roi = np.s_[0:0, 0:0, 0:0]
            return

        tmp_roi = geometry.one_roi(dense_array, return_index=True)
        dense_array = dense_array[tmp_roi]
        self.roi = add_spoint2roi(tmp_roi, spoint)
        spoint_new = get_spoint(self.roi)

        labels, seg_count = geometry.connected_components(dense_array != 0, return_N=True)
        # labels, seg_count = dense_array, 10
        self.segments = []
        for l in range(1, seg_count+1):
            self.segments.append(SingleSegment(labels == l, shape=self.shape, spoint=spoint_new))

    def todense(self):
        return self[tuple([np.s_[:] for i in range(len(self.shape))])]
        # dense = np.zeros(self.shape, self.dtype)

        # for segment in self.segments:
        #     dense[segment.roi] += segment.data
        # return dense

    def __getitem__(self, index):
        res = self.default_segment[index]

        for s in self.segments:
            if self.dtype == bool:
                res |= s[index]
            else:
                res += s[index]

        return res

    def __repr__(self):
        return f'{self.dtype}-voxelsize={self.voxelsize}-{roi_str(self.roi)}-\nsegments={self.segments!r}'

    def __str__(self):
        return f'{self.dtype}-voxelsize={self.voxelsize}-{roi_str(self.roi)}-segments={len(self.segments)}'


def get_spoint(roi):
    return [roi[i].start for i in range(len(roi))]


def add_spoint2roi(roi, spoint):
    if spoint is None:
        return roi

    final_roi = ()
    for i in range(len(roi)):
        if type(roi[i]) == int:
            final_roi += (roi[i]+spoint[i],)
        if type(roi[i]) == slice:
            final_roi += (slice(roi[i].start + spoint[i], roi[i].stop + spoint[i]),)
    return final_roi


def roi_str(roi):
    roistr = ()
    for i in roi:
        if type(i) == slice:
            roistr += (f'{i.start}:{i.stop}',)
        else:
            roistr += (i,)
    return str(roistr)


class SingleSegment:
    def __init__(self, dense_array=None, dtype=None, shape=None, spoint=None):

        assert (dense_array is None and shape is not None) or (dense_array is not None and (
            (shape is None and spoint is None) or (shape is not None and spoint is not None)))
        self.dtype = dtype if dtype else dense_array.dtype
        self.shape = shape if shape else dense_array.shape

        if dense_array is None:
            self.roi = np.s_[0:0, 0:0, 0:0]
            self.data = np.zeros((0, 0, 0))
        else:
            tmp_roi = geometry.one_roi(dense_array, return_index=True)
            self.data = dense_array[tmp_roi]
            self.roi = add_spoint2roi(tmp_roi, spoint)

    def todense(self):
        return self[tuple([np.s_[:] for i in range(len(self.shape))])]
        # dense = np.zeros(self.shape, self.dtype)
        # dense[self.roi] = self.data
        # return dense

    def __getitem__(self, index):

        if type(index) == tuple and len(index) == len(self.shape):
            new_shape = ()
            new_shape_idx = ()
            data_idx = ()
            not_in_range = False
            ret = ()
            for i in range(len(self.shape)):
                if type(index[i]) == int:
                    new_shape += (1,)
                    ret += 0,
                    # print(new_shape_idx)
                    new_shape_idx += (np.s_[0],)
                    if index[i] < self.roi[i].start or index[i] >= self.roi[i].stop:
                        not_in_range = True
                        data_idx += (0,)
                    else:
                        data_idx += (index[i]-self.roi[i].start,)
                elif type(index[i]) == slice and index[i].step == None:
                    index_s = 0 if index[i].start == None else index[i].start
                    index_e = min(self.shape[i], self.shape[i] if index[i].stop == None else index[i].stop)
                    ret += np.s_[:],
                    new_shape += (index_e-index_s,)

                    roi_s = max(self.roi[i].start, index_s)
                    roi_e = min(self.roi[i].stop, index_e)

                    if roi_s > roi_e:
                        not_in_range = True

                    new_shape_idx += (np.s_[roi_s-index_s:roi_e-index_s],)

                    data_idx += (np.s_[roi_s-self.roi[i].start:roi_e-self.roi[i].start],)
                else:
                    print(f'warning! this get item {index} is not supported and cause speed problem')
                    return self.todense()[index]

            res = np.zeros(new_shape, self.dtype)
            # print(new_shape, new_shape_idx, data_idx)
            if not not_in_range:
                res[new_shape_idx] = self.data[data_idx]

            return res[ret]
        print(f'warning! this get item {index} is not supported and cause speed problem')
        return self.todense()[index]

    def __repr__(self):
        return f'{self.shape}(memoryshape={self.data.shape})-{self.dtype}-{roi_str(self.roi)}'

    def __str__(self):
        return self.__repr__()
    # def __setitem__(self, index, data):
    #     if type(index) == tuple and len(index) == len(self.shape):
    #         new_shape = ()
    #         new_shape_idx = ()
    #         data_idx = ()
    #         ret = ()
    #         for i in range(len(self.shape)):
    #             if type(index[i]) == int:
    #                 new_shape += (1,)
    #                 ret += 0,
    #                 # print(new_shape_idx)
    #                 # new_shape_idx += (np.s_[0],)
    #                 if index[i] < self.roi[i].start:  # or index[i] > self.roi[i].stop:
    #                     data_idx += (0,)
    #                     new_shape_idx += (np.s_[index[i]:self.roi[i].stop],)
    #                 elif index[i] > self.roi[i].stop:
    #                     data_idx += (index[i]+1-self.roi[i].start,)
    #                     new_shape_idx += (np.s_[self.roi[i].start:index[i]+1],)
    #                 else:
    #                     data_idx += (index[i]-self.roi[i].start,)
    #                     new_shape_idx += (self.roi[i].stop-self.roi[i].start,)
    #             elif type(index[i]) == slice and index[i].step == None:
    #                 s = 0 if index[i].start == None else index[i].start
    #                 e = self.shape[i] if index[i].stop == None else index[i].stop
    #                 ret += np.s_[:],
    #                 new_shape += (e-s,)

    #                 roi_s = min(self.roi[i].start, s)
    #                 roi_e = max(self.roi[i].stop, e)
    #                 if roi_s > roi_e:
    #                     not_in_range = True

    #                 new_shape_s = roi_s-s
    #                 new_shape_idx += (np.s_[new_shape_s:roi_e-roi_s],)

    #                 data_s = roi_s-self.roi[i].start
    #                 data_idx += (np.s_[data_s:roi_e-roi_s],)
    #             else:
    #                 print('warning! this get item is not supported and cause speed problem')
    #                 return self.todense()[index]

    #         # res = np.zeros(new_shape, self.dtype)
    #         # print(new_shape, new_shape_idx, data_idx)
    #         if not not_in_range:

    #             res[new_shape_idx] = self.data[data_idx]

    #         # return res[ret]
    #     print('warning! this get item is not supported and cause speed problem')
    #     return self.todense()[index]


def test(data, data_segment=None):
    if data_segment is None:
        data_segment = MultiClassSegment(data)
    assert data_segment.todense().sum() == data.sum()
    assert data_segment.todense().shape == data.shape
    assert data_segment[data_segment.roi].sum() == data.sum()
    assert np.all(data_segment[data_segment.roi] == data[data_segment.roi])
    import pickle
    assert len(pickle.dumps(data_segment)) < data.nbytes/10

    pass
