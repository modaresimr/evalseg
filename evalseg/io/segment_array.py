import numpy as np
from .. import common, geometry
import cc3d
import operator
import memory_profiler
import copy

# class Segment:
#     def __init__(self, shape, dtype, voxelsize=None, fill_value=0):
#         if type(self) == Segment:  # do not call directly
#             raise Exception('Call MultiClassSegment, MultiPartSegment or SingleSegment')
#         self.dtype = dtype
#         self.shape = shape
#         self.voxelsize = np.array(voxelsize if not (voxelsize is None) else [1, 1, 1])
#         self.roi = np.s_[0:0, 0:0, 0:0]
#         self.fill_value = fill_value

#     def todense(self):
#         pass

#     def __getitem__(self, index):
#         pass


# class MultiPartSegment(Segment):
#     def __init__(self, dense_array: np.ndarray, voxelsize=None, dtype=None, shape=None, spoint=None, fill_value=0):
#         super().__init__(shape=shape if shape else dense_array.shape,
#                          dtype=dtype if dtype else dense_array.dtype,
#                          voxelsize=voxelsize)
#         assert dense_array is not None
#         assert (shape is None and spoint is None) or (shape is not None and spoint is not None)

#         tmp_roi = geometry.one_roi(dense_array, return_index=True)
#         dense_array = dense_array[tmp_roi]
#         self.roi = add_spoint2roi(tmp_roi, spoint)
#         spoint_new = get_spoint(self.roi)

#         labels, seg_count = geometry.connected_components(dense_array != 0, return_N=True)
#         # labels, seg_count = dense_array, 10
#         self.segments = []
#         for l in range(1, seg_count+1):
#             self.segments.append(SingleSegment(labels == l, shape=self.shape, spoint=spoint_new))

#     def todense(self):
#         return self[tuple([np.s_[:] for i in range(len(self.shape))])]

#     def __getitem__(self, index):
#         res = get_default_array_from_roi(index, self.shape, self.dtype, self.fill_value)

#         for s in self.segments:
#             if self.dtype == bool:

#                 res[s[index] != self.fill_value] = ~self.fill_value
#             else:
#                 res += s[index]

#         return res

#     def __repr__(self):
#         return f'{self.dtype}-voxelsize={self.voxelsize}-{roi_str(self.roi)}-\nsegments={self.segments!r}'

#     def __str__(self):
#         return f'{self.dtype}-voxelsize={self.voxelsize}-{roi_str(self.roi)}-segments={len(self.segments)}'


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

# @memory_profiler.profile


def get_shape_from_roi(index, orig_shape):
    if type(index) == int:
        return 0
    new_shape = ()
    for i in range(len(index)):
        if type(index[i]) == int:
            pass
        elif type(index[i]) == slice and index[i].step == None:
            index_s = 0 if index[i].start == None else index[i].start
            index_e = min(orig_shape[i], orig_shape[i] if index[i].stop == None else index[i].stop)
            new_shape += (index_e-index_s,)
        else:
            print(f'warning! this get item {index} is not supported and cause speed problem')
            return np.zeros(orig_shape, bool)[index].shape

    return new_shape


def get_default_array_from_roi(index, orig_shape, dtype, fill_value=0):
    if type(index) == int:
        return 0
    new_shape = ()
    ret = ()
    for i in range(len(index)):
        if type(index[i]) == int:
            pass
        elif type(index[i]) == slice and index[i].step == None:
            index_s = 0 if index[i].start == None else index[i].start
            index_e = min(orig_shape[i], orig_shape[i] if index[i].stop == None else index[i].stop)
            new_shape += (max(0, index_e-index_s),)
        else:
            print(f'warning! this get item {index} is not supported and cause speed problem')
            return np.zeros(orig_shape, dtype)[index]
    res = np.full(new_shape, fill_value, dtype)
    return res


# class MultiClassSegment(Segment):
#     def __init__(self, dense_array: np.ndarray, voxelsize=None):
#         super().__init__(shape=dense_array.shape, dtype=np.uint8, voxelsize=voxelsize, fill_value=0)
#         dense_array = dense_array.astype(np.uint8)

#         self.classes = {}
#         self.roi = geometry.one_roi(dense_array, return_index=True)
#         dense_array = dense_array[self.roi]
#         if dense_array.shape[0] == 0:
#             return
#         spoint_new = [self.roi[i].start for i in range(dense_array.ndim)]
#         for c in range(1, dense_array.max()+1):
#             self.classes[c] = MultiPartSegment(dense_array == c, voxelsize=voxelsize,
#                                                shape=self.shape, dtype=self.dtype, spoint=spoint_new)

#     def todense(self):
#         return self[tuple([np.s_[:] for i in range(len(self.shape))])]

#     def __getitem__(self, index):
#         res = get_default_array_from_roi(index, self.shape, self.dtype, self.fill_value)
#         for c in self.classes:
#             if type(res) == np.ndarray:
#                 res += (self.classes[c][index]*c).astype(self.dtype)
#             else:
#                 res += self.classes[c][index]*c

#         return res

#     def __repr__(self):
#         return f'{self.dtype}-voxelsize={self.voxelsize}-{roi_str(self.roi)}-\nsegments={self.classes!r}'

#     def __str__(self):
#         return f'{self.dtype}-voxelsize={self.voxelsize}-{roi_str(self.roi)}-segments={len(self.classes)}'


class SingleSegment:
    def __init__(self, dense_array: np.ndarray, voxelsize=None, dtype=None, shape=None, spoint=None, fill_value=0, calc_roi=True):
        self.dtype = dtype if dtype else dense_array.dtype
        self.shape = shape if shape else dense_array.shape
        self.voxelsize = np.array(voxelsize if not (voxelsize is None) else [1, 1, 1])
        self.fill_value = fill_value
        assert dense_array is not None
        assert (shape is None and spoint is None) or (shape is not None and spoint is not None)

        if calc_roi:
            tmp_roi = geometry.one_roi(dense_array, return_index=True, fill_value=fill_value)
        else:
            tmp_roi = tuple([slice(0, dense_array.shape[i]) for i in range(len(shape))])
        self.data = dense_array[tmp_roi]
        self.roi = add_spoint2roi(tmp_roi, spoint)

    def _operator_single(self, opt):
        newv = copy.deepcopy(self)
        newv.fill_value = opt(self.fill_value)
        newv.data = opt(self.data)

    def todense(self):
        # return self[tuple([np.s_[:] for i in range(len(self.shape))])]
        # # dense = np.zeros(self.shape, self.dtype)
        # # dense[self.roi] = self.data
        # # return dense
        res = np.full(self.shape, self.fill_value, self.dtype)
        res[self.roi] = self.data
        return res

    def __getitem__(self, index):
        # if index == self.roi:
        #     return self
        if type(index) == tuple and len(index) == len(self.shape):
            new_shape = ()
            new_shape_idx = ()
            data_idx = ()
            not_in_range = False
            # ret = ()
            # is_completely_inside_roi = True
            for i in range(len(self.shape)):
                if type(index[i]) == int:
                    # new_shape += (1,)
                    # ret += 0,
                    # print(new_shape_idx)
                    # new_shape_idx += (0,)
                    if index[i] < self.roi[i].start or index[i] >= self.roi[i].stop:
                        not_in_range = True
                        # is_completely_inside_roi = False
                    else:
                        data_idx += (index[i]-self.roi[i].start,)
                elif type(index[i]) == slice and index[i].step == None:
                    index_s = 0 if index[i].start == None else index[i].start
                    index_e = min(self.shape[i], self.shape[i] if index[i].stop == None else index[i].stop)
                    assert index_e >= 0
                    index_e = max(index_s, index_e)
                    # ret += np.s_[:],
                    new_shape += (index_e-index_s,)
                    # if self.roi[i].start > index_s or self.roi[i].stop < index_e:
                    #     is_completely_inside_roi = False
                    roi_s = max(self.roi[i].start, index_s)
                    roi_e = min(self.roi[i].stop, index_e)

                    if roi_s > roi_e:
                        not_in_range = True

                    new_shape_idx += (np.s_[roi_s-index_s:roi_e-index_s],)

                    data_idx += (np.s_[roi_s-self.roi[i].start:roi_e-self.roi[i].start],)
                else:
                    print(f'warning! this get item {index} is not supported and cause speed problem')
                    return SingleSegment(self.todense()[index], voxelsize=self.voxelsize, calc_roi=False)
            # if is_completely_inside_roi:
            #     return self.data[data_idx]

            if not_in_range:
                empty = np.zeros(len(new_shape), int)
                return SingleSegment(np.zeros(empty, self.dtype),
                                     fill_value=self.fill_value,
                                     shape=new_shape, spoint=empty, calc_roi=False)

            return SingleSegment(self.data[data_idx],
                                 fill_value=self.fill_value,
                                 shape=new_shape, spoint=get_spoint(new_shape_idx), calc_roi=False)

            # res = np.zeros(new_shape, self.dtype)
            # if self.fill_value:
            #     res[:] = self.fill_value
            # # print(new_shape, new_shape_idx, data_idx)
            # if not not_in_range:
            #     res[new_shape_idx] = self.data[data_idx]

            # return res
        print(f'warning! this get item {index} is not supported and cause speed problem')
        return self.todense()[index]

    def __eq__(self, other, calc_roi=True):
        return self._operator(operator.__eq__, other, calc_roi=calc_roi)

    def __ne__(self, other, calc_roi=True):
        return self._operator(operator.__ne__, other, calc_roi=calc_roi)

    def __and__(self, other, calc_roi=True):
        return self._operator(operator.__and__, other, calc_roi=calc_roi)

    def __or__(self, other):
        return self._operator(operator.__or__, other, calc_roi=False)

    def __xor__(self, other):
        return self._operator(operator.__xor__, other)

    def __invert__(self):
        assert self.dtype == bool
        return self._operator_single(np.invert)

    def __abs__(self):
        assert self.dtype == bool
        return self._operator_single(operator.__abs__)

    def __add__(self, other):
        return self._operator(operator.__add__, other)

    def __sub__(self, other):
        return self._operator(operator.__sub__, other)

    def __neg__(self):
        return self._operator_single(operator.__neg__)

    def __le__(self, other):
        return self._operator(operator.__le__, other)

    def __lt__(self, other):
        return self._operator(operator.__lt__, other)

    def __gt__(self, other):
        return self._operator(operator.__gt__, other)

    def __ge__(self, other):
        return self._operator(operator.__ge__, other)

    def _operator(self, opt, other, calc_roi=True):
        """Overrides the default implementation"""
        if isinstance(other, SingleSegment) or isinstance(other, SegmentArray):
            assert len(self.shape) == len(other.shape)
            rng = tuple([slice(min(self.roi[i].start, other.roi[i].start), max(self.roi[i].stop, other.roi[i].stop)) for i in range(len(self.roi))])
            res = opt(self[rng].todense(), other[rng].todense())
            new_fill_value = opt(self.fill_value, other.fill_value)
        elif isinstance(other, np.ndarray):
            rng = tuple([slice(0, other.shape[i]) for i in range(other.ndim)])
            res = opt(self[rng].todense(), other[rng])
            # return SegmentArray(res, voxelsize=self.voxelsize, shape=self.shape, spoint=get_spoint(self.roi),  fill_value=True, multi_part=True)
            if res.dtype == bool:
                new_fill_value = res.sum() > (res.size/2)
            else:
                new_fill_value = 0
        else:
            rng = self.roi
            res = opt(self.data, other)
            new_fill_value = opt(self.fill_value, other)

          #
        return SingleSegment(res, voxelsize=self.voxelsize, shape=self.shape, spoint=get_spoint(rng),  fill_value=new_fill_value, calc_roi=calc_roi)

    def __sizeof__(self) -> int:
        return np.dtype(self.dtype).itemsize * np.prod(self.data.shape)+4

    def __repr__(self):
        orig_size = np.dtype(self.dtype).itemsize * np.prod(self.shape)
        mysize = self.__sizeof__()
        compress = 100-(mysize*100)//orig_size
        return f'SingleSegment({self.dtype}){self.shape}(compress={compress}% {humanbytes(mysize)}/{humanbytes(orig_size)}) range={roi_str(self.roi)}'

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


class SegmentArray:
    def __init__(self, dense_array: np.ndarray, voxelsize=None, fill_value=0, multi_part=True, dtype=None, shape=None, spoint=None, mask_roi=None, calc_roi=True):

        self.dtype = dtype if dtype else dense_array.dtype
        self.shape = shape if shape else dense_array.shape
        self.voxelsize = np.array(voxelsize if not (voxelsize is None) else [1, 1, 1])
        self.fill_value = fill_value
        assert dense_array is not None
        assert (shape is None and spoint is None) or (shape is not None and spoint is not None)

        if calc_roi:
            tmp_roi = geometry.one_roi(dense_array, return_index=True, fill_value=fill_value, mask_roi=mask_roi)
        else:
            tmp_roi = mask_roi if mask_roi is not None else tuple([slice(0, dense_array.shape[i]) for i in range(len(shape))])

        dense_array = dense_array[tmp_roi]
        self.roi = add_spoint2roi(tmp_roi, spoint)

        if multi_part:
            labels, seg_count = geometry.connected_components(dense_array != fill_value, return_N=True)
            # labels, seg_count = dense_array, 10
            self.segments = []
            for l in range(1, seg_count+1):
                new_dense = np.full(dense_array.shape, self.fill_value, self.dtype)
                idx = labels == l
                new_dense[idx] = dense_array[idx]
                self.segments.append(SingleSegment(new_dense, shape=self.shape, fill_value=fill_value, spoint=get_spoint(self.roi)))
        else:
            self.segments = [SingleSegment(dense_array, shape=self.shape, fill_value=fill_value, spoint=get_spoint(self.roi),
                                           calc_roi=calc_roi)]

    def todense(self):
        return self[tuple([np.s_[:] for i in range(len(self.shape))])]

    def __getitem__(self, index):
        if isinstance(index, np.ndarray):
            index = SegmentArray(index, multi_part=False)
        if isinstance(index, SegmentArray):
            assert index.dtype == bool
            if index.fill_value == False:
                return self[index.roi][index[index.roi]]
            else:
                print('warning fill_value is true so it is not recommended')
                return self.todense()[index]
        elif len(self.segments) == 1:
            return self.segments[0][index].todense()
        else:
            res = get_default_array_from_roi(index, self.shape, self.dtype, self.fill_value)
            for s in self.segments:
                segres = s[index]
                assert segres.fill_value == self.fill_value

                segres_mask = (segres.data != self.fill_value)

                # res[segres.roi].__setitem__(segres_mask, segres.data[segres_mask])
                res[segres.roi][segres_mask] = segres.data[segres_mask]

                # segres_mask = segres != self.fill_value

        return res

    def max(self):
        return max([s.max() for s in self.segments])

    def min(self):
        return max([s.max() for s in self.segments])

    def sum(self, axis=None, dtype=None):
        if self.fill_value == 0 or self.fill_value == False:
            return self[self.roi].sum(axis, dtype=dtype)

        if self.fill_value == True:
            in_roi = self[self.roi].sum(axis, dtype=dtype)
            roi_size = np.prod(get_shape_from_roi(self.roi, self.shape))
            size = np.prod(self.shape)
            return size-roi_size+in_roi

        return self.todense().sum(axis, dtype=dtype)

    def __eq__2(self, other):
        """Overrides the default implementation"""
        if isinstance(other, SegmentArray):
            assert len(self.shape) == len(other.shape)
            # if self.fill_value == other.fill_value:
            rng = tuple([slice(min(self.roi[i].start, other.roi[i].start), max(self.roi[i].stop, other.roi[i].stop)) for i in range(len(self.roi))])
            res = self[rng] == other[rng]
            new_fill_value = self.fill_value == other.fill_value
            # else:
            #     rng = tuple([slice(0, other.shape[i]) for i in range(len(other.shape))])
            #     res = self[rng] == other[rng]
            #     # return SegmentArray(res, voxelsize=self.voxelsize, shape=self.shape, spoint=get_spoint(self.roi),  fill_value=True, multi_part=True)
            #     new_fill_value = False

        elif isinstance(other, np.ndarray):
            rng = tuple([slice(0, other.shape[i]) for i in range(other.ndim)])
            res = self[rng] == other[rng]
            # return SegmentArray(res, voxelsize=self.voxelsize, shape=self.shape, spoint=get_spoint(self.roi),  fill_value=True, multi_part=True)
            new_fill_value = res.sum() > (res.size/2)
        else:
            rng = self.roi
            res = self[rng] == other
            new_fill_value = self.fill_value == other

          #

        return SegmentArray(res, voxelsize=self.voxelsize, shape=self.shape, spoint=get_spoint(rng),  fill_value=new_fill_value, multi_part=True, find_roi_if_shape_is_bigger=False)

    def __eq__(self, other, calc_roi=True, multi_part=True):
        return self._operator(operator.__eq__, other, calc_roi=calc_roi, multi_part=multi_part)

    def __ne__(self, other, calc_roi=True):
        return self._operator(operator.__ne__, other, calc_roi=calc_roi)

    def __and__(self, other, calc_roi=True, multi_part=True):
        return self._operator(operator.__and__, other, calc_roi=calc_roi, multi_part=multi_part)

    def __or__(self, other, multi_part=True):
        return self._operator(operator.__or__, other, calc_roi=False, multi_part=multi_part)

    def __xor__(self, other):
        return self._operator(operator.__xor__, other)

    def __invert__(self):
        assert self.dtype == bool
        return self._operator_single(np.invert)

    def __abs__(self):
        assert self.dtype == bool
        return self._operator_single(operator.__abs__)

    def __add__(self, other):
        return self._operator(operator.__add__, other)

    def __sub__(self, other):
        return self._operator(operator.__sub__, other)

    def __neg__(self):
        return self._operator_single(operator.__neg__)

    def __le__(self, other):
        return self._operator(operator.__le__, other)

    def __lt__(self, other):
        return self._operator(operator.__lt__, other)

    def __gt__(self, other):
        return self._operator(operator.__gt__, other)

    def __ge__(self, other):
        return self._operator(operator.__ge__, other)

    def _operator_single(self, opt):
        """Overrides the default implementation"""
        new_v = copy.deepcopy(self)

        new_v.fill_value = opt(self.fill_value)

        for s in new_v.segments:
            s.data = opt(s.data)
            s.fill_value = opt(s.fill_value)

        return new_v

    def _operator(self, opt, other, calc_roi=True, multi_part=True):
        """Overrides the default implementation"""
        if isinstance(other, SegmentArray):
            assert len(self.shape) == len(other.shape)
            rng = tuple([slice(min(self.roi[i].start, other.roi[i].start), max(self.roi[i].stop, other.roi[i].stop)) for i in range(len(self.roi))])
            res = opt(self[rng], other[rng])
            new_fill_value = opt(self.fill_value, other.fill_value)
        elif isinstance(other, np.ndarray):
            rng = tuple([slice(0, other.shape[i]) for i in range(other.ndim)])
            res = opt(self[rng], other[rng])
            # return SegmentArray(res, voxelsize=self.voxelsize, shape=self.shape, spoint=get_spoint(self.roi),  fill_value=True, multi_part=True)
            if res.dtype == bool:
                new_fill_value = res.sum() > (res.size/2)
            else:
                new_fill_value = 0
        else:
            rng = self.roi
            res = opt(self[rng], other)
            new_fill_value = opt(self.fill_value, other)

          #

        return SegmentArray(res, voxelsize=self.voxelsize, shape=self.shape, spoint=get_spoint(rng),  fill_value=new_fill_value, multi_part=multi_part, calc_roi=calc_roi)

    def __sizeof__(self) -> int:
        return np.sum([s.__sizeof__() for s in self.segments])+4

    def __repr__(self):
        orig_size = np.dtype(self.dtype).itemsize * np.prod(self.shape)
        mysize = self.__sizeof__()
        compress = 100-(mysize*100)//orig_size
        return f'SegmentArray({self.dtype}){self.shape}[compress={compress}% size={humanbytes(mysize)}/{humanbytes(orig_size)}] range= {roi_str(self.roi)}'

    def __str__(self):
        return self.__repr__()


def test(data=None, data_segment=None):
    if data is None:
        data = np.array(
            [[2, 2, 2, 0, 0, 3],
             [2, 2, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 1, 1, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1]])

    if data_segment is None:
        data_segment = SegmentArray(data)
    seg_dense = data_segment.todense()
    assert seg_dense.shape == data.shape
    assert np.all(data_segment[data_segment.roi] == data[data_segment.roi])
    assert data_segment[data_segment.roi].sum() == data[data_segment.roi].sum()
    assert np.all(seg_dense == data)
    assert data_segment.sum() == data.sum()

    assert np.all((data_segment == 1) == (data == 1))
    assert np.all((data_segment == 0) == (data == 0))
    np.random.seed(1)
    for ri in range(50):
        rnd_rng = tuple([slice(np.random.randint(0, data.shape[i]), np.random.randint(0, data.shape[i])) for i in range(data.ndim)])
        assert np.all(data_segment[rnd_rng] == data[rnd_rng]), f'failed in range {rnd_rng} ri={ri}'
    np.random.seed()
    # import pickle
    # assert len(pickle.dumps(data_segment)) < data.nbytes/2, f'size not reduced {len(pickle.dumps(data_segment))}/{data.nbytes}'

    pass


def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string."""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B, 'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B / KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B / MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B / GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B / TB)
