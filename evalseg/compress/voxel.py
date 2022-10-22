import numpy as np


class Voxel(np.ndarray):
    def __new__(cls, ndarray):
        if ndarray.ndim != 3:
            raise Exception("Only 3d is supported")
        return ndarray.view(Voxel)

    def __getstate__(self):
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".avi") as temp:
            import cv2

            codec = cv2.VideoWriter.fourcc(*"WebP")
            vw = cv2.VideoWriter(
                temp.name, p, codec, 1, (self.shape[0], self.shape[1]), False
            )
            for z in range(self.shape[2]):
                img = self[:, :, z].astype(np.uint8)
                vw.write(img)
            vw.release()
            with open(temp.name, "rb") as f:
                return f.read()

    def __setstate__(self, d):
        print("set state", d)
        with tempfile.NamedTemporaryFile() as temp:
            temp.write(d)
            cap = cv2.VideoCapture(temp.name)
            allf = []
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    allf.append(frame)
                else:
                    break
            res = np.zeros(allf[0].shape + (len(allf),))
            for i in len(allf):
                res[:, :, i] = (
                    (allf[i][:, :, 0] << 16)
                    | (allf[i][:, :, 1] << 8)
                    | allf[i][:, :, 2]
                )


def convert2png_array(img3d):
    import tempfile

    dmin = float(img3d.min())
    dmax = float(img3d.max())
    bstr = [(dmin, dmax, img3d.shape)]
    data = ((img3d - dmin) * 2**16 / (dmax - dmin)).astype(np.uint16)

    for z in range(0, data.shape[2]):
        img = data[:, :, z]
        bstr.append(
            cv2.imencode(
                ".png",
                img.astype(np.uint16),
                [int(cv2.IMWRITE_PNG_COMPRESSION), 1],
            )[1].tobytes()
        )
    return bstr


def read_from_png_array(bs):
    dmin, dmax, shape = bs.pop(0)
    res = np.zeros(shape)
    # print(res.shape,allf[0].shape)
    for i in range(shape[2]):
        # buff =
        res[:, :, i] = cv2.imdecode(np.frombuffer(bs[i], np.uint8), -1)

    # res/=(2**16)
    return res
