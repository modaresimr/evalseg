import nibabel as nib
import numpy as np


def read_nib(f):
    try:
        if type(f) is str:
            nib_data = nib.load(f)
        else:
            from io import BytesIO

            fh = nib.FileHolder(fileobj=BytesIO(f))
            nib_data = nib.Nifti1Image.from_file_map(
                {"header": fh, "image": fh}
            )

        data = nib_data.dataobj[...]
        voxelsize = np.array(nib.affines.voxel_sizes(nib_data.affine))
        return data, voxelsize
    except:
        print(f"reading {f} failed!")
        return None, None
