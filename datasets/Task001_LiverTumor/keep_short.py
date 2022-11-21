"""
This script do gunzip and create a zip file
with a .keepkaggle file with password So kaggle
will not unzip this file 
"""
import os
import gzip
import zipfile
import time
from tqdm.auto import tqdm
from zipfile import ZipFile, ZIP_BZIP2


first_path = os.getcwd()
available_ids = {}
with ZipFile('Predictions.zip', "r") as preds:
    for n in preds.namelist():
        available_ids[n.split('/')[-1]] = available_ids.get(n.split('/')[-1], 0)+1

print(available_ids)


def keep_only(zipf, ids):
    old = zipf.replace('.zip', '_full.zip')
    os.rename(zipf, old)

    with ZipFile(old) as archive:
        with ZipFile(zipf, 'a', ZIP_BZIP2, compresslevel=5) as archive_new:
            for n in available_ids:
                if n == '.keepkaggle':
                    continue
                if n in archive.namelist():
                    archive_new.writestr(n, archive.read(n))

        with open('.keepkaggle', 'w') as f:
            pass
        os.system(f"7z.exe a -tzip -p2022 -mm=BZip2 {zipf} .keepkaggle")
        os.remove('.keepkaggle')


keep_only('CT.zip', available_ids)
# keep_only('GroundTruth.zip',available_ids)
