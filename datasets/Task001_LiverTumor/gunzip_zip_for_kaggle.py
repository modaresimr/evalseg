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

def recursive_action(file_folder,archive):
   if os.path.isfile(file_folder):
        ungzip_zip(file_folder,archive)  
   else:
        for f in tqdm(os.listdir(file_folder)):
            recursive_action(f'{file_folder}/{f}',archive)
        try:
            os.removedirs(file_folder)
        except:
            pass

def ungzip_zip(gzfile,archive):
    target=gzfile.replace('.gz','')
    for x in ['train_','_0000','pancreas_','multiorgan_']:
        target=target.replace(x,'')
    if '.gz' in gzfile:
        with gzip.open(gzfile,'rb') as r:
            with open(target,'wb') as w:
                w.write(r.read())
        os.remove(gzfile)
    elif '.nii' in target:
        os.rename(gzfile,target)
        
    if '.nii' in target:
        
        archive.write(target,arcname=target.replace(target.split('/')[0]+"/",''))
        # while os.path.exists(zipfile) and not os.access(zipfile,os.R_OK|os.X_OK):
        #     print('file is open... wait to be closed')
        #     time.sleep(.1)
        # os.system(f"7z.exe a -tzip -mm=BZip2 {zipfile} {target}")
        # while os.path.exists(f'{zipfile}.tmp'):
        #     print('waiting to finish ziping')
        #     time.sleep(.1)
        os.remove(target)


first_path= os.getcwd()
for current in os.listdir('.'):
    if os.path.isdir(current):# and current!='CT':
        dst=f'{current}.zip'
        with ZipFile(dst, "a", ZIP_BZIP2, compresslevel=5) as archive:
            recursive_action(current,archive)
        
        print('adding .keepkaggle file for avoiding kaggle to unzip data')
        with open('.keepkaggle', 'w') as f: 
            pass 
        os.system(f"7z.exe a -tzip -p2022 -mm=BZip2 {dst} .keepkaggle")
        os.remove('.keepkaggle')
        
        
