import os
import sys
import json
from glob import glob
import numpy as np
from natsort import natsorted

import tifffile
from nd2reader import ND2Reader
from skimage.transform import resize
from nd2 import ND2File
import h5py as h5
from msr import OBFFile

############# h5 files #################
# Split h5 files into individual files and channels, create a folder called "tif" and save them there
def resave_h5(folder):

    h5s = glob(folder+"*.h5")
    h5s = natsorted(h5s)

    # create out "tif" directory if it doesnt exist yet
    out = f"{folder}/tif"
    os.makedirs(out, exist_ok=True)

    # cycle through:
    for h5_file_path in h5s: # all h5s in folder
        name = os.path.splitext(os.path.basename(h5_file_path))[0]

        with h5.File(h5_file_path, 'r') as fd:

            for key in fd['experiment'].keys(): # all images in h5 file
                metadata = json.loads(fd[f'experiment/{key}/0'].attrs['measurement_meta'])

                for channel in range(len(fd[f'experiment/{key}/0/'])): # all channels in image
                    img = fd[f'experiment/{key}/0/{channel}']
                    tifffile.imsave(f"{out}/{name}_{key}_ch{channel}.tif", img[channel]) # save
                    

############# nd2 files #################
# reads all nd2files and resaves them as tif        
def resave_nd2(nd2files):
    
    nd2files_paths = glob(nd2files + "/raw/*nd2")
#     nd2files_paths = glob(nd2files + "/*nd2")
    
    # create out "tif" directory if it doesnt exist yet
    out = f"{nd2files}/tif/"
    os.makedirs(out, exist_ok=True)
    
    for nd2_file in nd2files_paths:
        with ND2Reader(nd2_file) as img:
            
            img.bundle_axes = ['z', 'y', 'x', 'c']
            img = np.array(img[0])
            name= os.path.basename(nd2_file).rsplit(".", 1)[0]
            
            for ch in range(img.shape[3]):
                tifffile.imsave(f"{nd2files}/tif/{name}_ch{ch}.tif", img[:, :, :, ch].astype(np.uint16))
                
 
####### moredimensional nd2 ###############
# reads all nd2 files where 1 file contains several stacks (created by the JOBS script) and resaves them as .tif
# def resave_auto_nd2(nd2files):
    
#     nd2files_paths = glob(nd2files + "/raw/*nd2")
    
#     # create out "tif" directory if it doesnt exist yet
#     out = f"{nd2files}/tif/"
#     os.makedirs(out, exist_ok=True)

#     for nd2_file in nd2files_paths:
#         with ND2Reader(nd2_file) as img:
#             img.bundle_axes = ['z', 'y', 'x', 'c', 'v']
#             img = np.array(img[0])

#             for field in range(img.shape[4]): 
#                 base_name = os.path.basename(nd2_file).rsplit(".", 1)[0]
#                 name = f"{base_name}_field{field}"

#                 for ch in range(img.shape[3]):
#                     tifffile.imsave(f"{nd2files}/tif/{name}_ch{ch}.tif", img[:, :, :, ch, field].astype(np.uint16))
                    
                    
#### updated version
def resave_auto_nd2(nd2files):
    
    nd2files_paths = glob(nd2files + "/*nd2")
    
    # create out "tif" directory if it doesnt exist yet
    out = f"{nd2files}/tif/"
    os.makedirs(out, exist_ok=True)

    for nd2_file in nd2files_paths:
        with ND2File(nd2_file) as reader:
            # NOTE: needs testing for different dimensionality files
            img = reader.asarray().transpose((0,2,1,3,4))
            
            for field in range(img.shape[0]): 
                base_name = os.path.basename(nd2_file).rsplit(".", 1)[0]
                name = f"{base_name}_field{field}"

                for ch in range(img.shape[1]):
                    tifffile.imsave(f"{nd2files}/tif/{name}_ch{ch}.tif", img[field, ch, :, :, :].astype(np.uint16))


################################################
# reads all msr files and resaves them as tif 
def resave_msr(folder,out):

    files = glob(f"{folder}/raw/*.msr")

    # create out "tif" directory if it doesnt exist yet
    out = f"{folder}/tif"
    os.makedirs(out, exist_ok=True)

    for file in files:

        with OBFFile(file) as f:

            for idx in range(0,len(f.shapes)):
                
                # reading image data
                img = f.read_stack(idx) # read stack with index idx into numpy array

                # metadata
                stack_sizes = f.sizes # list of stack sizes/shapes, including stack and dimension names
                pixel_sizes = f.pixel_shapes # like sizes, but with pixel sizes (unit: meters)
                
                # save
                name = os.path.splitext(os.path.basename(file))[0]
                tifffile.imsave(f"{out}/{name}_ch{idx}.tif", img) 

                # TODO save metadata, like pixel sizes as well
    #            resolution=(pixel_sizes[idx].sizes['ExpControl X']*1e+6, 
    #            pixel_sizes[idx].sizes['ExpControl Y']*1e+6, 'None')) 

################################################
# delete tif files after analysis is done to save space
def remove_tifs(folder):
    if "tif" in folder:
        try:
            shutil.rmtree(folder)
            print(f"Removed '{folder}'.")
        except OSError as e:
            print(f"Error removing folder '{folder}': {e}")
    else:
        user_input = input(f"Are you sure you want to delete '{folder}'? (yes/no): ")
        if user_input.lower() == "yes":
            try:
                shutil.rmtree(folder)
                print(f"Folder '{folder}' and its contents removed successfully.")
            except OSError as e:
                print(f"Error removing folder '{folder}': {e}")
        else:
            print("Aborted.")