import warnings
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.transform import resize
from nd2 import ND2File
from msr_reader import OBFFile

from calmutils.misc.visualization import get_orthogonal_projections_8bit

def load_multichannel_nd2(file_path):
    with ND2File(file_path) as reader:
        # nice OC name without whitespace
        channel_names = map(lambda s: s.channel.name.strip().replace(' ', '-'), reader.metadata.channels)
        # to cyzx
        # NOTE: needs testing for different dimensionality files
        imgs = reader.asarray().transpose((1,0,2,3))    
        # invert xyz voxel size to zyx to match img array
        pixel_size = reader.voxel_size()[::-1]
    
    return imgs, channel_names, pixel_size

def load_multistack_multichannel_nd2(file_path):
    with ND2File(file_path) as reader:
        # nice OC name without whitespace
        channel_names = map(lambda s: s.channel.name.strip().replace(' ', '-'), reader.metadata.channels)
        # to cyzx
        # NOTE: needs testing for different dimensionality files 
        imgs = reader.asarray().transpose((0,2,1,3,4))
        # invert xyz voxel size to zyx to match img array
        pixel_size = reader.voxel_size()[::-1]
    
    return imgs, channel_names, pixel_size

def load_msr(file):
    with OBFFile(file) as f:
            
        imgs = []
        channel_names = []

        for idx in range(0,len(f.sizes)):

            # reading image data
            img = f.read_stack(idx) # read stack with index idx into numpy array
            imgs.append(img)

            # metadata
            stack_sizes = f.sizes # list of stack sizes/shapes, including stack and dimension names
            pixel_sizes = f.pixel_sizes # like sizes, but with pixel sizes (unit: meters)
            channel_names.append(pixel_sizes[idx].name)
            pixel_sizes = [pixel_sizes[idx].sizes['ExpControl Z'],
                           pixel_sizes[idx].sizes['ExpControl Y'],
                           pixel_sizes[idx].sizes['ExpControl X']]

        imgs = np.asarray(imgs)

        return(imgs, channel_names, pixel_sizes)
