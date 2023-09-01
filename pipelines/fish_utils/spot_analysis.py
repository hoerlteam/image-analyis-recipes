import os
import numpy as np
import pandas as pd
from glob import glob

from skimage.io import imsave
from skimage.morphology import label
from skimage.segmentation import clear_border

import imageio
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.optimize import linear_sum_assignment
from itertools import product
from collections import defaultdict


# assigns each spot to a cell 
def add_cell_info(path,out,filter=True,mask_ending="_cp_masks"):
    
    masks_path = f"{path}/segmentation/"
    masks = glob(f"{masks_path}*.png")
    df = pd.read_csv(f"{path}/detections/merge.csv")
    
    df_list = []
    
    for file in masks:
        # subset spots for spots in current image
        name = file.split("/")[-1].split(".")[0]
        name = name.replace(mask_ending, "")
        subset_df = df[df['img'].str.contains(name)]
        mask = imread(file)
        
        # label all cells and remove cells on edges
        labelled_mask = label(mask)
        cleared_mask = clear_border(labelled_mask)
        
        # add cell info to spots
        cell = labelled_mask[subset_df['y'].astype(int), subset_df['x'].astype(int)]
        subset_df.insert(1, 'cell', cell)
        
        # info about whether spot is in cell on border
        spot_in_cleared_mask = cleared_mask[subset_df['y'].astype(int), subset_df['x'].astype(int)] != 0
        subset_df.insert(2, 'whole_cell', spot_in_cleared_mask)
        
        df_list.append(subset_df)
        
    spots = pd.concat(df_list, ignore_index=True)
    
    # remove all spots without a cell (0)
    if filter == True:
        spots.drop(spots[spots['cell'] == 0].index, inplace=True)
    
    spots.to_csv(out, index=False)

    
# calculates number of spots per cell (sensitivity)
def get_sensitivity(path,out):
    df = pd.read_csv(path)
    df = df[df['whole_cell'] == True]
    summarized_df = df.groupby(['img', 'cell', 'channel']).size().reset_index(name='nspot_per_cell')
    summarized_df.to_csv(out, index=False)  


# tries to match spot pairs in 2 different channels and outputs pairwise distances
def detect_spot_pairs(path, ch, voxel_size=(300, 130, 130)):
    df = pd.read_csv(f"{path}/stats_spots.csv")
    df['img'] = df['img'].apply(lambda x: x.rsplit('_', 1)[0])
    
    result = defaultdict(list)
    voxel_size = np.array(voxel_size)
    
    # Group the DataFrame by 'img' 
    grouped = df.groupby(['img'])

    for (img), group_df in grouped:
        spot_coords_ch1 = group_df.loc[group_df['channel'] == ch[0], ['x', 'y', 'z']].values
        spot_coords_ch2 = group_df.loc[group_df['channel'] == ch[1], ['x', 'y', 'z']].values
        distances = np.zeros((len(spot_coords_ch1), len(spot_coords_ch2)))

        for (i1,c1), (i2, c2) in product(enumerate(spot_coords_ch1), enumerate(spot_coords_ch2)):
            distances[i1, i2] = np.linalg.norm((c1 - c2)*voxel_size) # np.sqrt(np.sum((c1 - c2)**2))

        row_ind, col_ind = linear_sum_assignment(distances)
        
        for ri, ci in zip(row_ind, col_ind):
            result['img'].append(img)
            result['distance_nm'].append(distances[ri,ci])
            
            for dim_i, dim in enumerate('zyx'):
                result[f'pos_{dim}_ch1'].append(spot_coords_ch1[ri][dim_i])
                result[f'pos_{dim}_ch2'].append(spot_coords_ch2[ci][dim_i])

    result_df = pd.DataFrame(result)
    result_df.to_csv(f"{path}/distances.csv", index=False)
    
    return result_df