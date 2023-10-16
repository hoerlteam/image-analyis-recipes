import os
import numpy as np
import pandas as pd
from glob import glob
import re

from skimage.io import imsave
from skimage.morphology import label
from skimage.segmentation import clear_border
from skimage.measure import regionprops

import imageio
import matplotlib.pyplot as plt
from skimage.io import imread
from scipy.optimize import linear_sum_assignment
from itertools import product
from collections import defaultdict


# assigns each spot to a cell and filters spots.csv for spots in cells
def add_cell_info(masks,path_spots,out,filter=True,mask_ending="_cp_masks"):
    
    df = pd.read_csv(path_spots)
        
    df_list = []
    
    for file in masks:
        # subset spots for spots in current image
        name = file.split("/")[-1].split(".")[0]
        name = name.replace(mask_ending, "")
        name = re.sub(r'_ch\d+', '', name)
        subset_df = df[df['img'].str.contains(name)]
        
        # load mask
        file_type = os.path.splitext(file)[1]
        
        if file_type == ".npy":
            mask = np.load(file,allow_pickle=True).item()['masks']
        elif file_type == ".png":
            mask = imread(file)
        else:
            print("Please input valid segmentation masks (.npy and .png supported).")
        
        # label all cells and remove cells on edges
        labelled_mask = label(mask)
        cleared_mask = clear_border(labelled_mask)

        
    # add cell info to spots
        # for 2d masks
        if len(cleared_mask.shape) == 2:            
            cell = labelled_mask[subset_df['y'].astype(int), subset_df['x'].astype(int)]
            subset_df.insert(1, 'cell', cell)

            # info about whether spot is in cell on border
            spot_in_cleared_mask = cleared_mask[subset_df['y'].astype(int), subset_df['x'].astype(int)] != 0
            subset_df.insert(2, 'whole_cell', spot_in_cleared_mask)

            df_list.append(subset_df)
            
        # for 3d masks
        elif len(cleared_mask.shape) == 3:
            cell = labelled_mask[subset_df['z'].astype(int), subset_df['y'].astype(int), subset_df['x'].astype(int)]
            subset_df.insert(1, 'cell', cell)

            # info about whether spot is in cell touching border
            spot_in_cleared_mask = cleared_mask[subset_df['z'].astype(int), subset_df['y'].astype(int), subset_df['x'].astype(int)] != 0
            subset_df.insert(2, 'whole_cell', spot_in_cleared_mask)

            df_list.append(subset_df)
            
    spots = pd.concat(df_list, ignore_index=True)
    
    # remove all spots without a cell (0)
    if filter == True:
        spots.drop(spots[spots['cell'] == 0].index, inplace=True)
    
    spots.to_csv(out, index=False)
    
# calculates number of spots per cell (sensitivity)
def get_sensitivity(masks,path_spots,out,mask_ending="_cp_masks"):
    
    df = pd.read_csv(path_spots)
        
    df_list = []
    
    for file in masks:
        # subset spots for spots in current image
        name = file.split("/")[-1].split(".")[0]
        name = name.replace(mask_ending, "")
        name = re.sub(r'_ch\d+', '', name)
        subset_df = df[df['img'].str.contains(name)]
        
        # load mask
        file_type = os.path.splitext(file)[1]
        
        if file_type == ".npy":
            mask = np.load(file,allow_pickle=True).item()['masks']
        elif file_type == ".png":
            mask = imread(file)
        else:
            print("Please input valid segmentation masks (.npy and .png supported).")
        
        # label all cells and remove cells on edges
        labelled_mask = label(mask)
#         cleared_mask = clear_border(labelled_mask)
        cleared_mask = labelled_mask

        
        # add cell info to spots
        # for 2d masks
        if len(cleared_mask.shape) == 2:            
            cell = labelled_mask[subset_df['y'].astype(int), subset_df['x'].astype(int)]
            subset_df.insert(1, 'cell', cell)
            
        # for 3d masks
        elif len(cleared_mask.shape) == 3:
            cell = labelled_mask[subset_df['z'].astype(int), subset_df['y'].astype(int), subset_df['x'].astype(int)]
            subset_df.insert(1, 'cell', cell)

        # add number of spots in each cell
        cell_df = pd.DataFrame({'cell': np.unique(cleared_mask)})
        spots = subset_df.groupby(['img','cell']).size().reset_index(name='count')
        cell_df = spots.merge(cell_df, on='cell', how='outer')
        
        cell_df['count'].fillna(0, inplace=True)
        cell_df['img'].fillna(subset_df['img'].unique()[0], inplace=True)
        
        # measure cell sizes using regionprops
        region_props = regionprops(cleared_mask)
        cell_sizes = [[prop.label, prop.area] for prop in region_props]
        cell_sizes = pd.DataFrame(cell_sizes, columns=['cell', 'cell_size'])
        cell_df = cell_df.merge(cell_sizes, on='cell', how='outer')
        
        df_list.append(cell_df)
      
    spots_per_cell = pd.concat(df_list, ignore_index=True)
    
    # remove too small cells (faulty segmentation)
    spots_per_cell = spots_per_cell[spots_per_cell['cell_size'] > 50000]
    
    # get sensitivity
    spots_per_cell = spots_per_cell[spots_per_cell.cell != 0]
    
    spots_per_cell.to_csv(out, index=False)
    

# tries to match spot pairs in 2 different channels and outputs pairwise distances
def detect_spot_pairs(path, out, ch, voxel_size=(300, 130, 130)):
    df = pd.read_csv(path)
    df['img'] = df['img'].apply(lambda x: x.rsplit('_', 1)[0])
    
    result = defaultdict(list)
    voxel_size = np.array(voxel_size)
    
    # group the DataFrame by 'img' 
    grouped = df.groupby(['img'])

    for (img), group_df in grouped:
        spot_coords_ch1 = group_df.loc[group_df['channel'] == ch[0], ['z', 'y', 'x']].values
        spot_coords_ch2 = group_df.loc[group_df['channel'] == ch[1], ['z', 'y', 'x']].values
        distances = np.zeros((len(spot_coords_ch1), len(spot_coords_ch2)))

        for (i1,c1), (i2, c2) in product(enumerate(spot_coords_ch1), enumerate(spot_coords_ch2)):
            distances[i1, i2] = np.linalg.norm((c1 - c2)*voxel_size) # np.sqrt(np.sum((c1 - c2)**2))

        row_ind, col_ind = linear_sum_assignment(distances)
        
        for ri, ci in zip(row_ind, col_ind):
            result['img'].append(img)
            result['distance_um'].append(distances[ri,ci])
            
            for dim_i, dim in enumerate('zyx'):
                result[f'{dim}_1'].append(spot_coords_ch1[ri][dim_i])
                result[f'{dim}_2'].append(spot_coords_ch2[ci][dim_i])

    result_df = pd.DataFrame(result)
    
    # add acquisition info and reshape
    df = df.drop(columns=['c','t'])
    right = ["img","x","y","z"] 
    left1 =  ["img","x_1","y_1","z_1"] 
    left2 = ["img","x_2","y_2","z_2"] 
    
    result_df = result_df.merge(df, left_on=left1, right_on=right ,how='left')
    result_df = result_df.merge(df, left_on=left2, right_on=right ,how='left',suffixes=('_1', '_2'))
    
    result_df = result_df.T.drop_duplicates().T
    
    # save to csv
    result_df.to_csv(out, index=False)