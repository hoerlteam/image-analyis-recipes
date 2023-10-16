# correct chromatic shift (by David H.)
import json
from pathlib import Path
import numpy as np
import pandas as pd


def augment_coords(coords):
    # helper function to add extra 4th column of 1s so we can just multipy with transform matrix
    return np.hstack((coords, np.ones_like(coords, shape=(len(coords), 1))))

def correct_chrom_shift(in_path,
                        out_path,
                        csv_string,
                        transforms_path,
                        reference_channel, 
                        coordinate_column_names_unit=None,
                        coordinate_column_names_pixel=None,
                        channel_column_name = 'channel',
                        pixel_size = None,
                        channel_aliases = {
                            '405 CSU-W1': '405-CSU-W1',
                            '488 CSU-W1': '488-CSU-W1',
                            '561 CSU-W1': '561 CSU-W1',
                            '640 CSU-W1': '640 CSU-W1'}
                       ):
    
    ##### formatting and parameters #####
    # how to name columns that are added to tables
    column_name_suffix = ''
    reference_channel_column_name = 'shift_reference_channel'

    # check if enough information is given, raise Error otherwise
    if coordinate_column_names_pixel is None and coordinate_column_names_unit is None:
        raise ValueError('Please specify either pixel or unit columns to transform (or both)')
    if pixel_size is None and coordinate_column_names_unit is None:
        raise ValueError('You need to specify pixel size if only pixel coordinates are given')
      
    
    ##### open file and reshape #####
    with open(transforms_path) as fd:
        transform_info = json.load(fd)

    # transforms are saved as list of dicts containing channel pair and (flat) parameters
    # build dict channel pair -> transform matrix
    transforms = {}
    for transform_info_i in transform_info['transforms']:

        tr = np.array(transform_info_i['parameters']).reshape(4,4)

        # apply channel renaming if necessary
        channels = map(lambda c: channel_aliases[c] if c in channel_aliases else c, transform_info_i['channels'])

        transforms[tuple(channels)] = tr

    in_files = sorted(in_path.glob(csv_string))
    
    ##### correct shift #####
    # make out path if it does not exist already
    if not out_path.exists():
        out_path.mkdir()

    for in_file in in_files:

        df = pd.read_csv(in_file)

        # automatically determine pixel size if both pixel and unit coordinates are present
        # (we just use the first row, as we can assue it to be the same for every spot)
        pixel_size_was_none = False
        if pixel_size is None:
            pixel_size = (df[coordinate_column_names_unit].values / df[coordinate_column_names_pixel].values)[0]
            pixel_size_was_none = True

        dfs = []
        for ch, dfi in df.groupby(channel_column_name):

            # get the transform from current to reference channel
            tr = transforms[(ch, reference_channel)]

            # get coordinates to transform, 2 options:
            # 1. if we have unit columns, use those
            # 2. if we only have pixel columns, use those, multiply with pixel size
            if coordinate_column_names_unit is not None:
                coords = augment_coords(dfi[coordinate_column_names_unit].values).T
            else:
                coords = augment_coords(dfi[coordinate_column_names_pixel].values * pixel_size).T

            # transform
            coords_transformed = tr @ coords

            # add transformed coordinates in world coordinate units
            if coordinate_column_names_unit is not None:
                for i, dim_name in enumerate(coordinate_column_names_unit):
                    dfi[dim_name + column_name_suffix] = coords_transformed[i]

            # add transformed pixel coordinates
            if coordinate_column_names_pixel is not None:
                for i, dim_name in enumerate(coordinate_column_names_pixel):
                    dfi[dim_name + column_name_suffix] = coords_transformed[i] / pixel_size[i]

            # add reference channel
            dfi[reference_channel_column_name] = reference_channel
            dfs.append(dfi)

        # combine corrected dfs
        df_corrected = pd.concat(dfs).sort_index()

        # save
        out_file = out_path / (in_file.stem + '_shift-corrected.csv')
        df_corrected.to_csv(out_file, index=False)

        # reset pixel size so it is read again for the next table
        if pixel_size_was_none:
            pixel_size = None   

