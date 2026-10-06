from nd2 import ND2File
from pathlib import Path
import warnings
from calmutils.imageio import xarray_to_tiffs


SUPPORTED_PROEJCTIONS = ["max", "sum", "min", "mean"]


def resave_nd2_flexible(
    in_path,
    file_pattern="[!.]*.nd2",
    out_path=None,
    split_dimensions=("C", "T", "P"),
    prefixes={"C": "_ch", "T": "_tp", "P": "_pos", "Z": "_z", "X": "_x", "Y": "_y"},
    use_indices=True,
    min_index_len=1,
    projection=None,
    projection_dimension="Z",
):
    # by default, save in "tif" subfolder
    if out_path is None:
        out_path = Path(in_path) / 'tif'
    # make out_path a pathlib path in any case
    out_path = Path(out_path)

    # create if it does not yet exist
    if not out_path.exists():
        out_path.mkdir()

    for in_file in Path(in_path).glob(file_pattern):
        resave_nd2_flexible_single(
            in_file,
            out_path,
            split_dimensions,
            prefixes,
            use_indices,
            min_index_len,
            projection,
            projection_dimension
        )


def handle_projection_xarray(img, projection, projection_dimension):

    # nothing to do
    if projection is None:
        return img
    
    # catch projection along nonexistent dimension -> warn and return original
    if projection is not None and projection_dimension not in img.dims:
        warnings.warn(f"Projection along dimension {projection_dimension}")
        return img

    # apply projection along projection axis
    if projection is not None:
        if projection == "max":
            return img.max(projection_dimension)
        elif projection == "min":
            return img.min(projection_dimension)
        elif projection == "sum":
            return img.sum(projection_dimension)
        elif projection == "mean":
            return img.mean(projection_dimension)
        else:
            raise RuntimeError(f"Unsupported projection type '{projection}'. Supported projection types: {SUPPORTED_PROEJCTIONS}.")


def resave_nd2_flexible_single(
    in_file,
    out_path,
    split_dimensions=("C", "T", "P"),
    prefixes={"C": "_ch", "T": "_tp", "P": "_pos", "Z": "_z", "X": "_x", "Y": "_y"},
    use_indices=True,
    min_index_len=1,
    projection=None,
    projection_dimension="Z",
):

    # read file to XArray
    with ND2File(in_file) as reader:
        img = reader.to_xarray()
        pixel_size = list(reader.voxel_size())[::-1]

    img = handle_projection_xarray(img, projection, projection_dimension)

    xarray_to_tiffs(
        img,
        out_path,
        in_file.stem,
        split_dimensions,
        prefixes,
        use_indices,
        min_index_len,
        pixel_size
    )
