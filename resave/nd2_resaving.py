import ast
from nd2 import ND2File
from pathlib import Path
import warnings
from calmutils.imageio import xarray_to_tiffs


SUPPORTED_PROEJCTIONS = ["max", "sum", "min", "mean"]


def _as_list(patterns):
    if patterns is None:
        return []
    if isinstance(patterns, (str, Path)):
        return [patterns]
    return list(patterns)


def resave_nd2_flexible(
    in_path,
    file_pattern="[!.]*.nd2",
    exclude_pattern=None,
    out_path=None,
    split_dimensions=("C", "T", "P"),
    prefixes={"C": "_ch", "T": "_tp", "P": "_pos", "Z": "_z", "X": "_x", "Y": "_y"},
    use_indices=True,
    min_index_len=1,
    projection=None,
    projection_dimension="Z",
    pixel_size_scale=1.0,
):
    # by default, save in "tif" subfolder
    if out_path is None:
        out_path = Path(in_path) / 'tif'
    # make out_path a pathlib path in any case
    out_path = Path(out_path)

    # create if it does not yet exist
    if not out_path.exists():
        out_path.mkdir()

    files = {f for pat in _as_list(file_pattern) for f in Path(in_path).glob(pat)}
    excluded = {f for pat in _as_list(exclude_pattern) for f in Path(in_path).glob(pat)}
    files = sorted(files - excluded)
    if not files:
        warnings.warn(f'no files matching {file_pattern!r} (minus {exclude_pattern!r}) in {in_path}')
        return

    for in_file in files:
        resave_nd2_flexible_single(
            in_file,
            out_path,
            split_dimensions,
            prefixes,
            use_indices,
            min_index_len,
            projection,
            projection_dimension,
            pixel_size_scale
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
    pixel_size_scale=1.0,
):

    # read file to XArray
    with ND2File(in_file) as reader:
        img = reader.to_xarray()
        pixel_size = list(reader.voxel_size())[::-1]

    # parse pixel size scale as string literal
    if isinstance(pixel_size_scale, str):
        pixel_size_scale = ast.literal_eval(pixel_size_scale)
    
    # broadcast scalar pixel size scale, ensure len-3 float sequence
    if isinstance(pixel_size_scale, (int, float)):
        scale = [float(pixel_size_scale)] * 3
    else:
        scale = [float(s) for s in pixel_size_scale]
        if len(scale) == 1:
            scale *= 3
        elif len(scale) != 3:
            raise ValueError(f'pixel_size_scale must be a scalar or a [z,y,x] sequence '
                             f'of length 3, got {pixel_size_scale!r}')
    
    # adjust pixel size if necessary
    if scale != [1.0, 1.0, 1.0]:
        pixel_size = [p * s for p, s in zip(pixel_size, scale)]

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
