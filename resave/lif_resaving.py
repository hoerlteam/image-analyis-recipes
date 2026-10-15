from pathlib import Path
import warnings

import liffile
from calmutils.imageio import xarray_to_tiffs


SUPPORTED_PROJECTIONS = ["max", "sum", "min", "mean"]


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
            raise RuntimeError(f"Unsupported projection type '{projection}'. Supported projection types: {SUPPORTED_PROJECTIONS}.")


def get_pixel_size_micron(img):
    """Pixel/voxel size (z, y, x) in microns, guessed from xarray coordinate spacing.

    Note: liffile returns coordinates in meters, so they are rescaled to microns.
    Z coordinates can be descending (top-down acquisition), so absolute values are used.
    """
    import numpy as np

    def spacing(dim):
        if dim in img.coords:
            return abs(np.diff(img.coords[dim].values).mean()) * 1e6
        return 1

    return [spacing(dim) for dim in ("Z", "Y", "X")]


def resave_lif_flexible_single(
    in_file,
    out_path,
    split_dimensions=("C", "T", "P"),
    prefixes={"C": "_ch", "T": "_tp", "P": "_pos", "Z": "_z", "X": "_x", "Y": "_y"},
    use_indices=True,
    min_index_len=1,
    projection=None,
    projection_dimension="Z",
):
    # a .lif file can contain multiple images (e.g. single acquisitions and Z-stacks)
    with liffile.LifFile(in_file) as reader:
        for image in reader.images:
            # read image to XArray
            img = image.asxarray()

            img = handle_projection_xarray(img, projection, projection_dimension)

            # concatenate original file name and internal image name
            # (a .lif file can contain multiple images, possibly with nested names)
            image_name = image.name.replace("/", "_").replace("\\", "_")
            out_prefix = f"{in_file.stem}_{image_name}"

            xarray_to_tiffs(
                img,
                out_path,
                out_prefix,
                split_dimensions,
                prefixes,
                use_indices,
                min_index_len,
                get_pixel_size_micron(img),
            )


def resave_lif_flexible(
    in_path,
    file_pattern="[!.]*.lif",
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
        resave_lif_flexible_single(
            in_file,
            out_path,
            split_dimensions,
            prefixes,
            use_indices,
            min_index_len,
            projection,
            projection_dimension
        )
