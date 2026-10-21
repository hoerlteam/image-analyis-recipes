"""Quick 8bit PNG visualizations of microscopy images stored as xarrays.

Generic part: for each stack (group of non-spatial dimensions) of an input xarray,
create orthogonal projections (3D data) or a plain intensity-normalized image (2D data),
optionally compose channels to a single RGB image, and save as PNG.

LIF part: iterate over all images contained in all ```.lif``` files of an input folder
(each ```.lif``` file can contain multiple independent images, e.g. single acquisitions
and Z-stacks) and visualize each of them.
"""

from pathlib import Path
import warnings

import liffile
import numpy as np
from skimage.io import imsave

from calmutils.color import gray_images_to_rgb_composite
from calmutils.color.color import DEFAULT_COLOR_NAMES
from calmutils.misc.visualization import get_orthogonal_projections_8bit

from projection import normalize_intensity_to_8bit


# prefixes for grouping dimensions in output filenames (LIF M and ND2 P for multi-position)
DEFAULT_PREFIXES = {"C": "_ch", "T": "_tp", "M": "_pos", "P": "_pos"}


def get_pixel_size_micron(img):
    """Voxel size (z, y, x) in microns, from xarray coordinate spacing.

    Note: liffile returns coordinates in meters, so they are rescaled to microns.
    Z coordinates can be descending (top-down acquisition), so absolute values are used.
    """

    def spacing(dim):
        if dim in img.coords:
            return abs(np.diff(img.coords[dim].values).mean()) * 1e6
        return 1

    return [spacing(dim) for dim in ("Z", "Y", "X")]


def project_and_normalize_8bit(
    img,
    pixel_size,
    projection_type="max",
    intensity_range="auto",
    auto_range_quantiles=(0.02, 0.9995),
):
    """Orthogonal projections (scaled by pixel size) of 3D stacks, plain normalization for 2D images."""

    if img.ndim > 2:
        return get_orthogonal_projections_8bit(
            img,
            pixel_size=pixel_size,
            projection_type=projection_type,
            intensity_range=intensity_range,
            auto_range_quantiles=auto_range_quantiles,
        )
    return normalize_intensity_to_8bit(
        img,
        intensity_range=intensity_range,
        auto_range_quantiles=auto_range_quantiles,
    )


def _sanitize(name):
    """Make a channel or image name safe to use in a filename."""
    return str(name).strip().replace(" ", "-").replace("/", "_").replace("\\", "_")


def visualize_image_xarray(
    img_xa,
    pixel_size,
    out_dir,
    out_prefix,
    intensity_range="auto",
    auto_range_quantiles=(0.02, 0.9995),
    projection_type="max",
    do_color=False,
    color_dimension_label=("C", "S"),
    visualization_colors=DEFAULT_COLOR_NAMES,
    prefixes=None,
    use_channel_names=False,
    use_indices=True,
    min_index_len=1,
):
    """Create 8bit PNG visualizations of all image stacks in an xarray.

    Parameters
    ----------
    img_xa: xarray with image data. All dimensions except Z, Y, X (and the RGB
        dimension for color composites) are treated as separate output images.
    pixel_size: voxel size (z, y, x), used for anisotropy correction of orthogonal projections.
    out_dir: directory to save PNGs to (created if it does not exist).
    out_prefix: first part of output filenames, e.g. "{file stem}_{image name}".
    intensity_range: intensity range for leveling, 'auto' for quantile-based.
    auto_range_quantiles: quantiles used when intensity_range == 'auto'.
    projection_type: projection for 3D data, one of min/max/mean.
    do_color: whether to compose channels to a single RGB image (True) or
        create one visualization per channel (False).
    color_dimension_label: dimension label (or sequence of candidate labels, first
        one present in the image wins) holding the color channels for RGB composites.
        'C' for most formats, additionally 'S' for RGB-sampled LIF images.
    visualization_colors: color name per channel for RGB composites.
    prefixes: filename prefixes per grouping dimension. Dimensions without
        an entry get a "_<dimension>" prefix.
    use_channel_names: whether to use channel names (e.g. "DAPI") instead of
        integer channel indices in output filenames.
    use_indices: whether to use integer indices (instead of coordinate labels)
        for grouping dimensions in output filenames.
    min_index_len: zero-padding width of indices in output filenames.

    Returns
    -------
    list of written PNG files
    """
    if prefixes is None:
        prefixes = DEFAULT_PREFIXES

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    # dimension holding the color channels for RGB composites
    # (first candidate label present in the image wins)
    if isinstance(color_dimension_label, str):
        color_dimension_label = (color_dimension_label,)
    rgb_dim = next((d for d in color_dimension_label if d in img_xa.dims), None)

    # group by all non-channel/non-spatial dimensions
    # (channels stay together in case we want to create color composites)
    excluded_dims = {"Z", "Y", "X"} | ({rgb_dim} if do_color and rgb_dim is not None else set())
    grouping_dims = [d for d in img_xa.dims if d not in excluded_dims]
    if len(grouping_dims) > 0:
        img_grouped = img_xa.groupby(grouping_dims)
    else:
        # if no grouping dimensions remain, fake a single group
        img_grouped = [((), img_xa)]

    out_files = []
    for idx, img_i in img_grouped:

        # treat even a single grouping dimension as list of one
        if len(grouping_dims) == 1:
            idx = [idx]

        # get image data per channel dimension if we want to create color composites
        if do_color and rgb_dim is not None and rgb_dim in img_i.dims:
            channel_imgs = [img_ci.values.squeeze() for (_, img_ci) in img_i.groupby(rgb_dim)]
        # otherwise, work with the full image
        else:
            channel_imgs = [img_i.values.squeeze()]

        # project (3D) and normalize intensities to 8bit
        projected_imgs = [
            project_and_normalize_8bit(
                channel_img,
                pixel_size,
                projection_type=projection_type,
                intensity_range=intensity_range,
                auto_range_quantiles=auto_range_quantiles,
            )
            for channel_img in channel_imgs
        ]

        if do_color and rgb_dim is not None and rgb_dim in img_i.dims:

            # get channel indices of groups (groupby works lexicographically)
            channel_idx = [ci for (ci, _) in img_i.groupby(rgb_dim)]
            channel_idx_int = [img_xa.get_index(rgb_dim).get_loc(ci) for ci in channel_idx]

            # re-sort projected images to original channel order
            projected_imgs = np.array(projected_imgs)[np.argsort(channel_idx_int)]

            composite_img = (
                gray_images_to_rgb_composite(projected_imgs, color_names=visualization_colors)
                * 255
            ).astype(np.uint8)
        else:
            composite_img = projected_imgs[0]

        # get filename indices for grouping dimensions
        filename_idx = []
        for dim, val in zip(grouping_dims, idx):

            # optionally use channel names instead of integer channel indices
            if use_channel_names and dim == rgb_dim:
                filename_idx.append(_sanitize(val))
            elif use_indices:
                filename_idx.append(str(img_xa.get_index(dim).get_loc(val)).rjust(min_index_len, "0"))
            else:
                filename_idx.append(str(val))

        # construct output filename
        out_filename = out_dir / (
            out_prefix
            + "".join(prefixes.get(dim, f"_{dim.lower()}") + i for dim, i in zip(grouping_dims, filename_idx))
            + ".png"
        )

        # save, catch low contrast warning
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            imsave(str(out_filename), composite_img)
        out_files.append(out_filename)

    return out_files


def quick_visualize_lif(
    in_path,
    raw_subdir="",
    out_subdir="quick_visualization",
    file_pattern="[!.]*.lif",
    intensity_range="auto",
    auto_range_quantiles=(0.02, 0.9995),
    projection_type="max",
    do_color=False,
    color_dimension_label=("C", "S"),
    visualization_colors=DEFAULT_COLOR_NAMES,
    prefixes=None,
    use_channel_names=False,
):
    """Create quick 8bit PNG visualizations of all images in all ```.lif``` files in a folder.

    Each ```.lif``` file can contain multiple independent images (e.g. single acquisitions
    and Z-stacks); each of them is visualized separately. Output files are named
    ```{input file name (without ending)}_{image name}_{dimension tags}.png``` (only dimensions
    that are present in the image will show up in the output filename) and placed in
    ```{in_path}/{out_subdir}```.

    See ```visualize_image_xarray``` for the remaining parameters.

    Note: FLIM datasets (with a fluorescence lifetime dimension) are not supported.

    Returns
    -------
    dict mapping input files to lists of written PNG files
    """
    in_dir = Path(in_path) / raw_subdir
    out_dir = Path(in_path) / out_subdir
    out_dir.mkdir(exist_ok=True)

    results = {}
    for in_file in sorted(in_dir.glob(file_pattern)):

        out_files = []
        # a .lif file can contain multiple images (e.g. single acquisitions and Z-stacks)
        with liffile.LifFile(in_file) as reader:
            for image in reader.images:

                img_xa = image.asxarray()

                # concatenate original file name and internal image name
                # (image names can be nested, e.g. "Experiment/Image1")
                image_name = _sanitize(image.name)
                out_prefix = f"{in_file.stem}_{image_name}"

                out_files += visualize_image_xarray(
                    img_xa,
                    get_pixel_size_micron(img_xa),
                    out_dir,
                    out_prefix,
                    intensity_range=intensity_range,
                    auto_range_quantiles=auto_range_quantiles,
                    projection_type=projection_type,
                    do_color=do_color,
                    color_dimension_label=color_dimension_label,
                    visualization_colors=visualization_colors,
                    prefixes=prefixes,
                    use_channel_names=use_channel_names,
                )

        print(f"saved visualization(s) for {in_file}")
        results[in_file] = out_files

    return results
