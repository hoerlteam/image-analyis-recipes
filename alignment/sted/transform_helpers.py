from xml.etree import ElementTree
from collections import namedtuple
import json

from h5py import File
import numpy as np
from skimage.transform import AffineTransform

from calmutils.misc.json import recursive_dict_query
from calmutils.stitching import translation_matrix, scale_matrix
from msr_reader import OBFFile


# convenience data class for scan field info
ScanFieldMetadata = namedtuple(
    "ScanFieldMetadata",
    [
        "stage_offset",
        "stage_offset_global",
        "scan_offset",
        "scan_offset_global",
        "fov_length",
        "pixel_size",
        "stage_direction",
        "scan_direction",
    ],
)


# default stage / scan directions
# (relative to pixel coordinates in image data)
STAGE_DIRECTIONS_DEFAULT = np.array([1, 1, -1])
SCAN_DIRECTIONS_DEFAULT = np.array([1, 1, 1])


def load_stage_directions(hardware_parameter_dict):

    x_inverted = recursive_dict_query(
        hardware_parameter_dict, "OlympusIX/stage/invert_x"
    )
    y_inverted = recursive_dict_query(
        hardware_parameter_dict, "OlympusIX/stage/invert_y"
    )
    if x_inverted is None or y_inverted is None:
        raise ValueError("could not load stage directions")

    return (1, -1 if y_inverted else 1, -1 if x_inverted else 1)


def load_scan_directions(hardware_parameter_dict):

    x_flipped = recursive_dict_query(
        hardware_parameter_dict, "ExpControl/calibration/scan/flip_x_axis"
    )
    y_flipped = recursive_dict_query(
        hardware_parameter_dict, "ExpControl/calibration/scan/flip_y_axis"
    )
    z_flipped = recursive_dict_query(
        hardware_parameter_dict, "ExpControl/calibration/scan/flip_z_axis"
    )

    if x_flipped is None or y_flipped is None or z_flipped is None:
        raise ValueError("could not load scan directions")

    # NOTE: flipped z axis -> same direction as stage (1)
    return (1 if z_flipped else -1, -1 if y_flipped else 1, -1 if x_flipped else 1)


def get_scan_field_metadata(msr_path, stack_idx=0):

    with OBFFile(msr_path) as reader:
        # imspector metadata (including stage position)
        xml_imspector_metadata = reader.get_xml_metadata(stack_idx)

    # parse XML string, get scan range element
    et_root = ElementTree.fromstring(xml_imspector_metadata)
    range_elem = et_root.find("doc/ExpControl/scan/range")

    # get properties of scan field
    stage_off = (range_elem.find(f"coarse_{d}/off").text for d in "zyx")
    stage_global_off = (range_elem.find(f"coarse_{d}/g_off").text for d in "zyx")
    scan_off = (range_elem.find(f"{d}/off").text for d in "zyx")
    scan_global_off = (range_elem.find(f"{d}/g_off").text for d in "zyx")
    fov_len = (range_elem.find(f"{d}/len").text for d in "zyx")
    pixel_size = (range_elem.find(f"{d}/psz").text for d in "zyx")

    # for each of the vectors: individual elements to float, vec to np array
    ret = map(
        lambda vals: np.array(list(map(float, vals))),
        [
            stage_off,
            stage_global_off,
            scan_off,
            scan_global_off,
            fov_len,
            pixel_size,
            STAGE_DIRECTIONS_DEFAULT,
            SCAN_DIRECTIONS_DEFAULT,
        ],
    )

    # TODO: get actual stage / scan directions
    # NOTE: drop as newer versions of msr files no longer contain this metadata

    # return as namedtuple
    return ScanFieldMetadata(*ret)


def get_scan_field_metadata_h5(h5_file, acquisition_path, configuration_idx=0):

    # open h5 file, get attributes (measurement / hardware metadata) for given acquisition
    with File(h5_file) as fd:
        measurement_metadata = json.loads(
            fd[f"experiment/{acquisition_path}/{configuration_idx}"].attrs[
                "measurement_meta"
            ]
        )
        hardware_metadata = json.loads(
            fd[f"experiment/{acquisition_path}/{configuration_idx}"].attrs[
                "global_meta"
            ]
        )

    # get scan range subdirectory
    attrs_scan = recursive_dict_query(measurement_metadata, "ExpControl/scan/range")

    # get properties of scan field
    stage_off = (recursive_dict_query(attrs_scan, f"coarse_{d}/off") for d in "zyx")
    stage_global_off = (
        recursive_dict_query(attrs_scan, f"coarse_{d}/g_off") for d in "zyx"
    )
    scan_off = (recursive_dict_query(attrs_scan, f"{d}/off") for d in "zyx")
    scan_global_off = (recursive_dict_query(attrs_scan, f"{d}/g_off") for d in "zyx")
    fov_len = (recursive_dict_query(attrs_scan, f"{d}/len") for d in "zyx")
    pixel_size = (recursive_dict_query(attrs_scan, f"{d}/psz") for d in "zyx")

    # load scan / stage directions
    scan_directions = load_scan_directions(hardware_metadata)
    stage_directions = load_stage_directions(hardware_metadata)

    # for each of the vectors: individual elements to float, vec to np array
    ret = map(
        lambda vals: np.array(list(map(float, vals))),
        [
            stage_off,
            stage_global_off,
            scan_off,
            scan_global_off,
            fov_len,
            pixel_size,
            stage_directions,
            scan_directions,
        ],
    )

    # return as namedtuple
    return ScanFieldMetadata(*ret)


def world_coords_for_pixel_spots(
    spots, field_metadata: ScanFieldMetadata
):
    spots = np.array(spots).reshape((-1, 3))

    # combine offsets, consider z stage direction
    offset = (
        field_metadata.scan_direction
        * (field_metadata.scan_offset + field_metadata.scan_offset_global)
        + field_metadata.stage_direction
        * (field_metadata.stage_offset + field_metadata.stage_offset_global)
    )

    # Imspector offset is center of image -> to origin of image
    offset = offset - 1 / 2 * field_metadata.fov_length

    # add spot coords in world units
    spots_world = offset + spots * field_metadata.pixel_size
    return spots_world


def world_transform_to_pixel_transform(
    transform, origin_ref, origin_moving, pixel_size_ref, pixel_size_moving
):
    """
    Create a combined transformation matrix to transform pixels given a world coordinate transformation

    The steps are:
    1) switch from pixel to world coords
    2) move to (world) origin of moving image
    3) transform
    4) move back to origin (of reference image)
    5) switch back to pixel coordinates

    NOTE: transform should be from moving -> reference. invert if necessary.
    """
    transform = (
        np.linalg.inv(scale_matrix(pixel_size_ref))
        @ np.linalg.inv(translation_matrix(origin_ref))
        @ transform
        @ translation_matrix(origin_moving)
        @ scale_matrix(pixel_size_moving)
    )
    return transform


def affine_transform_nd(dimensionality):
    """
    return AffineTransform constructor with specified dimensionality, would default to 2 otherwise
    """
    return lambda: AffineTransform(dimensionality=dimensionality)
