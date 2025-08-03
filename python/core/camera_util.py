# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Camera utility functions.

This file contains utility functions to parse camera information saved in
a .json file and converting to a mitsuba sensor.
"""

import json
from typing import Any
import mitsuba as mi  # type: ignore
import numpy as np
from scipy.spatial import transform

# This is the transform to convert from the camera coordinate system to the
# coordinate system used by Mitsuba.
_CAMERA_COORDINATE_TRANSFORM = np.array(
    [
        [-1, 0, 0],
        [0, -1, 0],
        [0, 0, 1],
    ],
    dtype=np.float32,
)


def _parse_json_camera_to_mitsuba_sensor_dict(
    camera_dict: dict[str, Any],
    sensor_resolution: tuple[int, int] | None = None,
    sample_border: bool = False,
    reconstruction_filter: mi.ReconstructionFilter | None = None,
) -> dict[str, Any]:
  """Parses a camera saved in a dictionary to a Mitsuba sensor dictionary."""
  if camera_dict['projectionType'] != 'PERSPECTIVE':
    raise ValueError('Only perspective cameras are supported.')

  try:
    position = camera_dict['position']
    axis_angle = camera_dict['orientation']
    focal_length = camera_dict['focalLength']
    principal_point = camera_dict['principalPoint']
    height, width = int(camera_dict['sizeY']), int(camera_dict['sizeX'])
  except KeyError as e:
    print('Could not parse camera. Reason: ', e)
    return

  if reconstruction_filter is None:
    reconstruction_filter = {'type': 'box'}

  sensor_width, sensor_height = (
      sensor_resolution if sensor_resolution is not None else (width, height)
  )
  to_world = np.eye(4)
  to_world[:3, :3] = (
      transform.Rotation.from_rotvec(axis_angle).as_matrix().T
      @ _CAMERA_COORDINATE_TRANSFORM
  )
  to_world[:3, -1] = position
  fov = np.rad2deg(2 * np.arctan2(width, 2 * focal_length))
  principal_point_offset_x = principal_point[0] / width - 0.5
  principal_point_offset_y = principal_point[1] / height - 0.5

  return {
      'type': 'perspective',
      'fov': fov,
      'principal_point_offset_x': -principal_point_offset_x,
      'principal_point_offset_y': -principal_point_offset_y,
      'film': {
          'type': 'hdrfilm',
          'height': sensor_height,
          'width': sensor_width,
          'rfilter': reconstruction_filter,
          'sample_border': sample_border,
          'pixel_format': 'rgb',
      },
      'to_world': mi.ScalarTransform4f(to_world),
  }


def parse_json_camera_to_mitsuba_sensor_dict(
    camera_json_path: str,
    sensor_resolution: tuple[int, int] | None = None,
    sample_border: bool = False,
    reconstruction_filter: mi.ReconstructionFilter | None = None,
) -> dict[str, Any]:
  """Parses a camera saved in a .json file to a Mitsuba sensor dictionary.

  Args:
    camera_json_path: The path to the camera saved in a .json file.
    sensor_resolution: The resolution of the created sensor. If not provided,
      the resolution of the camera as saved in the .json file is used.
    sample_border: Whether to sample the border of the sensor.
    reconstruction_filter: The reconstruction filter to use for the sensor. If
      not provided, a box filter is used.

  Returns:
    A dictionary containing the equivalent Mitsuba sensor information.

  Raises:
    ValueError: If the camera is not a perspective camera.
    KeyError: If the json file does not contain the required fields.
  """
  with open(camera_json_path, 'r') as f:
    camera_dict = json.load(f)

  return _parse_json_camera_to_mitsuba_sensor_dict(
      camera_dict,
      sensor_resolution,
      sample_border,
      reconstruction_filter,
  )
