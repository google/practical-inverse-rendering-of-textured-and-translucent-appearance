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

"""IES light utility functions.

This file contains utility functions to parse IES light information saved in
a .json file and converting to a mitsuba light usable with the ies_emitter
plugin.
"""

import json
from typing import Any
import mitsuba as mi  # type: ignore
import numpy as np
from scipy.spatial import transform

_IES_PROFILE = r"""IESNA:LM-63-2002
[TEST] L052111407
[TESTLAB] LIGHT LABORATORY, INC. (www.lightlaboratory.com)
[ISSUEDATE] 7/16/2021
[MANUFAC] Blizzard Lighting LLC
[LUMCAT] HotBoxô InfiniWhite
[LUMINAIRE] 7x 5W ACW 3-in-1 LEDs
[BALLASTCAT] CUSTOM DRIVER
[OTHER] INDICATING THE CANDELA VALUES ARE ABSOLUTE AND
[MORE] SHOULD NOT BE FACTORED FOR DIFFERENT LAMP RATINGS.
[INPUT] 120.04VAC, 36.21W
[TEST PROCEDURE] IESNA:LM-79-08
TILT=NONE
1 -1 1 33 19 1 1 -0.29 -0.29 0
1 1 36.21
0 1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 22 24 26 28 30 35 40 45 50 55 60 65 70 75 80 85 90
0 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90
14931 14904 14862 14586 13950 12960 10065 7309 6174 5035 3931 2276 1412 995 729 543 556 412 302 217 154 70 38 19 10 8 7 6 5 4 4 3 0
14931 14904 14861 14584 13951 12960 10062 7306 6175 4701 3933 2277 1409 992 728 543 556 411 301 217 154 70 38 19 10 8 7 6 5 4 4 3 0
14931 14902 14858 14582 13956 12956 10060 7300 6178 4684 3934 2277 1408 991 728 542 556 410 300 217 153 69 38 20 10 8 7 6 5 4 4 3 0
14931 14899 14850 14570 13954 12956 10054 7296 6175 4670 3930 2277 1408 991 727 541 555 408 300 218 152 69 39 20 11 9 7 7 5 4 4 3 0
14931 14896 14839 14555 13948 12951 10051 7300 6179 4663 3935 2282 1411 993 727 540 554 409 299 217 153 69 38 20 10 9 7 7 5 4 4 3 0
14931 14892 14832 14546 13943 12955 10051 7304 6186 4659 3942 2286 1416 994 727 540 554 409 299 217 152 69 38 20 10 9 7 7 5 4 4 3 0
14931 14889 14825 14540 13935 12944 10049 7313 6192 4663 3945 2287 1417 995 727 540 554 409 298 216 152 68 38 20 10 9 7 7 5 4 3 3 0
14931 14888 14822 14534 13935 12942 10047 7316 6201 4666 3949 2292 1418 993 726 540 554 409 298 216 152 68 38 20 10 9 7 7 5 4 3 3 0
14931 14889 14819 14534 13932 12944 10048 7319 6206 4673 3965 2297 1419 993 725 539 554 408 299 217 152 69 38 20 11 9 7 7 5 4 3 3 0
14931 14894 14817 14531 13928 12943 9895 7317 6209 4691 3973 2301 1421 993 725 539 553 408 299 217 152 69 38 20 11 9 7 6 5 4 3 3 0
14931 14898 14816 14531 13928 12941 9378 7317 6212 4705 3979 2305 1422 994 725 539 553 409 299 217 152 69 38 20 10 8 7 6 5 4 3 3 0
14931 14901 14814 14528 13921 12943 9207 7325 6222 4722 3979 2310 1425 996 727 540 553 410 300 217 153 69 38 19 10 8 7 6 5 4 3 3 0
14931 14902 14810 14525 13922 12946 9197 7329 6229 4747 3978 2315 1427 996 726 540 553 410 301 217 153 70 38 19 10 8 7 6 5 4 3 3 0
14931 14900 14807 14521 13924 12948 9199 7327 6233 4778 3989 2314 1428 995 724 539 552 409 300 216 152 69 38 19 10 8 7 6 5 4 3 3 0
14931 14899 14803 14520 13928 12949 9197 7327 6234 4820 3997 2315 1425 995 724 538 552 407 298 216 152 69 38 20 10 8 7 6 5 4 3 3 0
14931 14894 14797 14515 13929 12953 9200 7331 6237 4838 4005 2316 1422 993 722 573 550 406 298 216 151 69 38 20 11 9 7 7 5 4 3 3 0
14931 14893 14796 14511 13926 12952 9192 7337 6243 4858 4003 2319 1423 992 721 585 550 406 298 216 151 68 38 20 10 9 7 7 5 4 4 3 0
14931 14887 14790 14508 13921 12945 9189 7336 6235 4879 3997 2320 1423 992 721 586 549 406 297 215 150 67 37 18 10 9 7 7 5 4 4 3 0
14931 14881 14782 14504 13919 12943 8404 7344 6243 5110 3999 2316 1424 990 721 637 548 406 296 215 149 67 37 17 10 9 7 7 5 4 4 3 0"""


def _parse_ies_profile(
    bin_count_multiplier: int = 2,
) -> tuple[float, np.ndarray]:
  """Extracts intensity profile and light source diameter from an IES profile.

  This currently supports only disk-shaped light sources and radially averages
  emission.

  Args:
    bin_count_multiplier: An integer multiplier for the number of bins to use
      for the intensity profile. We resample the intensity profile to be
      discretized over cosine values (instead of angles). If the IES profile
      contains high-frequency components, a larger multplier might be needed to
      preserve accuracy (e.g., 8 or 16).

  Returns:
    the diameter of the light source
    the intensity profile, discretized over the cosine of the vertical angle.
  """
  lines = _IES_PROFILE.splitlines()
  content_index = None
  for i, line in enumerate(lines):
    if line.startswith('TILT=NONE'):
      content_index = i
      break
  if not content_index:
    raise ValueError('Unsupported IES profile!')

  header = lines[content_index + 1 : content_index + 5]
  elements = header[0].split(' ')

  n_vertical = int(elements[3])
  n_horizontal = int(elements[4])

  size_x = float(elements[7])
  size_y = float(elements[8])
  size_z = float(elements[9])

  if not (size_x < 0 and size_y < 0 and size_z == 0):
    raise ValueError('Only disk light shapes are supported!')

  light_diameter = abs(size_x)

  units_type = int(elements[6])
  used_units = 'meters' if units_type == 2 else 'feet'
  if used_units == 'feet':
    light_diameter *= 0.3048  # Converts to meters.

  data = lines[content_index + 5 :]
  data = np.array([np.fromstring(d, dtype=np.float32, sep=' ') for d in data])

  if data.shape[0] != n_horizontal or data.shape[1] != n_vertical:
    raise ValueError('Invalid IES profile!')

  # Average over horizontal observations (assumes radial symmetry).
  vertical_angles = np.fromstring(header[-2], dtype=np.float32, sep=' ')
  if vertical_angles.shape[0] != n_vertical:
    raise ValueError('Invalid IES profile!')

  intensity_profile = np.mean(data, axis=0)

  n_bins = bin_count_multiplier * vertical_angles.shape[0]
  cosine_interp = np.linspace(0, 1, n_bins)
  intensity_profile = np.interp(
      cosine_interp,
      np.flip(np.cos(np.deg2rad(vertical_angles))),
      np.flip(intensity_profile),
  )
  return light_diameter, intensity_profile


def _create_ies_emitter(
    radiance: float = 1.0,
    to_world: mi.ScalarTransform4f | None = None,
    smooth: bool = True,
    bin_count_multiplier: int = 2,
) -> dict[str, object]:
  """Creates a dictionary containing both shape and emitter of the IES profile.

  This function internally parses the IES profile and instantiates an emitter
  of correct shape and size. Currently, only disk emitters are supported
  and the IES profile is averaged over horizontal angles (i.e., we assume
  radially symmetrical emission).

  Args:
    radiance: the radiance of the emitter itself.
    to_world: the to_world transform of the emitter.
    smooth: if set to True, use the smooth version of the IES emitter. This
      prevents discontinuities in the integrand and simplifies optimization. On
      the other hand, the smooth emitter produces slightly more noisy
      renderings.
    bin_count_multiplier: An integer multiplier for the number of bins to use
      for the intensity profile. We resample the intensity profile to be
      discretized over cosine values (instead of angles). If the IES profile
      contains high-frequency components, a larger multplier might be needed to
      preserve accuracy (e.g., 8 or 16).

  Returns:
    A dictionary containing the emitter and holding shape.
  """

  if to_world is None:
    to_world = mi.ScalarTransform4f(1.0)

  light_diameter, ies_values = _parse_ies_profile(
      bin_count_multiplier=bin_count_multiplier
  )

  # We need to pass the array as a string to the Mitsuba scene dict.
  ies_values_string = ' '.join(np.char.mod('%f', ies_values))
  return {
      'type': 'disk',
      'emitter': {
          'type': 'iesemitter',
          'ies_values': ies_values_string,
          'smooth': smooth,
          'nested_emitter': {
              'radiance': {
                  'type': 'rgb',
                  'value': [radiance, radiance, radiance],
              },
              'type': 'area',
          },
      },
      'to_world': to_world @ mi.ScalarTransform4f().scale(light_diameter / 2),
  }


def _parse_light(
    light: dict[str, Any], bin_count_multiplier: int = 2
) -> dict[str, object]:
  """Parses a single light dictionary."""
  light_transform = light['transform']
  radiance = light['intensity']
  to_world = np.eye(4)
  axis_angle = np.asarray(list(light_transform['rotation'].values()))
  position = np.asarray(list(light_transform['position'].values()))
  to_world[:3, :3] = transform.Rotation.from_rotvec(axis_angle).as_matrix()
  to_world[:3, -1] = position

  return _create_ies_emitter(
      radiance=radiance,
      to_world=mi.ScalarTransform4f(to_world),
      bin_count_multiplier=bin_count_multiplier,
  )


def parse_json_lights_to_mitsuba_emitters(
    json_lights_path: str,
    bin_count_multiplier: int = 2,
) -> dict[str, object]:
  """Parses a JSON file containing IES lights and converts to Mitsuba emitters.

  Args:
    json_lights_path: The path to the JSON file containing the IES lights.
    bin_count_multiplier: An integer multiplier for the number of bins to use
      for the intensity profile. We resample the intensity profile to be
      discretized over cosine values (instead of angles). If the IES profile
      contains high-frequency components, a larger multplier might be needed to
      preserve accuracy (e.g., 8 or 16).

  Returns:
    A dictionary mapping light indices to Mitsuba emitters.
  """
  with open(json_lights_path, 'r') as f:
    json_lights = json.load(f)
  lights = json_lights['lights']
  mitsuba_lights = {}
  for light_index, light in lights.items():
    mitsuba_lights[light_index] = _parse_light(light, bin_count_multiplier)
  return mitsuba_lights
