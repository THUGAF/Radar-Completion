from typing import Tuple
import os
from collections import OrderedDict
import numpy as np


AZIMUTH_RANGE = 360
MAX_NUM_REF_RANGE_BIN = 460
MAX_NUM_DOP_RANGE_BIN = 920
DEFAULT_REF = -33.0
DEFAULT_DOP = -64.5


def elevation_mapping(elevation: float) -> float:
    if elevation < 1.0:
        elevation = 0.5
    elif elevation >= 1.0 and elevation < 2.0:
        elevation = 1.5
    elif elevation >= 2.0 and elevation < 3.0:
        elevation = 2.4
    elif elevation >= 3.0 and elevation < 4.0:
        elevation = 3.4
    elif elevation >= 4.0 and elevation < 5.0:
        elevation = 4.3
    elif elevation >= 5.0 and elevation < 8.0:
        elevation = 6.0
    elif elevation > 8.0 and elevation < 12.0:
        elevation = 9.9
    elif elevation >= 12 and elevation < 18.0:
        elevation = 14.6
    elif elevation >= 18.0:
        elevation = 19.5
    return elevation


def azimuth_mapping(azimuth: float) -> float:
    azimuth = float(round(azimuth))
    if azimuth >= AZIMUTH_RANGE:
        azimuth = azimuth - AZIMUTH_RANGE
    return azimuth


def read_radar_bin(path: str) -> Tuple[np.ndarray, np.ndarray]:
    size = os.path.getsize(path)
    num_scans = size // 2432
    
    with open(path, 'rb') as f:
        total_data = {}
        
        for i in range(num_scans):
            # Header
            f.seek(28, 1)

            # Basic information
            f.seek(8, 1)
            # milliseconds = int.from_bytes(f.read(4), 'little')
            # days = int.from_bytes(f.read(2), 'little')
            # unambiguous_distance = int.from_bytes(f.read(2), 'little') / 10.0

            # Crucial information
            azimuth = int.from_bytes(f.read(2), 'little') / 8.0 * 180.0 / 4096.0
            azimuth = azimuth_mapping(azimuth)
            radial_order = int.from_bytes(f.read(2), 'little')
            radial_status = int.from_bytes(f.read(2), 'little')
            elevation = int.from_bytes(f.read(2), 'little') / 8.0 * 180.0 / 4096.0
            elevation = elevation_mapping(elevation)
            num_elevations = int.from_bytes(f.read(2), 'little')
            f.seek(8, 1)
            # ref_1st_range_bin = int.from_bytes(f.read(2), 'little')
            # dop_1st_range_bin = int.from_bytes(f.read(2), 'little')
            # ref_range_bin_distance = int.from_bytes(f.read(2), 'little')
            # dop_range_bin_distance = int.from_bytes(f.read(2), 'little')
            num_ref_range_bin = int.from_bytes(f.read(2), 'little')
            num_dop_range_bin = int.from_bytes(f.read(2), 'little')
            f.seek(4, 1)
            # num_sector = int.from_bytes(f.read(2), 'little')
            # correction_coefficient = int.from_bytes(f.read(4), 'little')
            f.seek(6, 1)
            # ref_pointer = int.from_bytes(f.read(2), 'little')
            # dop_pointer = int.from_bytes(f.read(2), 'little')
            # width_pointer = int.from_bytes(f.read(2), 'little')
            f.seek(4, 1)
            # dop_speed_res = int.from_bytes(f.read(2), 'little') / 4.0
            # vcp_mode = int.from_bytes(f.read(2), 'little')
            f.seek(8, 1)
            f.seek(6, 1)
            # ref_rev_pointer = int.from_bytes(f.read(2), 'little')
            # dop_rev_pointer = int.from_bytes(f.read(2), 'little')
            # width_rev_pointer = int.from_bytes(f.read(2), 'little')
            nyquist_speed = int.from_bytes(f.read(2), 'little') / 100.0
            f.seek(38, 1)

            if radial_order == 1:
                if not elevation in total_data.keys():
                    total_data[elevation] = {}
                    for a in range(AZIMUTH_RANGE):
                        total_data[elevation][a] = np.ones(MAX_NUM_REF_RANGE_BIN) * DEFAULT_REF

            # Reflectivity
            refs = np.ones(MAX_NUM_REF_RANGE_BIN) * DEFAULT_REF
            if num_ref_range_bin > 0:
                for n in range(num_ref_range_bin):
                    ref = (int.from_bytes(f.read(1), 'little') - 2) / 2 - 32.0
                    refs[n] = ref
                total_data[elevation][azimuth] = refs
            
            # Doppler speed
            f.seek(num_dop_range_bin, 1)
            
            # Spectrual width
            f.seek(num_dop_range_bin, 1)
            
            # Tail
            pointer = f.tell()
            f.seek(2432 * (i + 1) - pointer, 1)

        ordered_total_data = {}
        for e in total_data.keys():
            ordered_total_data[e] = np.array(list(OrderedDict(sorted(total_data[e].items(), key=lambda x: x[0])).values()))

        elevations = np.array(list(ordered_total_data.keys()))
        reflectivities = np.stack(list(ordered_total_data.values()))

    return elevations, reflectivities
