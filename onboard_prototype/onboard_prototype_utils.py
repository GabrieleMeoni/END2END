# Dictionary for onboard coarse spatial coregistration
from torchvision.transforms.functional import rotate
import torch
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
from glob import glob

# Coregistration LUT
COREGISTRATION_DICT_OPTIMIZED = {
    "S2A": {
        1: {
            "shifts": [[177.0, 14.0], [-22.0, 14.0]],
            "m_shifts": [177.0, 22.0, 14.0, 0],
        },
        2: {
            "shifts": [[-185.0, 16.0], [16.0, 14.0]],
            "m_shifts": [16.0, 185.0, 16.0, 0],
        },
        3: {
            "shifts": [[174.0, 13.0], [-22.0, 14.0]],
            "m_shifts": [174.0, 22.0, 14.0, 0],
        },
        4: {
            "shifts": [[-182.0, 15.0], [16.0, 13.0]],
            "m_shifts": [16.0, 182.0, 15.0, 0],
        },
        5: {
            "shifts": [[172.0, 13.0], [-23.0, 13.0]],
            "m_shifts": [172.0, 23.0, 13.0, 0],
        },
        6: {
            "shifts": [[-183.0, 12.0], [15.0, 12.0]],
            "m_shifts": [15.0, 183.0, 12.0, 0],
        },
        7: {
            "shifts": [[173.0, 12.0], [-22.0, 12.0]],
            "m_shifts": [173.0, 22.0, 12.0, 0],
        },
        8: {
            "shifts": [[-183.0, 12.0], [15.0, 12.0]],
            "m_shifts": [15.0, 183.0, 12.0, 0],
        },
        9: {
            "shifts": [[172.0, 11.0], [-24.0, 11.0]],
            "m_shifts": [172.0, 24.0, 11.0, 0],
        },
        10: {
            "shifts": [[-186.0, 9.0], [15.0, 12.0]],
            "m_shifts": [15.0, 186.0, 12.0, 0],
        },
        11: {
            "shifts": [[175.0, 11.0], [-24.0, 11.0]],
            "m_shifts": [175.0, 24.0, 11.0, 0],
        },
        12: {
            "shifts": [[-191.0, 6.0], [15.0, 11.0]],
            "m_shifts": [15.0, 191.0, 11.0, 0],
        },
    },
    "S2B": {
        1: {"shifts": [[174.0, 2.0], [-25.0, 3.0]], "m_shifts": [174.0, 25.0, 3.0, 0]},
        2: {"shifts": [[-187.0, 5.0], [14.0, 3.0]], "m_shifts": [14.0, 187.0, 5.0, 0]},
        3: {"shifts": [[172.0, 3.0], [-24.0, 3.0]], "m_shifts": [172.0, 24.0, 3.0, 0]},
        4: {"shifts": [[-184.0, 4.0], [14.0, 2.0]], "m_shifts": [14.0, 184.0, 4.0, 0]},
        5: {"shifts": [[171.0, 2.0], [-24.0, 2.0]], "m_shifts": [171.0, 24.0, 2.0, 0]},
        6: {"shifts": [[-185.0, 2.0], [13.0, 2.0]], "m_shifts": [13.0, 185.0, 2.0, 0]},
        7: {"shifts": [[170.0, 1.0], [-24.0, 1.0]], "m_shifts": [170.0, 24.0, 1.0, 0]},
        8: {"shifts": [[-185.0, 1.0], [13.0, 1.0]], "m_shifts": [13.0, 185.0, 1.0, 0]},
        9: {"shifts": [[171.0, 1.0], [-25.0, 1.0]], "m_shifts": [171.0, 25.0, 1.0, 0]},
        10: {
            "shifts": [[-187.0, -2.0], [12.0, 1.0]],
            "m_shifts": [12.0, 187.0, 1.0, 2.0],
        },
        11: {
            "shifts": [[-187.0, -2.0], [12.0, 1.0]],
            "m_shifts": [12.0, 187.0, 1.0, 2.0],
        },
        12: {
            "shifts": [[-187.0, -2.0], [12.0, 1.0]],
            "m_shifts": [12.0, 187.0, 1.0, 2.0],
        },
    },
}


# S2 detector full scale
S2_DETECTOR_FS = 4095


# Coarse coregistration function
def coarse_coregistration(
    X,
    satellite,
    granule_detector_number,
    rotate_swir_bands=True,
    X_prev=None,
    crop_empty_pixels=False,
):
    """Applying coarse coregistration.

    Args:
        X (torch.tensor): bands to register.
        satellite (str): satellite name.
        granule_detector_number (int): detector number.
        rotate_swir_bands (bool, optional): if True, SWIR bands are rotated. Defaults to True.
        X_prev (torch.tensor, optional): previous granules used to fill missing pixels. If None, missing pixels are/are not cropped depending on crop_empty_pixels. Defaults to None.
        crop_empty_pixels (bool, optional): if True and X_prev is None, missing pixels due to registration are cropped. Defaults to False.

    Returns:
        torch.tensor: coarsely registered granule.
    """
    granule_detector_number = int(granule_detector_number)

    def apply_shift(band, shifts):
        band_shifted = torch.zeros_like(band)
        shifts = [int(shifts[0]), int(shifts[1])]
        if (shifts[0] == 0) and (shifts[1] == 0):
            band_shifted = band
        elif (shifts[0] == 0) and (shifts[1] < 0):
            band_shifted[:, : int(shifts[1])] = band[:, -int(shifts[1]) :]
        elif (shifts[0] == 0) and (shifts[1] > 0):
            band_shifted[:, int(shifts[1]) :] = band[:, : -int(shifts[1])]
        elif (shifts[0] < 0) and (shifts[1] == 0):
            band_shifted[: int(shifts[0]), :] = band[-int(shifts[0]) :, :]
        elif (shifts[0] > 0) and (shifts[1] == 0):
            band_shifted[int(shifts[0]) :, :] = band[: -int(shifts[0]), :]
        elif (shifts[0] > 0) and (shifts[1] > 0):
            band_shifted[int(shifts[0]) :, int(shifts[1]) :] = band[
                : -int(shifts[0]), : -int(shifts[1])
            ]
        elif (shifts[0] > 0) and (shifts[1] < 0):
            band_shifted[int(shifts[0]) :, : int(shifts[1])] = band[
                : -int(shifts[0]), -int(shifts[1]) :
            ]
        elif (shifts[0] < 0) and (shifts[1] > 0):
            band_shifted[: int(shifts[0]), int(shifts[1]) :] = band[
                -int(shifts[0]) :, : -int(shifts[1])
            ]
        else:
            band_shifted[: int(shifts[0]), : int(shifts[1])] = band[
                -int(shifts[0]) :, -int(shifts[1]) :
            ]

        return band_shifted

    def shift_band(band, shifts, granule_detector_number, filler_found, b_prev=None):
        band_shifted = apply_shift(band, shifts)

        if b_prev is not None:
            if rotate_swir_bands:
                band_filler = rotate(b_prev.unsqueeze(2), 180).squeeze(2)
            else:
                band_filler = b_prev

            band_filler = apply_shift(band_filler, shifts=[0, shifts[1]])

            if granule_detector_number % 2:
                # Band filler is on top:
                if shifts[0] > 0:
                    top = shifts[0]
                    band_shifted[: int(top),] = band_filler[-int(top) :]
                    filler_found["top"] = True
            else:
                if shifts[0] < 0:
                    bottom = -shifts[0]
                    band_shifted[-int(bottom) :, :] = band_filler[: int(bottom), :]
                    filler_found["bottom"] = True

        return band_shifted, filler_found

    # Get shifts
    bands_shifts_dict = COREGISTRATION_DICT_OPTIMIZED[satellite][
        granule_detector_number
    ]

    # Assigning to None and changed later if needed.
    filler_found = {"top": None, "bottom": None}
    if rotate_swir_bands:
        band_11 = rotate(X[1].unsqueeze(2), 180).squeeze(2)
        band_12 = rotate(X[2].unsqueeze(2), 180).squeeze(2)
    else:
        band_11, band_12 = X[1], X[2]
    shifts_11 = bands_shifts_dict["shifts"][0]
    shifts_12 = bands_shifts_dict["shifts"][1]
    if X_prev is not None:
        band_11_prev = X_prev[1]
        band_12_prev = X_prev[2]
    else:
        band_11_prev = None
        band_12_prev = None

    band_11_shifted, filler_found = shift_band(
        band_11, shifts_11, granule_detector_number, filler_found, band_11_prev
    )
    band_12_shifted, filler_found = shift_band(
        band_12, shifts_12, granule_detector_number, filler_found, band_12_prev
    )
    # ------------------------------- Filling:

    # Managing crop exmpty pixel case
    band_8a = X[0]
    if crop_empty_pixels:
        max_top, max_bottom, max_left, max_right = bands_shifts_dict[
            "m_shifts"
        ]  # Maximum of offsets init.

        # The band shall be cropped only if it has not been filled.
        if (filler_found["bottom"] is None) and (max_bottom != 0):
            band_8a = band_8a[: -int(max_bottom)]
            band_11_shifted = band_11_shifted[: -int(max_bottom)]
            band_12_shifted = band_12_shifted[: -int(max_bottom)]

        # The band shall be cropped only if it has not been filled.
        if (filler_found["top"] is None) and (max_top != 0):
            band_8a = band_8a[int(max_top) :]
            band_11_shifted = band_11_shifted[int(max_top) :]
            band_12_shifted = band_12_shifted[int(max_top) :]

        if max_right != 0:
            band_8a = band_8a[:, int(max_left) : -int(max_right)]
            band_11_shifted = band_11_shifted[:, int(max_left) : -int(max_right)]
            band_12_shifted = band_12_shifted[:, int(max_left) : -int(max_right)]
        else:
            band_8a = band_8a[:, int(max_left) :]
            band_11_shifted = band_11_shifted[:, int(max_left) :]
            band_12_shifted = band_12_shifted[:, int(max_left) :]

    coregistered_bands_tensor = torch.zeros([3, band_8a.shape[0], band_8a.shape[1]]).to(
        X.device
    )
    coregistered_bands_tensor[0] = band_8a
    coregistered_bands_tensor[1] = band_11_shifted
    coregistered_bands_tensor[2] = band_12_shifted
    return coregistered_bands_tensor


# Load single data
def load_single_data(data_dir):
    """Load a single data and returns a tuple containing of data, metadata.

    Args:
        data_dir (str): path to a granule directory.

    Returns:
        tuple: (granule data, event name, granule number, satellite name, detector number).
    """
    info = data_dir.split("_")
    detector_number = info[-1]
    granule_number = info[-3]
    satellite = info[-2]
    event = info[-5].split(os.sep)[-1] + "_" + info[-4]
    for n, b in enumerate(["B8A", "B11", "B12"]):
        b_path = os.path.join(data_dir, b + ".tif")

        band = cv2.imread(b_path, cv2.IMREAD_ANYDEPTH)

        if n == 0:
            b_tensor = np.zeros([3, band.shape[0], band.shape[1]])

        b_tensor[n] = band

    return (
        torch.from_numpy(img_as_ubyte(b_tensor / S2_DETECTOR_FS).astype(np.float32)),
        event,
        granule_number,
        satellite,
        detector_number,
    )


def load_data(data_dir):
    """Load data and cluster them according to event : {detector_number : { granule_number : data}}

    Args:
        data_dir (str): directory containing all the granules.

    Returns:
        dictionary: event : {detector_number : { granule_number : data}}.
    """
    data_dirs = glob(os.path.join(data_dir, "*"))
    event_list = []
    event_info = []

    for data_dir in data_dirs:
        b_tensor, event, granule_number, satellite, detector_number = load_single_data(
            data_dir
        )
        if event not in event_list:
            event_list.append(event)
            granule_number_tensor_dict_per_detector_number = {granule_number: b_tensor}
            event_info.append(
                [
                    satellite,
                    {detector_number: granule_number_tensor_dict_per_detector_number},
                ]
            )

        else:
            event_idx = event_list.index(event)
            granule_number_tensor_dict_per_detector_number = event_info[event_idx][1]

            if not (
                detector_number
                in list(granule_number_tensor_dict_per_detector_number.keys())
            ):
                a = list(granule_number_tensor_dict_per_detector_number.keys()) + [
                    detector_number
                ]
                b = list(granule_number_tensor_dict_per_detector_number.values()) + [
                    {granule_number: b_tensor}
                ]
                granule_number_tensor_dict_per_detector_number = dict(zip(a, b))
            else:
                my_dict = granule_number_tensor_dict_per_detector_number[
                    detector_number
                ]
                a = list(my_dict.keys()) + [granule_number]
                b = list(my_dict.values()) + [b_tensor]
                granule_number_tensor_dict_per_detector_number[detector_number] = dict(
                    zip(a, b)
                )

            event_info[event_idx][1] = granule_number_tensor_dict_per_detector_number

    return dict(zip(event_list, event_info))


def plot(x, x_max=255, ax=None):
    x = x / x_max
    if ax is None:
        plt.imshow(x.swapaxes(0, 1).swapaxes(1, 2))
    else:
        ax.imshow(x.swapaxes(0, 1).swapaxes(1, 2))
