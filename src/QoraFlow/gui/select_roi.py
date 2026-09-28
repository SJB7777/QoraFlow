from pathlib import Path
#import matplotlib
#matplotlib.use("TkAgg")
#import os
#os.environ["QT_API"] = "pyqt5"
import numpy as np
import pandas as pd
from roi_rectangle import RoiRectangle
import zarr

from ..config import ExpConfig
from ..filesystem import get_run_scan_dir
from ..integrator.loader import get_hdf5_images
from .roi_core import RoiSelector


def get_metadata_roi(scan_dir: str | Path, config: ExpConfig) -> RoiRectangle:
    """Get Roi from metadata"""
    scan_dir = Path(scan_dir)
    files = list(scan_dir.glob("*.h5"))
    file: Path = files[0]
    metadata = pd.read_hdf(file, key="metadata")
    roi_coord = np.array(
        metadata[
            f"detector_{config.param.hutch.value}_{config.param.detector.value}_parameters.ROI"
        ].iloc[0][0]
    )

    roi = np.array(
        [
            roi_coord[config.param.y1],
            roi_coord[config.param.y2],
            roi_coord[config.param.x1],
            roi_coord[config.param.x2],
        ],
        dtype=np.int_,
    )

    return RoiRectangle.from_tuple(roi)


def select_roi(
    scan_dir: str | Path, config: ExpConfig, index_mode: int | None = None
) -> RoiRectangle:
    """Select ROI from GUI"""
    scan_dir = Path(scan_dir)
    files = list(scan_dir.glob("*.h5"))
    files.sort(key=lambda name: int(name.stem[1:]))
    if index_mode is None:
        index = len(files) // 2
    else:
        index = index_mode

    file: Path = scan_dir / files[index]
    image = get_hdf5_images(file, config).sum(axis=0)
    return RoiRectangle.from_tuple(RoiSelector().select_roi(np.log1p(image)))


def select_roi_zarr(
    file: str | Path,
    step: int | None = None,
) -> RoiRectangle:
    """Select ROI from 10 central shots of the middle scan step."""

    root = zarr.open_group(file, mode="r")
    detector = root["det-eh1-jungfrau2"]

    n_steps, n_shots = detector.shape[0], detector.shape[1]

    if step is None:
        step = n_steps // 2

    mid = n_shots // 2
    start = max(mid - 5, 0)
    stop = min(mid + 5, n_shots)

    image = np.asarray(
        detector[step, start:stop],
        dtype=np.float32,
    )
    image = image.reshape(-1, *image.shape[-2:]).mean(0)

    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)

    coords = RoiSelector().select_roi(
        np.log1p(np.maximum(image, 0))
    )

    if coords is None:
        raise RuntimeError("ROI selection cancelled or nothing selected.")

    return RoiRectangle.from_tuple(coords)

def auto_roi(scan_dir: str | Path, config: ExpConfig, index_mode: int | None = None, half_size: int = 20):
    """Get Roi based on the maximum value of the image"""
    scan_dir = Path(scan_dir)
    files = list(scan_dir.glob("*.h5"))
    files.sort(key=lambda file: int(file.stem[1:]))
    if index_mode is None:
        index = len(files) // 2
    else:
        index = index_mode
    
    file: Path = scan_dir / files[index]
    
    image = get_hdf5_images(file, config).sum(axis=0)

    # Normalize image
    img = np.nan_to_num(image)
    img = img - np.min(img)
    if np.max(img) != 0:
        img = img / np.max(img)

    # Threshold to remove background noise (e.g., keep top 10%)
    threshold = np.percentile(img, 90)
    mask = img >= threshold
    masked_img = img * mask

    # Compute centroid (intensity-weighted)
    total = masked_img.sum()

    indices = np.indices(masked_img.shape)
    y_c = int(np.sum(indices[0] * masked_img) / total)
    x_c = int(np.sum(indices[1] * masked_img) / total)

    return RoiRectangle(
        max(y_c - half_size, 0),
        min(y_c + half_size, img.shape[0]),
        max(x_c - half_size, 0),
        min(x_c + half_size, img.shape[1])
    )


if __name__ == "__main__":
    from QoraFlow.config import ConfigManager

    config: ExpConfig = ConfigManager.load_config()
    scan_dir = get_run_scan_dir(config.path.load_dir, 114, 1)
    print(config)

    roi1 = get_metadata_roi(scan_dir, config)
    print(roi1)
    roi2 = select_roi(scan_dir, config, 0)
    print(roi2)
    roi3 = auto_roi(scan_dir, config, 0)
    print(roi3)
