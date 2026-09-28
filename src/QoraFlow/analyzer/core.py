from collections.abc import Mapping
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from roi_rectangle import RoiRectangle
from scipy.ndimage import rotate
from scipy.optimize import curve_fit

from ..mathematics import gaussian, mul_delta_q


class DataAnalyzer:
    """
    Analyzes data from a given file and performs various operations on the images.

    Args:
        file (str): The path to the file containing the data.
        angle (int, optional): The angle to rotate the images. Defaults to 0.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not contain the required keys.
    """

    def __init__(self, file: str | Path, angle: int = 0) -> None:
        file = Path(file)
        if not file.exists():
            raise FileNotFoundError(f"The file {file} does not exist.")

        data: Mapping[str, npt.NDArray] = np.load(file)
        if "delay" not in data:
            raise ValueError(
                f"The file {file} does not contain the required keys: 'delay'"
            )

        self.delay: npt.NDArray = data["delay"]
        
        if "pon" not in data:
            self.poff_images: npt.NDArray = data["poff"]
            self.pon_images: npt.NDArray = np.full_like(data["poff"], 1)
        elif "poff" not in data:
            self.pon_images: npt.NDArray = data["pon"]
            self.poff_images: npt.NDArray = np.full_like(data["pon"], 1)
        else:
            self.poff_images: npt.NDArray = data["poff"]
            self.pon_images: npt.NDArray = data["pon"]
        if angle:
            self.poff_images = rotate(
                self.poff_images, angle, axes=(1, 2), reshape=False
            )
            self.pon_images = rotate(self.pon_images, angle, axes=(1, 2), reshape=False)

        self.poff_images = np.maximum(0, self.poff_images)
        self.pon_images = np.maximum(0, self.pon_images)

    def get_summed_image(self) -> tuple[npt.NDArray, npt.NDArray]:
        """
        return:
            pump off image, pump on image
        """
        return self.poff_images.sum(axis=0), self.pon_images.sum(axis=0)

    def pon_subtract_by_poff(self):
        """Subtrack pump on images by pump off images"""
        return np.maximum(self.pon_images - self.poff_images, 0)

    def _roi_center_of_masses(
        self, roi_rect: RoiRectangle, images: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        roi_images = roi_rect.slice(images)
        height, width = roi_rect.height, roi_rect.width

        y_coords, x_coords = np.mgrid[:height, :width]

        total_mass = np.sum(roi_images, axis=(1, 2))
        x_centroids = np.sum(x_coords * roi_images, axis=(1, 2)) / total_mass
        y_centroids = np.sum(y_coords * roi_images, axis=(1, 2)) / total_mass

        return x_centroids, y_centroids

    def _roi_gaussian(
        self, roi_rect: RoiRectangle, images: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:

        roi_images = roi_rect.slice(images)
        height, width = roi_rect.height, roi_rect.width

        intensities = []
        com_xs = []
        com_ys = []

        for image in roi_images:
            max_y, max_x = np.unravel_index(
                np.argmax(image), image.shape
            )  # pylint: disable=unbalanced-tuple-unpacking

            x: npt.NDArray = np.arange(0, width)
            y: npt.NDArray = np.arange(0, height)

            x_data = image.sum(axis=0)
            y_data = image.sum(axis=1)

            initial_guess_x = [
                x_data[max_x],
                max_x,
                (np.max(x_data) - np.min(x_data)) / 4,
            ]
            try:
                params_x = curve_fit(gaussian, x, x_data, p0=initial_guess_x)[0]
            except RuntimeError as e:
                print(e, ": x")
                params_x = [np.nan, np.nan, np.nan]

            initial_guess_y = [
                y_data[max_y],
                max_y,
                (np.max(y_data) - np.min(y_data)) / 4,
            ]
            try:
                params_y = curve_fit(gaussian, y, y_data, p0=initial_guess_y)[0]
            except RuntimeError as e:
                print(e, ": y")
                params_y = [np.nan, np.nan, np.nan]

            gaussian_a_x, gaussain_com_x, _ = params_x
            gaussian_a_y, gaussain_com_y, _ = params_y

            gaussian_a = np.sqrt(gaussian_a_x * gaussian_a_y)
            intensities.append(gaussian_a)
            com_xs.append(gaussain_com_x)
            com_ys.append(gaussain_com_y)

        return np.stack(intensities), np.stack(com_xs), np.stack(com_ys)

    def _roi_intensities(self, roi_rect: RoiRectangle, images: npt.NDArray):
        roi_images = roi_rect.slice(images)


        return roi_images.mean(axis=(1, 2))

    def analyze_by_roi(self, roi_rect: RoiRectangle) -> pd.DataFrame:

        poff_com_x, poff_com_y = self._roi_center_of_masses(roi_rect, self.poff_images)
        poff_intensity = self._roi_intensities(roi_rect, self.poff_images)
        pon_com_x, pon_com_y = self._roi_center_of_masses(roi_rect, self.pon_images)
        pon_intensity = self._roi_intensities(roi_rect, self.pon_images)

        # poff_guassain_intensity, poff_gussian_com_x, poff_gussian_com_y = self._roi_gaussian(
        #     roi_rect, self.poff_images
        # )
        # pon_guassain_intensity, pon_gussian_com_x, pon_gussian_com_y = self._roi_gaussian(
        #     roi_rect, self.pon_images
        # )

        roi_df = pd.DataFrame(
            data={
                "poff_com_x": mul_delta_q(poff_com_x - poff_com_x[0]),
                "poff_com_y": mul_delta_q(poff_com_y - poff_com_y[0]),
                "poff_intensity": poff_intensity / poff_intensity[0],
                "pon_com_x": mul_delta_q(pon_com_x - pon_com_x[0]),
                "pon_com_y": mul_delta_q(pon_com_y - pon_com_y[0]),
                "pon_intensity": pon_intensity / pon_intensity[0],
                # "poff_gussian_com_x": poff_gussian_com_x,
                # "poff_gussian_com_y": poff_gussian_com_y,
                # "poff_guassain_intensity": poff_guassain_intensity / poff_guassain_intensity[0],
                # "pon_gussian_com_x": pon_gussian_com_x,
                # "pon_gussian_com_y": pon_gussian_com_y,
                # "pon_guassain_intensity": pon_guassain_intensity / pon_guassain_intensity[0],
            }
        )
        roi_df = roi_df.set_index(self.delay)
        return roi_df
