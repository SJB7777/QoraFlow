import os

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib import patches
from roi_rectangle import RoiRectangle

from ..config import ConfigManager, ExpConfig
from ..filesystem import get_run_scan_dir
from ..integrator.loader import get_hdf5_images


class RoiSelector:
    def __init__(self):
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.fx, self.fy = -1, -1
        self.rect = None
        self.ax = None

    def on_mouse_press(self, event):

        if event.inaxes is not None and event.button == 1:
            self.drawing = True
            self.ix, self.iy = int(event.xdata), int(event.ydata)
            self.fx, self.fy = self.ix, self.iy
            if self.rect is not None:
                self.rect.remove()
            self.rect = patches.Rectangle(
                (self.ix, self.iy),
                1,
                1,
                linewidth=1,
                edgecolor="r",
                facecolor="none",
            )
            self.ax.add_patch(self.rect)
            plt.draw()

    def on_mouse_release(self, event):

        if event.inaxes is not None and self.drawing:
            self.drawing = False
            self.fx, self.fy = int(event.xdata), int(event.ydata)
            if self.rect is not None:
                self.rect.set_width(self.fx - self.ix)
                self.rect.set_height(self.fy - self.iy)
                plt.draw()

    def on_mouse_move(self, event):

        if event.inaxes is not None and self.drawing:
            self.fx, self.fy = int(event.xdata), int(event.ydata)
            if self.rect is not None:
                self.rect.set_width(self.fx - self.ix)
                self.rect.set_height(self.fy - self.iy)
                plt.draw()

    def select_roi(self, image: npt.NDArray) -> tuple[int, int, int, int] | None:
        if image.ndim != 2:
            raise TypeError(f"Invalid shape {image.shape} for image data")
        fig, self.ax = plt.subplots(figsize=(10, 6))
        self.ax.imshow(image)

        fig.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        fig.canvas.mpl_connect("button_release_event", self.on_mouse_release)
        fig.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)

        plt.show()

        if self.ix == -1 or self.iy == -1 or self.fx == -1 or self.fy == -1:
            return None
        x1, y1 = min(self.ix, self.fx), min(self.iy, self.fy)
        x2, y2 = max(self.ix, self.fx), max(self.iy, self.fy)
        return (y1, y2, x1, x2)


def select_roi_by_run_scan(
    run: int, scan: int, index_mode: int | None = None
) -> RoiRectangle | None:
    config: ExpConfig = ConfigManager.load_config()
    load_dir = config.path.load_dir
    scan_dir = get_run_scan_dir(load_dir, run, scan)
    files = os.listdir(scan_dir)

    if index_mode is None:
        index = len(files) // 2
    elif isinstance(index_mode, int):
        index = index_mode

    images = get_hdf5_images(os.path.join(scan_dir, files[index]), config)
    image = np.log1p(images.sum(axis=0))
    image = np.maximum(0, image)
    roi = RoiSelector().select_roi(image)
    if roi is None:
        return None
    return RoiRectangle(*roi)


def get_roi_auto(
    image,
    half_width: int = 5,
) -> RoiRectangle:
    """get roi_rect by max pixel"""
    cy, cx = np.unravel_index(np.argmax(image), image.shape)
    return RoiRectangle(
        cy - half_width, cy + half_width, cx - half_width, cx + half_width
    )

if __name__ == "__main__":
    roi_rect = select_roi_by_run_scan(144, 1, 0)
    print(roi_rect)
