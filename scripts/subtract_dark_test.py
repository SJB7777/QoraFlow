import matplotlib.pyplot as plt
import numpy as np

from QoraFlow.filesystem import get_run_scan_dir
from QoraFlow.integrator.loader import PalXFELLoader

# ConfigManager.initialize(r"D:\Members\IsaacYong\Dev\QoraFlow\config.yaml")
# config = ConfigManager.load_config()
load_dir = r"X:\251128_FXS\raw_data\h5\type=raw"
file = get_run_scan_dir(load_dir, 121, 1, sub_path="p0060.h5")
loader = PalXFELLoader(file)
data = loader.get_data()
images = data["images"]
raw_imgs = images
# raw_imgs = np.maximum(images, 0)
# dark_file = r"D:\Members\JiseongOh\251128_XFEL\analysis\rocking\dark_images\dark.npy"
dark_file = r"C:\Users\user\Downloads\dark.npy"
dark_image = np.load(dark_file)
if dark_image.ndim == 3:
    dark_image = np.mean(dark_image, axis=0)
dark_image = np.maximum(dark_image, 0)
subtracted_imgs = raw_imgs - dark_image[None, ...]
# subtracted_imgs = np.maximum(subtracted_imgs, 0)

global_vmax = max(raw_imgs.max(), subtracted_imgs.max())

fig, ax = plt.subplots(3, 1, figsize=(8, 12), layout="constrained")
ax[0].set_title("Raw Image")
im0 = ax[0].imshow(np.log1p(raw_imgs.mean(0)))
ax[1].set_title("Dark Image")
im1 = ax[1].imshow(np.log1p(dark_image))
ax[2].set_title("Dark Subtracted Image")
im2 = ax[2].imshow(np.log1p(subtracted_imgs.mean(0)))
cbar0 = fig.colorbar(im0, ax=ax[0], location='right', aspect=50)
cbar1 = fig.colorbar(im1, ax=ax[1], location='right', aspect=50)
cbar2 = fig.colorbar(im2, ax=ax[2], location='right', aspect=50)
plt.show()
