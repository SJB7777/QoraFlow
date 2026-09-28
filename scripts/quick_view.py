import matplotlib.pyplot as plt
import numpy as np
from roi_rectangle import RoiRectangle
import zarr

from QoraFlow.config import ConfigManager, ExpConfig
from QoraFlow.logger import Logger, setup_logger
from QoraFlow.gui.select_roi import select_roi_zarr


ConfigManager.initialize("config.yaml")
config: ExpConfig = ConfigManager.load_config()
logger: Logger = setup_logger()

run_n = 161

file = config.path.load_dir / "archive" / f"run{run_n:05}" / "scan00001" / "raw.zarr"
root = zarr.open(file, mode='r')

det = root['det-eh1-jungfrau2']
shape = det.shape
chunks = det.chunks


num_frames = det.shape[0]
mid_num = num_frames // 2

roi_rect: RoiRectangle = select_roi_zarr(file, mid_num)

samplinged = np.nanmean(det[:,::30,*roi_rect.get_slices()], axis=(0, 1))

fig, ax = plt.subplots(1, 1, figsize=(8, 6))
ax.imshow(np.log(samplinged))
ax.set_title(f"run={run_n} Quick View {roi_rect}")
plt.show()
