import numpy as np

from scipy.io import savemat
from QoraFlow.logger import Logger, setup_logger
from QoraFlow.config import ConfigManager, ExpConfig
from QoraFlow.integrator.loader import PalXFELLoader
from QoraFlow.filesystem import get_run_scan_dir

ConfigManager.initialize("config.yaml")
config: ExpConfig = ConfigManager.load_config()
logger: Logger = setup_logger()

def main() -> None:
    dark_num: int = 171
    dark_file_1 = get_run_scan_dir(config.path.load_dir, dark_num, 1, sub_path='p0001.h5')
    dark_file_2 = get_run_scan_dir(config.path.load_dir, dark_num, 1, sub_path='p0002.h5')

    config.path.save_dark_dir.mkdir(parents=True, exist_ok=True)
    dark_file = config.path.save_dark_dir.with_suffix(".mat")
    print(dark_file)
    loader1 = PalXFELLoader(dark_file_1)
    data1 = loader1.get_data()
    loader2 = PalXFELLoader(dark_file_2)
    data2 = loader2.get_data()
    dark = np.concat([data1["pon"], data2["pon"]], axis=0)

    print(dark.shape)
    savemat(dark_file, {"image": dark})

if __name__ == "__main__":
    main()
