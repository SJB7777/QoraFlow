import sys
from abc import ABC, abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import hdf5plugin
import numpy as np
import numpy.typing as npt
import pandas as pd

from ..config import ConfigManager, ExpConfig
from ..config.enums import Hertz
from ..logger import Logger, setup_logger


class RawDataLoader(ABC):
    @abstractmethod
    def __init__(self, file: Path | str) -> None:
        """Initialize"""

    @abstractmethod
    def get_data(self) -> dict[str, npt.NDArray]:
        """Retrieve data"""
    
    @abstractmethod
    def get_chunked_data(self, chunk_size:int) -> Iterator[dict[str, npt.NDArray]]:
        """Yield Data"""

    def close(self):
        """Optional close method"""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class PalXFELLoader(RawDataLoader):
    """Load hdf5 file and remove unmatching data, supporting lazy image loading."""

    def __init__(self, file: Path | str):
        """
        Initializes the Loader. Data arrays (images) are loaded lazily in get_data().
        """
        self.file: Path = Path(file)
        if not self.file.exists():
            raise FileNotFoundError(f"No such file: {self.file!s}")

        self.logger: Logger = setup_logger()
        self.config: ExpConfig = ConfigManager.load_config()

        self._h5_file = h5py.File(self.file, "r")
        self._image_path = f"detector/{self.config.param.hutch.value}/{self.config.param.detector.value}/image"
        self._qbpm_path = f"qbpm/{self.config.param.hutch.value}/qbpm1"

        metadata: pd.DataFrame = pd.read_hdf(self.file, key="metadata")
        merged_df: pd.DataFrame = self._get_metadata_and_qbpm_df(metadata)

        if merged_df.empty:
            self.close()
            raise ValueError(f"No matching data found in {self.file}")

        self.qbpm: npt.NDArray[np.float32] = np.array(merged_df["qbpm"].tolist(), dtype=np.float32)
        self.pump_state: npt.NDArray[np.bool_] = self._get_pump_mask(merged_df)
        self.delay: np.float64 | float = self._get_delay(merged_df)

        raw_indices = merged_df["original_image_index"].values.astype(np.int64)
        self.metadata_index: npt.NDArray[np.int64] = np.sort(raw_indices)

        self.logger.debug(
            f"Initialized {len(self.metadata_index)} indices for images and {len(self.qbpm)} qbpm data."
        )

    def close(self):
            """Explicitly close the HDF5 file handle."""
            if hasattr(self, "_h5_file") and self._h5_file:
                try:
                    self._h5_file.close()
                except Exception:
                    pass
                self._h5_file = None

    def __del__(self):
            self.close()

    def _get_metadata_and_qbpm_df(self, metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Merges metadata, image timestamps, and qbpm data based on timestamps.
        Critically, it loads only the QBPM *values* into memory, not images.
        """
        hf = self._h5_file

        if "detector" not in hf:
            raise KeyError(f"Key 'detector' not found in {self.file}")

        image_group = hf[self._image_path]
        images_ts = np.array(image_group["block0_items"], dtype=np.int64)

        qbpm_group = hf[self._qbpm_path]
        qbpm_ts = np.array(qbpm_group["waveforms.ch1/axis1"], dtype=np.int64)
        # qbpm = np.sum(
        #     np.stack(
        #         [qbpm_group[f"waveforms.ch{i + 1}/block0_values"] for i in range(4)],
        #         axis=0,
        #         dtype=np.float32,
        #     ),
        #     axis=(0, 2),
        # )

        qbpm = 0
        for channel in range(1, 5):
            qbpm_channel = qbpm_group[f"waveforms.ch{channel}/block0_values"][()]
            background = qbpm_channel[:, :7400].mean(axis=1)
            qbpm_channel_new = (qbpm_channel - background[:, np.newaxis])[:, 7400:]
            qbpm += qbpm_channel_new.sum(axis=1)

        image_ts_df = pd.DataFrame(
            {
                "original_image_index": np.arange(len(images_ts), dtype=np.int64)
            }, 
            index=images_ts
        )
        qbpm_df = pd.DataFrame({"qbpm": qbpm}, index=qbpm_ts)

        merged_df = image_ts_df.join(qbpm_df, how="inner")

        return metadata.join(merged_df, how="inner")

    def _get_delay(self, merged_df: pd.DataFrame) -> np.float64 | float:
        """Retrieves the delay value."""
        if "th_value" in merged_df:
            return np.asarray(merged_df["th_value"], dtype=np.float64)[0]
        if "delay_value" in merged_df:
            return np.asarray(merged_df["delay_value"], dtype=np.float64)[0]
        return np.nan

    def _get_pump_mask(self, merged_df: pd.DataFrame) -> npt.NDArray[np.bool_]:
        """Generates a pump status mask."""
        if self.config.param.pump_setting is Hertz.ZERO:
            return np.zeros(merged_df.shape[0], dtype=np.bool_)
        return np.asarray(
            merged_df[
                f"timestamp_info.RATE_{self.config.param.xray.value}_{self.config.param.pump_setting.value}"
            ],
            dtype=np.bool_,
        )

    def get_chunked_data(self, chunk_size: int = 100) -> Iterator[dict[str, Any]]:
        """
        [Optimized Method]
        Yields data in small chunks to prevent OOM errors.
        This is the preferred method for large-scale processing.
        """
        if not self._h5_file:
            self._h5_file = h5py.File(self.file, "r")

        image_dataset = self._h5_file[self._image_path]["block0_values"]
        total_frames = len(self.metadata_index)

        # 1. Chunking Loop
        for start_idx in range(0, total_frames, chunk_size):
            end_idx = min(start_idx + chunk_size, total_frames)
            
            # 2. Slice Indices for current chunk
            current_indices = self.metadata_index[start_idx:end_idx]

            # 3. Load ONLY the current chunk of images from Disk -> RAM
            # Note: HDF5 random access is used here.
            images_chunk = image_dataset[current_indices]

            # 4. Slice Metadata (Already in RAM, fast)
            current_pump_state = self.pump_state[start_idx:end_idx]
            current_qbpm = self.qbpm[start_idx:end_idx]

            # 5. Prepare Output
            data_chunk: dict[str, Any] = {"delay": self.delay}

            poff_mask = ~current_pump_state
            pon_mask = current_pump_state

            # Apply masks within the chunk
            if np.any(poff_mask):
                data_chunk["poff"] = images_chunk[poff_mask]
                data_chunk["poff_qbpm"] = current_qbpm[poff_mask]
            
            if np.any(pon_mask):
                data_chunk["pon"] = images_chunk[pon_mask]
                data_chunk["pon_qbpm"] = current_qbpm[pon_mask]

            yield data_chunk

    def get_data(self) -> dict[str, npt.NDArray]:
        """
        Retrieves data by loading images from HDF5 lazily.
        Returns: Images and qbpm data for both pump-on and pump-off states.
        """
        if not self._h5_file:
            self._h5_file = h5py.File(self.file, "r")

        image_block_data = self._h5_file[self._image_path]["block0_values"]

        images: npt.NDArray = image_block_data[self.metadata_index]
        
        data: dict[str, npt.NDArray] = {"delay": self.delay}

        poff_images = images[~self.pump_state]
        poff_qbpm = self.qbpm[~self.pump_state]
        pon_images = images[self.pump_state]
        pon_qbpm = self.qbpm[self.pump_state]

        if poff_images.size > 0:
            data["poff"] = poff_images
            data["poff_qbpm"] = poff_qbpm
        if pon_images.size > 0:
            data["pon"] = pon_images
            data["pon_qbpm"] = pon_qbpm
        data["images"] = images
        return data


def get_hdf5_images(file: str, config: ExpConfig) -> npt.NDArray:
    """get images form hdf5"""
    with h5py.File(file, "r") as hf:
        if "detector" not in hf:
            raise KeyError(f"Key 'detector' not found in {file}")
        key: str = f"detector/{config.param.hutch.value}/{config.param.detector.value}/image/block0_values"
        images: np.ndarray = hf[key][:]

        return np.maximum(images, 0)


if __name__ == "__main__":
    import time

    from QoraFlow.filesystem import get_run_scan_dir
    ConfigManager.initialize("config.yaml")
    config: ExpConfig = ConfigManager.load_config()
    load_dir: Path = config.path.load_dir
    # 테스트할 파일 경로 확인 (예시)
    try:
        file: Path = get_run_scan_dir(load_dir, 165, 1, sub_path="p0005.h5")
        if not file.exists():
            print(f"File not found: {file}")
    except Exception as e:
        print(f"Path Error: {e}")
        sys.exit()

    print(f"Target: {file}")

    # 1. 기존 방식 테스트
    print("-" * 30)
    print("[Test 1] Legacy get_data() (High Memory)")
    start_total = time.time()
    with PalXFELLoader(file) as loader:
        data = loader.get_data()
        if 'poff' in data:
            print(f"Full Load POFF: {data['poff'].shape}")
    print(f"Time: {time.time() - start_total:.4f} sec")

    # 2. 최적화 방식 테스트
    print("-" * 30)
    print("[Test 2] Optimized get_chunked_data() (Low Memory)")
    start_total = time.time()
    chunk_count = 0
    with PalXFELLoader(file) as loader:
        # 100장씩 잘라서 로드
        for chunk in loader.get_chunked_data(chunk_size=100):
            chunk_count += 1
            if 'poff' in chunk:
                print(f"  Chunk {chunk_count} POFF: {chunk['poff'].shape}")
            if 'pon' in chunk:
                print(f"  Chunk {chunk_count} PON: {chunk['pon'].shape}")
    print(f"Time: {time.time() - start_total:.4f} sec")
    print("-" * 30)
