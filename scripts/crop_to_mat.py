from pathlib import Path
from itertools import batched

from scipy.io import savemat
import numpy as np
from tqdm import tqdm

from roi_rectangle import RoiRectangle
from QoraFlow.filesystem import get_run_scan_dir
from QoraFlow.logger import Logger, setup_logger
from QoraFlow.config import ConfigManager, ExpConfig
from QoraFlow.integrator.loader import PalXFELLoader
from QoraFlow.gui.select_roi import select_roi


ConfigManager.initialize("config.yaml")

config: ExpConfig = ConfigManager.load_config()
logger: Logger = setup_logger()


def main() -> None:
    load_dir: Path = config.path.load_dir

    run_n: int = 157
    scan_n: int = 1

    scan_dir: Path = get_run_scan_dir(load_dir, run_n, scan_n)
    pfiles: list[Path] = sorted(scan_dir.rglob("p*.h5"))

    roi_rect: RoiRectangle = select_roi(scan_dir, config, None)

    logger.info(f"ROI: {roi_rect}")

    mat_root: Path = config.path.analysis_dir / "crop_mat" / f"run={run_n:04}"
    mat_root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Process in batches
    # ------------------------------------------------------------
    pbar1 = tqdm(
        batched(pfiles, config.param.merge_num),
        desc="Batches",
        position=0,
    )

    for batch_n, files in enumerate(pbar1):

        logger.info(
            f"=== Batch {batch_n}: "
            f"{len(files)} files ==="
        )

        # --------------------------------------------------------
        # Reset result for this batch
        # --------------------------------------------------------
        result = {
            "roi": {
                "y1": roi_rect.y1,
                "y2": roi_rect.y2,
                "x1": roi_rect.x1,
                "x2": roi_rect.x2,
            }
        }

        acc_bytes = {
            "poff": 0,
            "pon": 0,
            "poff_qbpm": 0,
            "pon_qbpm": 0,
        }

        # --------------------------------------------------------
        # Load / accumulate this batch
        # --------------------------------------------------------
        for file in tqdm(
            files,
            desc=f"Batch {batch_n}",
            position=1,
            leave=False,
        ):
            if not file.exists():
                logger.warning(f"File not found: {file}")
                continue

            try:
                loader = PalXFELLoader(file)
                data = loader.get_data()

            except KeyError as e:
                logger.warning(f"{file} : {e}")
                continue

            for flag in ["poff", "pon"]:

                if flag not in data:
                    continue

                img = data[flag]
                roi_img = roi_rect.slice(img)

                result.setdefault(flag, []).append(roi_img)

                qbpm = data[f"{flag}_qbpm"]

                result.setdefault(
                    f"{flag}_qbpm",
                    [],
                ).append(qbpm)

                acc_bytes[flag] += roi_img.nbytes
                acc_bytes[f"{flag}_qbpm"] += qbpm.nbytes

            pbar1.set_postfix(
                batch=f"{batch_n}",
                size=f"{sum(acc_bytes.values()) / 1024**3:.2f} GiB",
            )
            
        # --------------------------------------------------------
        # Stack + transpose
        #
        # (A, B, Y, X) -> (Y, X, B, A)
        # --------------------------------------------------------
        for key in list(result.keys()):

            if key == "roi":
                continue

            if not result[key]:
                logger.warning(
                    f"Batch {batch_n}: {key} is empty"
                )
                continue

            result[key] = np.stack(result[key])

        # --------------------------------------------------------
        # Check batch size
        # --------------------------------------------------------
        logger.info(
            f"=== Batch {batch_n} DATA SIZE ==="
        )

        total_bytes = 0

        for key, value in result.items():

            if isinstance(value, np.ndarray):

                size = value.nbytes
                total_bytes += size

                logger.info(
                    f"{key}: "
                    f"shape={value.shape}, "
                    f"dtype={value.dtype}, "
                    f"bytes={size:,}, "
                    f"GiB={size / 1024**3:.3f}"
                )

        logger.info(
            f"Batch {batch_n} TOTAL: "
            f"{total_bytes:,} bytes "
            f"({total_bytes / 1024**3:.3f} GiB)"
        )

        # --------------------------------------------------------
        # Output filename
        # --------------------------------------------------------
        mat_file = mat_root / (
            f"run={run_n:04}" +
            f"_scan={scan_n:04}" +
            f"_batch={batch_n:04}.mat"
        )


        # --------------------------------------------------------
        # Save Mat
        # --------------------------------------------------------
        savemat(
            mat_file,
            result,
            do_compression=False,
        )

        logger.info(
            f'"{mat_file}" has saved'
        )

        # --------------------------------------------------------
        # Explicitly release batch data
        # --------------------------------------------------------
        del result


if __name__ == "__main__":
    main()