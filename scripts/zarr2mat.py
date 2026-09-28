from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
import subprocess
import numpy as np
import polars as pl
import zarr
from scipy.io import savemat
from tqdm import tqdm

from roi_rectangle import RoiRectangle

from QoraFlow.logger import Logger, setup_logger
from QoraFlow.config import ConfigManager, ExpConfig
from QoraFlow.gui.select_roi import select_roi_zarr


ConfigManager.initialize("config.yaml")
config: ExpConfig = ConfigManager.load_config()
logger: Logger = setup_logger()

PARQUET_SRC = Path(
    "/xfel/usr/var-lib_for_XSS/kube_subdirs"
    "/default-jupyter-data-pvc-pvc-22c9ab9b-08d0-40a0-9d0e-d3231fe4032c"
    "/.cache/lance_analyzer"
)

# ============================================================
# Configuration
# ============================================================

EXP_NAME = "ue_260903_FXS"

RUN_N = 161
SCAN_N = 1
PUMP_RATE = 30

DETECTOR_KEY = "det-eh1-jungfrau2"
QBPM_KEY = "qbpm_eh1_qbpm1"

# QBPM intensity per shot, taken from the DAQ pipeline's own integral rather
# than re-integrating the raw waveform out of zarr. The waveform is still in
# the store (root[QBPM_KEY]) if it is ever needed, but reading it costs
# ~0.2 s/step and the previous in-script integration split the waveform at
# sample 7400, in the middle of the pulse (the pulse runs ~5150-13240), so
# its "background" window contained signal and the result came out
# proportional to *minus* the true integral.
QBPM_COLUMN = f"{QBPM_KEY}:integral_total"

# Parquet columns used as alignment keys / cross-checks.
PULSE_ID_COLUMN = "pulseId"
SCAN_STEP_COLUMN = f"{EXP_NAME}:scan:step"      # 1-based, zarr step is 0-based

# Metadata column defining pump ON/OFF.
# PUMP_RATE = 0 -> no pump, every shot is treated as OFF.
PUMP_COLUMN = f"RATE_HX_{PUMP_RATE}HZ" if PUMP_RATE != 0 else None

# Abort instead of writing near-empty MAT files when the parquet cache
# does not cover the whole zarr scan.
MIN_COVERAGE = 0.99

# scipy savemat (v5) limit is ~2 GiB per variable.
MAT_V5_LIMIT = 2 * 1024**3

# The zarr store is sharded (detector shard = one 64x64 tile x 300 shots), so
# every read is a fan-out of small byte-range reads over NFS. Worth ~2% on the
# ROI reads alone; it halves the time of any large read such as the QBPM
# waveform, should that ever be needed again.
ZARR_CONCURRENCY = 32

# Read step N+1 while step N is being written. Skipped for very large ROIs,
# where holding two steps in memory costs more than the overlap saves.
PREFETCH_LIMIT = 512 * 1024**2

zarr.config.set({"async.concurrency": ZARR_CONCURRENCY})


def sync_parquet_cache(dst_dir: Path) -> None:
    """
    Mirror the DAQ parquet caches locally with rsync.

    The DAQ cache is rewritten in place while a run is going on, so a local
    copy taken mid-run is a truncated prefix of the final file. rsync's
    quick check (size + mtime) catches exactly that case, writes through a
    temporary file so an interrupted transfer never leaves a half-written
    parquet in place, and skips unchanged files for free.

    Ownership and permissions are deliberately not preserved: the source is
    root-owned and this runs as a normal user.
    """
    if not PARQUET_SRC.is_dir():
        logger.warning(f"Parquet source not available: {PARQUET_SRC}")
        return

    dst_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "rsync",
        "--recursive",
        "--times",           # keep mtimes so the quick check works next run
        "--omit-dir-times",
        "--human-readable",
        "--out-format=synced %n (%l bytes)",
        "--stats",
        f"--include={EXP_NAME}_*.parquet",
        "--exclude=*",
        f"{PARQUET_SRC}/",
        f"{dst_dir}/",
    ]

    logger.info(f"rsync {PARQUET_SRC} -> {dst_dir}")

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    for line in result.stdout.splitlines():
        if ".parquet" in line or "Number of regular files transferred" in line:
            logger.info(line.strip())

    if result.returncode != 0:
        raise RuntimeError(
            f"rsync failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )


# ============================================================
# Metadata
# ============================================================

def get_metadata(
    index: np.ndarray,
    metadata_all: pl.DataFrame,
    step: int,
) -> pl.DataFrame:
    """
    Match one scan step's Zarr timestamps with cached Parquet metadata.

    Zarr:
        index.shape = (steps, shots, 2)
        index[..., 0] = timestamp
        index[..., 1] = pulse_id

    The join is order-preserving on the Zarr side, so `original_image_index`
    stays monotonically increasing and the returned rows line up row-by-row
    with the images that get loaded for those indices.
    """
    image_metadata = pl.DataFrame(
        {
            "timestamp": index[:, 0],
            "original_image_index": np.arange(len(index), dtype=np.int64),
            "pulse_id_zarr": index[:, 1],
        }
    )

    metadata = image_metadata.join(
        metadata_all,
        on="timestamp",
        how="inner",
        maintain_order="left",
    )

    # A duplicated timestamp in the parquet cache would silently duplicate
    # image rows; catch it instead of shipping misaligned data.
    if metadata.height > image_metadata.height:
        raise ValueError(
            f"step {step}: join produced {metadata.height} rows from "
            f"{image_metadata.height} shots - duplicated timestamps "
            f"in the parquet cache"
        )

    # Independent cross-check: the pulse id stored in the zarr index must
    # agree with the pulse id of the parquet row it matched.
    if PULSE_ID_COLUMN in metadata.columns and metadata.height:
        mismatched = metadata.filter(
            pl.col("pulse_id_zarr") != pl.col(PULSE_ID_COLUMN)
        )

        if mismatched.height:
            raise ValueError(
                f"step {step}: {mismatched.height} shot(s) matched a parquet "
                f"row with a different pulse id - timestamp join is not a "
                f"valid alignment key for this run"
            )

    return metadata


# ============================================================
# Pump ON/OFF
# ============================================================

def get_pump_indices(
    metadata: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return image indices for pump ON and OFF."""
    all_indices = metadata["original_image_index"].to_numpy().astype(np.int64)

    # No pump: everything is OFF.
    if PUMP_COLUMN is None:
        return np.empty((0,), dtype=np.int64), all_indices

    if PUMP_COLUMN not in metadata.columns:
        raise KeyError(
            f"Pump column '{PUMP_COLUMN}' not found.\n"
            f"Available columns:\n{metadata.columns}"
        )

    pump = metadata[PUMP_COLUMN]

    if pump.dtype == pl.Boolean:
        pump_on = pump
    else:
        pump_on = pump.cast(pl.Float64, strict=False).fill_null(0) != 0

    on = (
        metadata.filter(pump_on)["original_image_index"]
        .to_numpy()
        .astype(np.int64)
    )
    off = (
        metadata.filter(~pump_on)["original_image_index"]
        .to_numpy()
        .astype(np.int64)
    )

    return on, off


def get_column(
    metadata: pl.DataFrame,
    column: str,
    indices: np.ndarray,
) -> np.ndarray:
    """Values of `column` for the rows whose image index is in `indices`."""
    if column not in metadata.columns:
        return np.empty((0,), dtype=np.int64)

    return (
        metadata.filter(pl.col("original_image_index").is_in(indices))[column]
        .to_numpy()
    )


# ============================================================
# Load ROI images
# ============================================================

def load_roi_images(
    detector: zarr.Array,
    step: int,
    roi: RoiRectangle,
) -> np.ndarray:
    """
    Load every shot of one scan step, cropped to the ROI.

    Detector: (steps, shots, 512, 1024), sharded as (1, 300, 64, 64).
    Output:   (shots, roi_y, roi_x)

    Only the 64x64 tiles the ROI overlaps are fetched from each shard, so a
    32x32 ROI costs ~0.4 s/step against ~9 s/step for full frames. Pump ON
    and OFF are sliced out of this array afterwards rather than read
    separately, which avoids parsing the shard index twice.
    """
    ny, nx = roi.y2 - roi.y1, roi.x2 - roi.x1

    images = np.asarray(
        detector[step, :, roi.y1:roi.y2, roi.x1:roi.x2],
        dtype=np.float32,
    )

    return images.reshape(-1, ny, nx)


# ============================================================
# Read one scan step (I/O only, safe to run in a worker thread)
# ============================================================

def read_step(
    root: zarr.Group,
    detector: zarr.Array,
    roi: RoiRectangle,
    step: int,
) -> dict[str, np.ndarray]:
    """All Zarr I/O for one scan step, for every shot."""
    return {
        "index": np.asarray(root["index"][step], dtype=np.int64),
        "images": load_roi_images(detector, step, roi),
    }


# ============================================================
# Process one scan step
# ============================================================

def process_step(
    payload: dict[str, np.ndarray],
    metadata_all: pl.DataFrame,
    step: int,
    n_shots: int,
) -> dict[str, np.ndarray]:
    """
    Assemble one scan step's MAT payload. Pure CPU: `payload` already holds
    every shot, so ON/OFF are plain array slices.
    """
    metadata = get_metadata(payload["index"], metadata_all, step)

    if metadata.height != n_shots:
        logger.warning(
            f"step {step}: only {metadata.height}/{n_shots} shots "
            f"matched the parquet cache"
        )

    if SCAN_STEP_COLUMN in metadata.columns and metadata.height:
        scan_steps = metadata[SCAN_STEP_COLUMN].unique().to_list()

        # Parquet counts steps from 1, the zarr axis from 0.
        if scan_steps != [step + 1]:
            logger.warning(
                f"step {step}: matched parquet rows carry scan step "
                f"{scan_steps}, expected [{step + 1}]"
            )

    on_indices, off_indices = get_pump_indices(metadata)

    logger.info(
        f"step {step} | matched={metadata.height} | "
        f"ON={len(on_indices)}, OFF={len(off_indices)}"
    )

    images = payload["images"]

    # Dropped detector frames arrive as all-NaN tiles and survive the
    # timestamp join, so they have to be reported explicitly.
    n_dropped = int(np.isnan(images).all(axis=(1, 2)).sum())

    if n_dropped:
        logger.warning(
            f"step {step}: {n_dropped}/{n_shots} frame(s) are entirely NaN "
            f"in the ROI (dropped by the detector)"
        )

    return {
        "pon": images[on_indices],
        "pon_qbpm": get_column(metadata, QBPM_COLUMN, on_indices),
        "pon_pulse_id": get_column(metadata, "pulse_id_zarr", on_indices),
        "pon_timestamp": get_column(metadata, "timestamp", on_indices),
        "poff": images[off_indices],
        "poff_qbpm": get_column(metadata, QBPM_COLUMN, off_indices),
        "poff_pulse_id": get_column(metadata, "pulse_id_zarr", off_indices),
        "poff_timestamp": get_column(metadata, "timestamp", off_indices),
        "n_shots_zarr": np.int64(n_shots),
        "n_shots_matched": np.int64(metadata.height),
    }


# ============================================================
# Alignment pre-check
# ============================================================

def check_coverage(
    root: zarr.Group,
    metadata_all: pl.DataFrame,
    n_steps: int,
    n_shots: int,
) -> None:
    """
    Verify the parquet cache covers the whole zarr scan before any step is
    processed. A truncated cache otherwise produces silently empty MAT files.

    Row counts per scan step answer that for the price of a column scan.
    Reading every zarr timestamp instead would cost ~15 s (the index array is
    sharded one tiny chunk per shot), and the exact per-step timestamp match
    is checked anyway in `get_metadata` as each step is processed.
    """
    if SCAN_STEP_COLUMN not in metadata_all.columns:
        logger.warning(
            f"'{SCAN_STEP_COLUMN}' missing; falling back to a full "
            f"timestamp scan of the zarr index"
        )

        ts_zarr = np.asarray(root["index"][:], dtype=np.int64)[..., 0]
        per_step = np.isin(
            ts_zarr, metadata_all["timestamp"].to_numpy()
        ).sum(axis=1)
    else:
        counts = dict(
            metadata_all
            .group_by(SCAN_STEP_COLUMN)
            .len()
            .iter_rows()
        )

        # Parquet counts steps from 1, the zarr axis from 0.
        per_step = np.array(
            [counts.get(step + 1, 0) for step in range(n_steps)],
            dtype=np.int64,
        )

    total = int(per_step.sum())
    expected = n_steps * n_shots
    coverage = total / expected

    empty_steps = np.flatnonzero(per_step == 0)
    partial_steps = np.flatnonzero((per_step > 0) & (per_step < n_shots))

    logger.info(
        f"parquet coverage: {total}/{expected} shots ({coverage:.3%}), "
        f"{len(empty_steps)} empty step(s), "
        f"{len(partial_steps)} partial step(s)"
    )

    if len(empty_steps):
        logger.warning(
            f"steps with no parquet rows at all: "
            f"{empty_steps.min()}..{empty_steps.max()} "
            f"({len(empty_steps)} of {n_steps})"
        )

    if coverage < MIN_COVERAGE:
        raise ValueError(
            f"Parquet cache covers only {coverage:.1%} of the zarr shots "
            f"({total}/{expected}). The local cache is most likely a "
            f"truncated mid-run copy - check that\n"
            f"  {PARQUET_SRC}\n"
            f"was actually re-synced (compare file sizes), then re-run."
        )

    # Cheap sanity check that the timestamps really do line up, on the two
    # steps most likely to be wrong if the cache belongs to another run.
    for step in {0, n_steps - 1}:
        ts_zarr = np.asarray(root["index"][step], dtype=np.int64)[:, 0]
        matched = int(np.isin(ts_zarr, metadata_all["timestamp"].to_numpy()).sum())

        if matched < n_shots:
            logger.warning(
                f"step {step}: {matched}/{n_shots} timestamps found in the "
                f"parquet cache despite a full row count"
            )


# ============================================================
# Main
# ============================================================

def main() -> None:
    load_dir: Path = config.path.load_dir

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    scan_dir = (
        load_dir
        / "archive"
        / f"run{RUN_N:05}"
        / f"scan{SCAN_N:05}"
    )

    zarr_file = scan_dir / "raw.zarr"

    if not zarr_file.exists():
        raise FileNotFoundError(f"Zarr not found:\n{zarr_file}")

    parquet_file = (
        load_dir
        / "parquet_cache"
        / f"{EXP_NAME}_cache_run_{RUN_N}.parquet"
    )
    sync_parquet_cache(parquet_file.parent)
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet file not found:\n{parquet_file}")

    # Parquet is read ONCE, not per step.
    metadata_all = pl.read_parquet(parquet_file)

    for column in ("timestamp", QBPM_COLUMN):
        if column not in metadata_all.columns:
            raise KeyError(f"'{column}' not found in {parquet_file}")

    n_null = metadata_all[QBPM_COLUMN].null_count()

    if n_null:
        logger.warning(
            f"{QBPM_COLUMN}: {n_null}/{metadata_all.height} null value(s); "
            f"they reach the MAT file as NaN"
        )

    n_duplicate = metadata_all.height - metadata_all["timestamp"].n_unique()

    if n_duplicate:
        raise ValueError(
            f"{parquet_file.name}: {n_duplicate} duplicated timestamp(s); "
            f"cannot be used as a join key"
        )

    logger.info(
        f"{parquet_file.name}: {metadata_all.height} rows, "
        f"{parquet_file.stat().st_size:,} bytes"
    )

    # --------------------------------------------------------
    # Open Zarr and validate axes
    # --------------------------------------------------------

    root = zarr.open(zarr_file, mode="r")

    if DETECTOR_KEY not in root:
        raise KeyError(f"{DETECTOR_KEY} not found in {zarr_file}")

    detector = root[DETECTOR_KEY]
    n_steps, n_shots = detector.shape[0], detector.shape[1]

    for key in ("index", QBPM_KEY):
        if key not in root:
            raise KeyError(f"{key} not found in {zarr_file}")

        if root[key].shape[:2] != detector.shape[:2]:
            raise ValueError(
                f"{key} shape {root[key].shape[:2]} != "
                f"detector {detector.shape[:2]}"
            )

    logger.info(
        f"{zarr_file.name}: detector shape={detector.shape}, "
        f"n_steps={n_steps}, n_shots={n_shots}, "
        f"pump_column={PUMP_COLUMN}"
    )

    # Fail here, before the ROI dialog and hours of I/O.
    check_coverage(root, metadata_all, n_steps, n_shots)

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------
    roi_step = n_steps // 2
    roi_rect: RoiRectangle = select_roi_zarr(zarr_file, roi_step)

    logger.info(f"ROI (from step {roi_step}): {roi_rect}")

    ny, nx = roi_rect.y2 - roi_rect.y1, roi_rect.x2 - roi_rect.x1

    est_bytes = n_shots * ny * nx * 4

    if est_bytes > MAT_V5_LIMIT:
        raise ValueError(
            f"ROI {ny}x{nx} x {n_shots} shots = "
            f"{est_bytes / 1024**3:.2f} GiB per variable, "
            f"over the ~2 GiB MAT v5 limit. Use a smaller ROI."
        )

    logger.info(
        f"ROI {ny}x{nx}: up to {est_bytes / 1024**3:.3f} GiB per variable, "
        f"~{n_steps * est_bytes / 1024**3:.1f} GiB total"
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    mat_root = (
        config.path.analysis_dir
        / "crop_mat"
        / f"run{RUN_N}"
    )

    mat_root.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # One MAT file per scan step
    # ========================================================

    total_saved = 0

    # Overlap the next step's Zarr reads with this step's MAT write.
    prefetch = est_bytes <= PREFETCH_LIMIT

    logger.info(
        f"prefetch {'enabled' if prefetch else 'disabled'} "
        f"({est_bytes / 1024**2:.0f} MiB per step in flight)"
    )

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending: Future | None = (
            pool.submit(read_step, root, detector, roi_rect, 0)
            if prefetch
            else None
        )

        for step in tqdm(range(n_steps), desc="scan steps"):
            if pending is not None:
                payload = pending.result()

                pending = (
                    pool.submit(read_step, root, detector, roi_rect, step + 1)
                    if step + 1 < n_steps
                    else None
                )
            else:
                payload = read_step(root, detector, roi_rect, step)

            result = process_step(payload, metadata_all, step, n_shots)

            del payload

            result["step_index"] = np.int64(step)
            result["roi"] = {
                "y1": roi_rect.y1,
                "y2": roi_rect.y2,
                "x1": roi_rect.x1,
                "x2": roi_rect.x2,
            }

            total_bytes = 0

            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    total_bytes += value.nbytes

                    logger.info(
                        f"{key}: shape={value.shape}, dtype={value.dtype}, "
                        f"GiB={value.nbytes / 1024**3:.3f}"
                    )

            logger.info(
                f"step {step} TOTAL: {total_bytes:,} bytes "
                f"({total_bytes / 1024**3:.3f} GiB)"
            )

            mat_file = mat_root / (
                f"run={RUN_N:04}_scan={SCAN_N:04}_step={step:04}.mat"
            )

            savemat(mat_file, result, do_compression=False)

            logger.info(f'"{mat_file}" has saved')

            total_saved += 1

            del result

    logger.info(f"Finished: {total_saved}/{n_steps} steps saved.")


if __name__ == "__main__":
    main()
