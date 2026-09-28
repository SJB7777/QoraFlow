from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.io import loadmat

from QoraFlow.logger import Logger, setup_logger
from QoraFlow.config import ConfigManager, ExpConfig


ConfigManager.initialize("config.yaml")
config: ExpConfig = ConfigManager.load_config()
logger: Logger = setup_logger()

RUN_N =74
SCAN_N = 1


def draw_images(
    ax,
    image: np.ndarray,
    label: str,
    n_steps: int,
    n_shots: int,
    fig,
) -> None:
    """Show image on a log color scale with original-value ticks."""
    positive = image[np.isfinite(image) & (image > 0)]

    if positive.size == 0:
        logger.warning(f"{label}: no positive pixels, using linear scale")
        norm = None
    else:
        norm = LogNorm(
            vmin=np.percentile(positive, 1),
            vmax=np.percentile(positive, 99.9),
        )

    im = ax.imshow(image, norm=norm)

    ax.set_title(f"{label} ({n_steps} steps, {n_shots} shots)")
    fig.colorbar(im, ax=ax)


def main() -> None:
    mat_root = (
        config.path.analysis_dir
        / "crop_mat"
        / f"run{RUN_N}"
    )

    files = sorted(
        mat_root.glob(f"run={RUN_N:04}_scan={SCAN_N:04}_step=*.mat")
    )

    if not files:
        raise FileNotFoundError(f"No step files in {mat_root}")

    logger.info(f"Found {len(files)} step files")

    # Per-step shot means and the shot count behind each mean.
    stacks: dict[str, list[np.ndarray]] = {"pon": [], "poff": []}
    weights: dict[str, list[int]] = {"pon": [], "poff": []}

    for f in tqdm(files):
        mat = loadmat(f)

        for key in ("pon", "poff"):
            images = mat[key]

            if images.size == 0:
                continue

            stacks[key].append(np.nanmean(images, axis=0))
            weights[key].append(images.shape[0])

    # Keep only the keys that actually have data.
    present = [key for key in ("pon", "poff") if stacks[key]]

    if not present:
        raise ValueError("both pon and poff are empty in every step")

    for key in ("pon", "poff"):
        if key not in present:
            logger.warning(f"{key}: empty in every step, not plotted")

    fig, axes = plt.subplots(
        1,
        len(present),
        figsize=(6.5 * len(present), 5),
        squeeze=False,
    )

    for ax, key in zip(axes[0], present):
        per_step = stacks[key]
        w = np.asarray(weights[key], dtype=np.float64)

        logger.info(
            f"{key}: {len(per_step)}/{len(files)} steps, "
            f"{int(w.sum())} shots total, "
            f"shots/step min={int(w.min())} max={int(w.max())}"
        )

        # (steps, ny, nx) -> (ny, nx), weighted by shots per step
        z_stack = np.stack(per_step, axis=0)
        mean_image = np.average(z_stack, axis=0, weights=w)

        draw_images(
            ax,
            mean_image,
            key,
            len(per_step),
            int(w.sum()),
            fig,
        )

    fig.suptitle(f"run={RUN_N} scan={SCAN_N} | {len(files)} steps")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
