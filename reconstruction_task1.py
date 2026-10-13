"""1. feladat: keprekonstrukcio szurt visszavetelessel.

A program az image.png kepet rekonstrualja egyre kevesebb, a [0, 180)
szogtartomanyon egyenletesen elosztott parhuzamos vetuletbol.
"""

from pathlib import Path

import matplotlib
import numpy as np
from skimage import color, io, transform
from skimage.transform import iradon, radon

matplotlib.use("Agg")
import matplotlib.pyplot as plt


INPUT_FILE = Path("image.png")
OUTPUT_DIR = Path("results_task1")
PROJECTION_COUNTS = (180, 160, 140, 120, 100, 80, 60, 40, 30, 20, 10, 5)
IMAGE_SIZE = 256


def load_binary_image(path: Path) -> np.ndarray:
    """Betolti, szurkearnyalatossa es 256 x 256 meretu binaris keppe alakitja."""
    image = io.imread(path)

    if image.ndim == 3:
        if image.shape[2] == 4:
            image = color.rgba2rgb(image)
        image = color.rgb2gray(image)

    image = transform.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        anti_aliasing=True,
        preserve_range=True,
    )

    # A bemeneti kepet is a teljes intenzitastartomany kozepen kuszoboljuk.
    threshold = (float(image.min()) + float(image.max())) / 2.0
    return image > threshold


def reconstruct(image: np.ndarray, projection_count: int) -> np.ndarray:
    """Szurt visszaveteles rekonstrukcio adott szamu parhuzamos vetuletbol."""
    angles = np.linspace(0.0, 180.0, projection_count, endpoint=False)
    sinogram = radon(image.astype(float), theta=angles, circle=True)

    reconstruction = iradon(
        sinogram,
        theta=angles,
        filter_name="ramp",
        interpolation="linear",
        circle=True,
        output_size=image.shape[0],
    )

    # A binaris bemenet intenzitastartomanya [0, 1], ennek kozepe 0.5.
    # Ez 8 bites abrazolasban a feladatban megadott 127/128-as hatarnak felel meg.
    return reconstruction > 0.5


def save_overview(images: list[tuple[int, np.ndarray]]) -> None:
    """Az osszes binaris rekonstrukciot egyetlen osszefoglalo abrara menti."""
    figure, axes = plt.subplots(3, 4, figsize=(10, 8))
    for axis, (projection_count, image) in zip(axes.flat, images):
        axis.imshow(image, cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"{projection_count} vetulet")
        axis.axis("off")

    figure.suptitle("Szurt visszaveteles rekonstrukciok", fontsize=14)
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "overview.png", dpi=160, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    original = load_binary_image(INPUT_FILE)
    OUTPUT_DIR.mkdir(exist_ok=True)

    results = []
    for projection_count in PROJECTION_COUNTS:
        binary_reconstruction = reconstruct(original, projection_count)
        results.append((projection_count, binary_reconstruction))

        output_path = OUTPUT_DIR / f"reconstruction_{projection_count:03d}.png"
        io.imsave(output_path, binary_reconstruction.astype(np.uint8) * 255)
        print(f"Elkeszult: {output_path}")

    save_overview(results)
    print(f"Osszefoglalo abra: {OUTPUT_DIR / 'overview.png'}")


if __name__ == "__main__":
    main()
