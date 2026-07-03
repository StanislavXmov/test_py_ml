from pathlib import Path

from PIL import Image


def resize_half_and_save(src: str | Path, dst: str | Path) -> Path:
    src, dst = Path(src), Path(dst)
    with Image.open(src) as img:
        w, h = img.size
        resized = img.resize((w // 2, h // 2), Image.Resampling.LANCZOS)
        dst.parent.mkdir(parents=True, exist_ok=True)
        resized.save(dst, format="PNG")
    return dst


if __name__ == "__main__":
    resize_half_and_save(
        "neural_network/image3.webp",
        "neural_network/image3.png",
    )
