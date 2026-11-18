"""Load GrainSet image lists without relying on the current working directory."""
from pathlib import Path
import pandas as pd

__all__ = ["get_imglists"]

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_imglists(root, split="train", phase="train"):
    """Read a split list from datasets/datalist (or legacy runs/datalist)."""
    data_root = Path(root).expanduser().resolve()
    grain = data_root.name
    selected_split = phase if split == "train" else split
    name = f"{grain}_{selected_split}.txt"

    candidates = [
        _PROJECT_ROOT / "datasets" / "datalist" / name,
        _PROJECT_ROOT / "runs" / "datalist" / name,
    ]
    datalist = next((p for p in candidates if p.is_file()), None)
    if datalist is None:
        locations = "\n - ".join(str(p) for p in candidates)
        raise FileNotFoundError(
            f"Could not find {name!r}. Searched:\n - {locations}"
        )

    images, labels = [], []
    with datalist.open("r", encoding="utf-8-sig") as stream:
        for number, raw in enumerate(stream, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(
                    f"Malformed line {number} in {datalist}: "
                    "expected '<image_path> <class_label>'"
                )
            filename, label = parts
            try:
                labels.append(int(label))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid class label on line {number} of {datalist}"
                ) from exc
            images.append(str(data_root / filename))

    print(f"* {selected_split} : {len(images)} ({datalist})")
    return pd.DataFrame({"filename": images, "label": labels})