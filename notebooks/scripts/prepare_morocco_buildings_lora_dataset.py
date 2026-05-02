import argparse
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face building segmentation dataset into LoRA image/mask pairs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tferhan/morocco_satellite_buildings_semantic_segmentation_512_v2",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-column", type=str, default=None)
    parser.add_argument("--mask-column", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="satellite view of small houses, roof geometry, driveways, residential block")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--invert-mask", action="store_true")
    return parser.parse_args()


def choose_column(row, explicit_name, candidates, description):
    if explicit_name is not None:
        if explicit_name not in row:
            raise KeyError(f"{description} column '{explicit_name}' not found. Available columns: {list(row.keys())}")
        return explicit_name

    for candidate in candidates:
        if candidate in row:
            return candidate

    image_like_keys = [key for key, value in row.items() if is_image_like(value)]
    if len(image_like_keys) == 1:
        return image_like_keys[0]

    raise KeyError(
        f"Could not infer {description} column. Available columns: {list(row.keys())}. "
        f"Pass --{description.replace(' ', '-').replace('/', '-')}-column explicitly."
    )


def is_image_like(value):
    if isinstance(value, Image.Image):
        return True
    if isinstance(value, np.ndarray):
        return value.ndim in {2, 3}
    if isinstance(value, list):
        array = np.asarray(value)
        return array.ndim in {2, 3}
    return False


def as_pil_image(value, mode_hint):
    if isinstance(value, Image.Image):
        image = value
    else:
        array = np.asarray(value)
        if array.ndim not in {2, 3}:
            raise TypeError(f"Expected image-like data with 2 or 3 dimensions, got shape {array.shape}")

        if np.issubdtype(array.dtype, np.floating):
            if array.max() <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        elif array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)

        image = Image.fromarray(array)

    if mode_hint == "image":
        return image.convert("RGB")
    return image.convert("L")


def normalize_mask(mask, invert_mask=False):
    mask = mask.convert("L")
    data = mask.point(lambda value: 255 if value > 0 else 0)
    if invert_mask:
        data = data.point(lambda value: 0 if value > 0 else 255)
    return data


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"
    masks_dir = args.output_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(args.dataset, args.subset, split=args.split)
    if len(dataset) == 0:
        raise ValueError("Loaded dataset split is empty.")

    first = dataset[0]
    image_column = choose_column(
        first,
        args.image_column,
        ["image", "images", "img", "tile", "pixel_values", "pixel_value"],
        "image",
    )
    mask_column = choose_column(
        first,
        args.mask_column,
        ["mask", "masks", "label", "labels", "annotation", "annotations"],
        "mask",
    )

    metadata_path = args.output_dir / "metadata.jsonl"
    count = 0

    with metadata_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            image = row[image_column]
            mask = row[mask_column]

            image = as_pil_image(image, mode_hint="image")
            mask = normalize_mask(as_pil_image(mask, mode_hint="mask"), invert_mask=args.invert_mask)

            stem = f"{index:06d}"
            image_name = f"{stem}.png"
            mask_name = f"{stem}_mask.png"

            image.save(images_dir / image_name)
            mask.save(masks_dir / mask_name)

            row_meta = {
                "image": f"images/{image_name}",
                "mask": f"masks/{mask_name}",
                "prompt": args.prompt,
                "source_dataset": args.dataset,
                "source_split": args.split,
                "source_index": index,
            }
            handle.write(json.dumps(row_meta) + "\n")
            count += 1

            if args.limit > 0 and count >= args.limit:
                break

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "subset": args.subset,
        "image_column": image_column,
        "mask_column": mask_column,
        "count": count,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
