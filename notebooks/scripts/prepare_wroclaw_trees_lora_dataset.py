import argparse
import json
import math
from pathlib import Path

from datasets import load_dataset
from PIL import Image, ImageDraw


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert the Wroclaw trees dataset into image/mask pairs for LoRA training."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="Filipstrozik/satellite_trees_wroclaw_2022",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--subset", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-column", type=str, default="images")
    parser.add_argument("--metadata-column", type=str, default="metadata")
    parser.add_argument("--prompt", type=str, default="satellite view of trees, canopy cover, urban vegetation")
    parser.add_argument("--radius-scale", type=float, default=1.0)
    parser.add_argument("--min-radius", type=float, default=2.0)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def draw_tree_mask(image_size, transformed_trees, radius_scale, min_radius):
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)

    for tree in transformed_trees:
        x = float(tree["x"])
        y = float(tree["y"])
        radius = max(min_radius, float(tree.get("radius", min_radius)) * radius_scale)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    return mask


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

    metadata_path = args.output_dir / "metadata.jsonl"
    count = 0

    with metadata_path.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(dataset):
            image = row[args.image_column]
            metadata = row[args.metadata_column]

            if not isinstance(image, Image.Image):
                raise TypeError(f"Expected image column '{args.image_column}' to yield PIL images, got {type(image)}")
            if not isinstance(metadata, dict):
                raise TypeError(f"Expected metadata column '{args.metadata_column}' to yield dict objects, got {type(metadata)}")

            transformed_trees = metadata.get("transformed_trees", [])
            if not transformed_trees:
                continue

            image = image.convert("RGB")
            mask = draw_tree_mask(image.size, transformed_trees, args.radius_scale, args.min_radius)

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
                "tree_count": len(transformed_trees),
            }
            handle.write(json.dumps(row_meta) + "\n")
            count += 1

            if args.limit > 0 and count >= args.limit:
                break

    summary = {
        "dataset": args.dataset,
        "split": args.split,
        "subset": args.subset,
        "count": count,
        "output_dir": str(args.output_dir),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
