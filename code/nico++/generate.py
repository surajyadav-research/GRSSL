import os
import csv
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFilter

from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    SamProcessor,
    SamModel,
)
from diffusers import FluxFillPipeline
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
REPO_ROOT = Path(__file__).resolve().parents[2]

# =========================================================
# 1. CONFIG UPDATED FOR FOX/WOLF & 4 CONTEXTS
# =========================================================

DATASET_ROOT = REPO_ROOT / "data" / "nico++" / "nico_original"
OUTPUT_ROOT = REPO_ROOT / "data" / "nico++" / "generated"

SPLITS = ["train"]   # change to ["train", "test"] if you want to run test too

ALL_PROMPTS = [
    # 1. Grass
    "A dense grassy field, endless bright green blades of grass, vibrant lush meadow vegetation, 4k",

    # 2. Outdoor
    "An empty asphalt highway, cracked gray pavement, urban road landscape with painted lane lines, highly detailed photography, 4k",

    # 3. Rock
    "A solid rock formation, massive smooth gray boulders and harsh stone surfaces, earthy tones, 4k",

    # 4. Water
    "Clear water filling a deep water pond, gentle water ripples moving across the water surface. Highly detailed water photography, 4k."
]

TEXT_QUERIES = {
    "fox": [
        "fox . wild animal . canine . predator .",
        "fox . animal . creature . wild dog .",
    ],
    "wolf": [
        "wolf . wild animal . canine . predator .",
        "wolf . animal . creature . wild dog .",
    ],
}

BOX_THRESHOLD = 0.45
FALLBACK_BOX_THRESHOLD = 0.40

NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 30.0
SAVE_DEBUG_EVERY = 30
SAVE_EXT = ".png"

FIXED_IMAGE_SIZE = (1024, 1024)

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32


# =========================================================
# 2. OUTPUT DIRS
# =========================================================

for split in SPLITS:
    (OUTPUT_ROOT / split).mkdir(parents=True, exist_ok=True)

    debug_base = OUTPUT_ROOT / "debug" / split
    (debug_base / "detections").mkdir(parents=True, exist_ok=True)
    (debug_base / "masks").mkdir(parents=True, exist_ok=True)
    (debug_base / "generated").mkdir(parents=True, exist_ok=True)
    (debug_base / "failures").mkdir(parents=True, exist_ok=True)


# =========================================================
# 3. LOAD MODELS
# =========================================================

print("Loading Grounding DINO...")
gdino_processor = AutoProcessor.from_pretrained(
    "IDEA-Research/grounding-dino-base",
    token=HF_TOKEN if HF_TOKEN else None,
)

gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-base",
    token=HF_TOKEN if HF_TOKEN else None,
).to(device)

print("Loading SAM...")
sam_processor = SamProcessor.from_pretrained(
    "facebook/sam-vit-base",
    token=HF_TOKEN if HF_TOKEN else None,
)

sam_model = SamModel.from_pretrained(
    "facebook/sam-vit-base",
    token=HF_TOKEN if HF_TOKEN else None,
).to(device)

print("Loading FLUX Fill...")
flux_pipe = FluxFillPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Fill-dev",
    torch_dtype=pipe_dtype,
    token=HF_TOKEN if HF_TOKEN else None,
).to(device)

if hasattr(flux_pipe, "enable_attention_slicing"):
    flux_pipe.enable_attention_slicing()


# =========================================================
# 4. HELPERS
# =========================================================

def get_image_paths(folder: Path):
    if not folder.exists():
        return []

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    image_paths = []

    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            image_paths.append(p)

    return sorted(image_paths)


def make_safe_output_name(split: str, context_name: str, class_name: str, img_path: Path):
    """
    Creates a unique output name.

    Example:
    train/grass/fox/0001.jpg  -> train_grass_fox_0001
    train/grass/wolf/0001.jpg -> train_grass_wolf_0001

    This prevents fox and wolf images with the same filename from colliding.
    """
    stem = img_path.stem

    safe_split = str(split).replace(" ", "_")
    safe_context = str(context_name).replace(" ", "_")
    safe_class = str(class_name).replace(" ", "_")
    safe_stem = str(stem).replace(" ", "_")

    return f"{safe_split}_{safe_context}_{safe_class}_{safe_stem}"


def build_records():
    records = []

    classes = ["fox", "wolf"]
    train_contexts = ["grass", "outdoor", "rock", "water"]

    # Train split: train -> context -> class
    for context in train_contexts:
        for cls_name in classes:
            folder = DATASET_ROOT / "train" / context / cls_name

            for img_path in get_image_paths(folder):
                records.append({
                    "split": "train",
                    "class_name": cls_name,
                    "context_name": context,
                    "img_path": img_path,
                })

    # Test split: test -> class
    for cls_name in classes:
        folder = DATASET_ROOT / "test" / cls_name

        for img_path in get_image_paths(folder):
            records.append({
                "split": "test",
                "class_name": cls_name,
                "context_name": "none",
                "img_path": img_path,
            })

    return records


def detect_object_box(image: Image.Image, class_name: str):
    img_w, img_h = image.size
    queries = TEXT_QUERIES[class_name]

    for text in queries:
        inputs = gdino_processor(
            images=image,
            text=text,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = gdino_model(**inputs)

        pred_boxes = outputs.pred_boxes[0]
        logits = outputs.logits[0]
        scores = torch.sigmoid(logits).max(dim=-1).values

        cx, cy, bw, bh = pred_boxes.unbind(-1)

        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h

        xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

        for threshold in (BOX_THRESHOLD, FALLBACK_BOX_THRESHOLD):
            keep = scores >= threshold

            if not keep.any():
                continue

            kept_scores = scores[keep]
            kept_boxes = xyxy[keep]

            best_idx = int(kept_scores.argmax().item())
            box = kept_boxes[best_idx].tolist()
            score = kept_scores[best_idx].item()

            return box, {
                "score": score,
                "label": class_name,
                "threshold_used": threshold,
                "query_used": text,
            }

    return None, None


def segment_with_sam(image: Image.Image, box_xyxy):
    input_boxes = [[box_xyxy]]

    inputs = sam_processor(
        image,
        input_boxes=input_boxes,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = sam_model(**inputs)

    masks = sam_processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )

    iou_scores = outputs.iou_scores[0, 0].detach().cpu()
    best_mask_idx = int(torch.argmax(iou_scores).item())

    mask_np = masks[0][0][best_mask_idx].numpy().astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask_np).convert("L")

    return mask_pil, float(iou_scores[best_mask_idx].item())


def save_sidecar_csv(path: Path, rows: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])

        for k, v in rows.items():
            writer.writerow([k, v])


def overlay_box(image: Image.Image, box_xyxy, text=None):
    img = image.copy()
    draw = ImageDraw.Draw(img)

    x1, y1, x2, y2 = box_xyxy
    draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

    if text:
        tx = max(5, int(x1))
        ty = max(5, int(y1) - 20)
        draw.text((tx, ty), text, fill="red")

    return img


def overlay_mask(image: Image.Image, mask_pil: Image.Image, alpha=120):
    img = image.convert("RGBA")
    mask = mask_pil.convert("L")

    overlay = Image.new("RGBA", img.size, (255, 0, 0, 0))
    overlay_np = np.array(overlay)
    mask_np = np.array(mask)

    overlay_np[mask_np > 0] = [255, 0, 0, alpha]
    overlay = Image.fromarray(overlay_np, mode="RGBA")

    composed = Image.alpha_composite(img, overlay)

    return composed.convert("RGB")


def save_failure_visual(split, output_base_name, image, reason, extra_text=None):
    fail_dir = OUTPUT_ROOT / "debug" / split / "failures"
    out_path = fail_dir / f"{output_base_name}_failure.jpg"

    img = image.copy()
    draw = ImageDraw.Draw(img)

    msg = reason if extra_text is None else f"{reason} | {extra_text}"
    draw.text((10, 10), msg, fill="red")

    img.save(out_path, quality=95)


def save_detection_failures_csv(csv_path: Path, failure_rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "output_base_name",
            "image_name",
            "file_name",
            "class_name",
            "context_name",
            "reason"
        ])

        writer.writerows(failure_rows)


# =========================================================
# 5. MAIN
# =========================================================

all_records = build_records()

if not all_records:
    print(f"No images found in the specified root folder: {DATASET_ROOT}")

else:
    split_to_records = {}

    for r in all_records:
        split_to_records.setdefault(r["split"], []).append(r)

    for split in SPLITS:
        records = split_to_records.get(split, [])

        if not records:
            print(f"Skipping {split}: no images found")
            continue

        out_dir = OUTPUT_ROOT / split

        debug_base = OUTPUT_ROOT / "debug" / split
        detection_debug_dir = debug_base / "detections"
        mask_debug_dir = debug_base / "masks"
        generated_debug_dir = debug_base / "generated"
        detection_failure_csv = debug_base / "detection_failures.csv"

        print(f"\nProcessing split: {split} | {len(records)} images")

        detection_failure_rows = []

        for idx, record in enumerate(records, start=1):
            img_path = record["img_path"]
            class_name = record["class_name"]
            context_name = record["context_name"]

            file_name = img_path.name
            original_img_stem = img_path.stem

            # IMPORTANT:
            # Unique name includes split, context, class, and original file stem.
            # This prevents fox/wolf/context filename collisions.
            img_name = make_safe_output_name(
                split=split,
                context_name=context_name,
                class_name=class_name,
                img_path=img_path
            )

            should_save_periodic_debug = (idx % SAVE_DEBUG_EVERY == 0)

            print(f"[{split}] {idx}/{len(records)} -> {img_path}")
            print(f"  -> output base name: {img_name}")

            try:
                original_image = Image.open(img_path).convert("RGB")

                image = original_image.resize(
                    FIXED_IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )

                # -----------------------------------------
                # A. Detect object pre-generation
                # -----------------------------------------
                obj_box, det_info = detect_object_box(image, class_name)

                if obj_box is None:
                    print(f"  -> No {class_name} detected | skipping image completely")

                    save_failure_visual(
                        split,
                        img_name,
                        image,
                        f"no_{class_name}_detected"
                    )

                    detection_failure_rows.append([
                        img_name,
                        original_img_stem,
                        file_name,
                        class_name,
                        context_name,
                        f"no_{class_name}_detected"
                    ])

                    continue

                det_score = det_info["score"] if det_info else None

                if det_score is not None:
                    det_text = f"{class_name} | score={det_score:.3f}"
                else:
                    det_text = class_name

                det_vis = overlay_box(image, obj_box, det_text)

                # Debug detection image also uses unique name.
                det_vis.save(
                    detection_debug_dir / f"{img_name}_det.jpg",
                    quality=95
                )

                # -----------------------------------------
                # B. Segment object with SAM
                # -----------------------------------------
                obj_mask, sam_iou = segment_with_sam(image, obj_box)
                mask_np = np.array(obj_mask)

                if mask_np.sum() == 0:
                    print("  -> Empty SAM mask | skipping image")

                    save_failure_visual(
                        split,
                        img_name,
                        image,
                        "empty_sam_mask",
                        f"sam_iou={sam_iou:.4f}"
                    )

                    detection_failure_rows.append([
                        img_name,
                        original_img_stem,
                        file_name,
                        class_name,
                        context_name,
                        "empty_sam_mask"
                    ])

                    continue

                box_area = (obj_box[2] - obj_box[0]) * (obj_box[3] - obj_box[1])
                mask_area = np.count_nonzero(mask_np)

                fill_ratio = mask_area / box_area if box_area > 0 else 0

                if fill_ratio < 0.35:
                    print(f"  -> Fragmented SAM mask | fill ratio {fill_ratio:.2f} | skipping")

                    save_failure_visual(
                        split,
                        img_name,
                        image,
                        "fragmented_sam_mask",
                        f"fill_ratio={fill_ratio:.2f}"
                    )

                    detection_failure_rows.append([
                        img_name,
                        original_img_stem,
                        file_name,
                        class_name,
                        context_name,
                        "fragmented_mask"
                    ])

                    continue

                # -----------------------------------------
                # C. Save real/original image only if detected and validated
                # -----------------------------------------
                real_img_path = out_dir / f"{img_name}_orig{SAVE_EXT}"
                real_csv_path = out_dir / f"{img_name}_orig.csv"

                if not real_img_path.exists():
                    image.save(real_img_path)
                else:
                    print(f"  -> original already exists, skipping: {real_img_path.name}")

                if not real_csv_path.exists():
                    real_sidecar = {
                        "source_file": file_name,
                        "source_path": str(img_path),
                        "original_image_stem": original_img_stem,
                        "output_base_name": img_name,
                        "split": split,
                        "class_name": class_name,
                        "source_context": context_name,
                        "type": "original_image",
                        "original_size": str(original_image.size),
                        "saved_size": str(image.size),
                        "output_image": real_img_path.name,
                    }

                    save_sidecar_csv(real_csv_path, real_sidecar)
                else:
                    print(f"  -> original csv already exists, skipping: {real_csv_path.name}")

                # Debug masks also use unique name.
                obj_mask.save(mask_debug_dir / f"{img_name}_mask_only.png")

                mask_vis = overlay_mask(image, obj_mask)
                mask_vis.save(
                    mask_debug_dir / f"{img_name}_mask_overlay.jpg",
                    quality=95
                )

                # -----------------------------------------
                # D. Advanced mask preparation
                # -----------------------------------------
                expanded_mask = obj_mask.filter(ImageFilter.MaxFilter(15))
                flux_mask = ImageOps.invert(expanded_mask)

                shrunk_mask = obj_mask.filter(ImageFilter.MinFilter(5))
                paste_mask = shrunk_mask.filter(ImageFilter.GaussianBlur(radius=5.0))

                # -----------------------------------------
                # E. Generate all 4 prompt variants
                # -----------------------------------------
                completed_count = 0

                for prompt_idx, prompt in enumerate(ALL_PROMPTS):
                    out_img_path = out_dir / f"{img_name}_var{prompt_idx}{SAVE_EXT}"
                    info_path = out_dir / f"{img_name}_var{prompt_idx}.csv"

                    if out_img_path.exists() and info_path.exists():
                        print(f"  -> output exists, skipping: {out_img_path.name}")
                        completed_count += 1
                        continue

                    with torch.no_grad():
                        result = flux_pipe(
                            prompt=prompt,
                            image=image,
                            mask_image=flux_mask,
                            num_inference_steps=NUM_INFERENCE_STEPS,
                            guidance_scale=GUIDANCE_SCALE,
                            height=1024,
                            width=1024,
                        )

                    generated_image = result.images[0]
                    generated_image.paste(image, (0, 0), paste_mask)

                    if generated_image.size != FIXED_IMAGE_SIZE:
                        generated_image = generated_image.resize(
                            FIXED_IMAGE_SIZE,
                            Image.Resampling.LANCZOS
                        )

                    # Post-generation quality control
                    post_box, post_info = detect_object_box(generated_image, class_name)

                    if post_box is None:
                        print(f"  -> Post-gen detection failed for var{prompt_idx} | variant rejected")

                        save_failure_visual(
                            split,
                            f"{img_name}_var{prompt_idx}",
                            generated_image,
                            "post_gen_detection_failed"
                        )

                        detection_failure_rows.append([
                            f"{img_name}_var{prompt_idx}",
                            original_img_stem,
                            file_name,
                            class_name,
                            context_name,
                            "post_gen_detection_failed"
                        ])

                        continue

                    generated_image.save(out_img_path)

                    sidecar = {
                        "source_file": file_name,
                        "source_path": str(img_path),
                        "original_image_stem": original_img_stem,
                        "output_base_name": img_name,
                        "split": split,
                        "class_name": class_name,
                        "source_context": context_name,
                        "type": "generated_image",
                        "variation_index": prompt_idx,
                        "prompt": prompt,
                        "text_query": det_info["query_used"] if det_info else "",
                        "box_x1": obj_box[0],
                        "box_y1": obj_box[1],
                        "box_x2": obj_box[2],
                        "box_y2": obj_box[3],
                        "pre_det_score": det_info["score"] if det_info else "",
                        "post_det_score": post_info["score"] if post_info else "",
                        "sam_iou": sam_iou,
                        "original_size": str(original_image.size),
                        "saved_size": str(generated_image.size),
                        "output_image": out_img_path.name,
                    }

                    save_sidecar_csv(info_path, sidecar)

                    if should_save_periodic_debug:
                        generated_image.save(
                            generated_debug_dir / f"{img_name}_var{prompt_idx}_generated{SAVE_EXT}"
                        )

                    completed_count += 1

                print(f"  -> completed {completed_count} valid variants + 1 real image")

            except Exception as e:
                print(f"  -> failed: {e}")

                try:
                    image = Image.open(img_path).convert("RGB").resize(
                        FIXED_IMAGE_SIZE,
                        Image.Resampling.LANCZOS
                    )

                    save_failure_visual(
                        split,
                        img_name,
                        image,
                        "exception",
                        str(e)
                    )

                    detection_failure_rows.append([
                        img_name,
                        original_img_stem,
                        file_name,
                        class_name,
                        context_name,
                        f"exception: {e}"
                    ])

                except Exception:
                    detection_failure_rows.append([
                        img_name,
                        original_img_stem,
                        file_name,
                        class_name,
                        context_name,
                        f"exception: {e}"
                    ])

        save_detection_failures_csv(detection_failure_csv, detection_failure_rows)

        print(f"\nSaved detection failure CSV to: {detection_failure_csv}")

print("\nDone.")