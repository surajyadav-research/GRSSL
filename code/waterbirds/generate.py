import os
import csv
import hashlib
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
# 1. CONFIG
# =========================================================
DATASET_ROOT = REPO_ROOT / "data" / "waterbirds" / "waterbirds_original"
OUTPUT_ROOT = REPO_ROOT / "data" / "waterbirds" / "generated"

# Process splits you want
SPLITS = ["train"]  # e.g. ["train", "val", "test"]

# Waterbirds prompts
PROMPTS = [
    # --- Water backgrounds ---
    "A highly detailed, photorealistic wide shot of a calm blue ocean with gentle rolling waves, bright sunny sky, natural lighting, 4k",
    "A serene, misty freshwater lake at sunrise, calm water reflecting the golden hour light, tranquil nature photography",

    # --- Land backgrounds ---
    "A rugged mountain landscape with rocky peaks, sparse alpine vegetation, and a clear blue sky, crisp daylight, high resolution",
    "A dense, lush green forest floor with dappled sunlight filtering through the trees, mossy environment, cinematic lighting, highly detailed",
]

# Waterbirds place labels: 1 = water, 0 = land. Generated variants inherit
# the source class label but take the environment label from the prompt.
PROMPT_PLACES = [1, 1, 0, 0]

# Grounding DINO bird queries
TEXT_QUERIES = [
    "bird . waterbird . seabird . duck . goose . gull . heron . cormorant . pelican . egret . tern . albatross . crane . flamingo . swan . hawk . eagle . owl . parrot . robin .",
    "bird . animal . wildlife . creature .",
]

# Detection thresholds
BOX_THRESHOLD = 0.25
FALLBACK_BOX_THRESHOLD = 0.10

# Generation hyperparameters
NUM_INFERENCE_STEPS = 20
GUIDANCE_SCALE = 30.0
SAVE_DEBUG_EVERY = 30
SAVE_EXT = ".png"

# Generate at fixed size
FIXED_IMAGE_SIZE = (1024, 1024)

# Mask quality rules
MIN_MASK_FILL_RATIO = 0.10
MAX_MASK_FILL_RATIO = 1.50

# Debug / caching
OVERWRITE_EXISTING = False

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
def get_image_paths(split_dir: Path):
    """
    Waterbirds layout assumed:
    DATASET_ROOT / split / images / *.jpg|png|...
    """
    image_dir = split_dir / "images"
    if not image_dir.exists():
        return []

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in exts])


def load_metadata_map(split_dir: Path):
    """
    Reads metadata.csv if present.
    Expected to contain at least file_name and possibly y, place, split, etc.
    """
    metadata_path = split_dir / "metadata.csv"
    metadata_map = {}

    if not metadata_path.exists():
        return metadata_map

    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_name = row.get("file_name")
            if file_name:
                metadata_map[file_name] = row

    return metadata_map


def stable_seed_from_name(name: str, prompt_idx: int):
    """
    Deterministic seed per image + prompt index.
    """
    text = f"{name}|{prompt_idx}"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def save_sidecar_csv(path: Path, rows: dict):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in rows.items():
            writer.writerow([k, v])


def save_failure_visual(split, base_name, image, reason, extra_text=None):
    fail_dir = OUTPUT_ROOT / "debug" / split / "failures"
    out_path = fail_dir / f"{base_name}_failure.jpg"

    img = image.copy()
    draw = ImageDraw.Draw(img)
    msg = reason if extra_text is None else f"{reason} | {extra_text}"
    draw.text((10, 10), msg, fill="red")
    img.save(out_path, quality=95)


def save_detection_failures_csv(csv_path: Path, failure_rows):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_name", "file_name", "reason"])
        writer.writerows(failure_rows)


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


def detect_bird_box(image: Image.Image):
    """
    Try each TEXT_QUERY at BOX_THRESHOLD then FALLBACK_BOX_THRESHOLD.
    Returns (box_xyxy, det_info) or (None, None) if nothing found.
    """
    img_w, img_h = image.size

    for text in TEXT_QUERIES:
        inputs = gdino_processor(
            images=image,
            text=text,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = gdino_model(**inputs)

        pred_boxes = outputs.pred_boxes[0]  # cxcywh normalized
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

            # clamp to image bounds
            box[0] = max(0.0, min(box[0], img_w - 1))
            box[1] = max(0.0, min(box[1], img_h - 1))
            box[2] = max(0.0, min(box[2], img_w - 1))
            box[3] = max(0.0, min(box[3], img_h - 1))

            if box[2] <= box[0] or box[3] <= box[1]:
                continue

            return box, {
                "score": score,
                "label": "bird",
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


def prepare_flux_masks(obj_mask: Image.Image):
    """
    1) expanded mask -> invert for background editing in FLUX
    2) shrunk+blurred mask -> used to paste original bird back for clean preservation
    """
    expanded_mask = obj_mask.filter(ImageFilter.MaxFilter(15))
    flux_mask = ImageOps.invert(expanded_mask)  # white background to edit, black bird to keep

    shrunk_mask = obj_mask.filter(ImageFilter.MinFilter(5))
    paste_mask = shrunk_mask.filter(ImageFilter.GaussianBlur(radius=5.0))

    return flux_mask, paste_mask


# =========================================================
# 5. MAIN
# =========================================================
for split in SPLITS:
    split_dir = DATASET_ROOT / split
    image_paths = get_image_paths(split_dir)
    metadata_map = load_metadata_map(split_dir)

    if not image_paths:
        print(f"Skipping {split}: no images found")
        continue

    out_dir = OUTPUT_ROOT / split
    debug_base = OUTPUT_ROOT / "debug" / split
    detection_debug_dir = debug_base / "detections"
    mask_debug_dir = debug_base / "masks"
    generated_debug_dir = debug_base / "generated"
    detection_failure_csv = debug_base / "detection_failures.csv"

    print(f"\nProcessing split: {split} | {len(image_paths)} images")
    detection_failure_rows = []

    for idx, img_path in enumerate(image_paths, start=1):
        file_name = img_path.name
        img_name = img_path.stem
        meta = metadata_map.get(file_name, {})
        should_save_periodic_debug = (idx % SAVE_DEBUG_EVERY == 0)

        print(f"[{split}] {idx}/{len(image_paths)} -> {file_name}")

        try:
            original_image = Image.open(img_path).convert("RGB")
            image = original_image.resize(FIXED_IMAGE_SIZE, Image.Resampling.LANCZOS)

            # -----------------------------------------
            # A. Detect bird
            # -----------------------------------------
            bird_box, det_info = detect_bird_box(image)

            if bird_box is None:
                print("  -> No bird detected | skipping image completely")
                save_failure_visual(split, img_name, image, "no_bird_detected")
                detection_failure_rows.append([img_name, file_name, "no_bird_detected"])
                continue

            det_score = det_info["score"] if det_info else None
            det_text = f"bird | score={det_score:.3f}" if det_score is not None else "bird"
            det_vis = overlay_box(image, bird_box, det_text)
            det_vis.save(detection_debug_dir / f"{img_name}_det.jpg", quality=95)

            # -----------------------------------------
            # B. Segment bird with SAM
            # -----------------------------------------
            bird_mask, sam_iou = segment_with_sam(image, bird_box)
            mask_np = np.array(bird_mask)

            if mask_np.sum() == 0:
                print("  -> Empty SAM mask | skipping image")
                save_failure_visual(split, img_name, image, "empty_sam_mask", f"sam_iou={sam_iou:.4f}")
                detection_failure_rows.append([img_name, file_name, "empty_sam_mask"])
                continue

            box_area = max((bird_box[2] - bird_box[0]) * (bird_box[3] - bird_box[1]), 1.0)
            mask_area = float(np.count_nonzero(mask_np))
            fill_ratio = mask_area / box_area

            if fill_ratio < MIN_MASK_FILL_RATIO:
                print(f"  -> Fragmented SAM mask (fill ratio {fill_ratio:.3f}) | skipping")
                save_failure_visual(split, img_name, image, "fragmented_sam_mask", f"fill_ratio={fill_ratio:.3f}")
                detection_failure_rows.append([img_name, file_name, "fragmented_sam_mask"])
                continue

            if fill_ratio > MAX_MASK_FILL_RATIO:
                print(f"  -> Overgrown SAM mask (fill ratio {fill_ratio:.3f}) | skipping")
                save_failure_visual(split, img_name, image, "overgrown_sam_mask", f"fill_ratio={fill_ratio:.3f}")
                detection_failure_rows.append([img_name, file_name, "overgrown_sam_mask"])
                continue

            bird_mask.save(mask_debug_dir / f"{img_name}_mask_only.png")
            mask_vis = overlay_mask(image, bird_mask)
            mask_vis.save(mask_debug_dir / f"{img_name}_mask_overlay.jpg", quality=95)

            # -----------------------------------------
            # C. Save original resized image
            # -----------------------------------------
            real_img_path = out_dir / f"{img_name}_orig{SAVE_EXT}"
            real_csv_path = out_dir / f"{img_name}_orig.csv"

            if OVERWRITE_EXISTING or not real_img_path.exists():
                image.save(real_img_path)

            if OVERWRITE_EXISTING or not real_csv_path.exists():
                real_sidecar = {
                    "source_file": file_name,
                    "source_path": str(img_path),
                    "split": split,
                    "type": "original_image",
                    "original_size": str(original_image.size),
                    "saved_size": str(image.size),
                    "output_image": real_img_path.name,
                }
                for k, v in meta.items():
                    real_sidecar[f"meta_{k}"] = v
                save_sidecar_csv(real_csv_path, real_sidecar)

            # -----------------------------------------
            # D. Prepare masks
            # -----------------------------------------
            flux_mask, paste_mask = prepare_flux_masks(bird_mask)

            # -----------------------------------------
            # E. Generate prompt variants
            # -----------------------------------------
            completed_count = 0

            for prompt_idx, prompt in enumerate(PROMPTS):
                out_img_path = out_dir / f"{img_name}_var{prompt_idx}{SAVE_EXT}"
                info_path = out_dir / f"{img_name}_var{prompt_idx}.csv"

                if (
                    not OVERWRITE_EXISTING
                    and out_img_path.exists()
                    and info_path.exists()
                ):
                    print(f"  -> output exists, skipping: {out_img_path.name}")
                    completed_count += 1
                    continue

                seed = stable_seed_from_name(img_name, prompt_idx)
                generator = torch.Generator(device=device).manual_seed(seed)

                with torch.no_grad():
                    result = flux_pipe(
                        prompt=prompt,
                        image=image,
                        mask_image=flux_mask,
                        num_inference_steps=NUM_INFERENCE_STEPS,
                        guidance_scale=GUIDANCE_SCALE,
                        height=FIXED_IMAGE_SIZE[1],
                        width=FIXED_IMAGE_SIZE[0],
                        generator=generator,
                    )

                generated_image = result.images[0]

                # Paste original bird back for cleaner subject preservation
                generated_image.paste(image, (0, 0), paste_mask)

                if generated_image.size != FIXED_IMAGE_SIZE:
                    generated_image = generated_image.resize(
                        FIXED_IMAGE_SIZE,
                        Image.Resampling.LANCZOS
                    )

                # -----------------------------------------
                # F. Post-generation validation
                # -----------------------------------------
                post_box, post_info = detect_bird_box(generated_image)

                if post_box is None:
                    print(f"  -> Post-gen detection failed for var{prompt_idx} | variant rejected")
                    save_failure_visual(
                        split,
                        f"{img_name}_var{prompt_idx}",
                        generated_image,
                        "post_gen_detection_failed"
                    )
                    detection_failure_rows.append([f"{img_name}_var{prompt_idx}", file_name, "post_gen_detection_failed"])
                    continue

                # -----------------------------------------
                # G. Save generated image only
                # -----------------------------------------
                generated_image.save(out_img_path)

                sidecar = {
                    "source_file": file_name,
                    "source_path": str(img_path),
                    "split": split,
                    "type": "generated_image",
                    "variation_index": prompt_idx,
                    "prompt": prompt,
                    "seed": seed,
                    "text_query": det_info["query_used"] if det_info else "",
                    "box_x1": bird_box[0],
                    "box_y1": bird_box[1],
                    "box_x2": bird_box[2],
                    "box_y2": bird_box[3],
                    "pre_det_score": det_info["score"] if det_info else "",
                    "post_det_score": post_info["score"] if post_info else "",
                    "sam_iou": sam_iou,
                    "mask_fill_ratio": fill_ratio,
                    "original_size": str(original_image.size),
                    "saved_size": str(generated_image.size),
                    "output_image": out_img_path.name,
                }
                for k, v in meta.items():
                    sidecar[f"meta_{k}"] = v
                sidecar["meta_place"] = PROMPT_PLACES[prompt_idx]
                save_sidecar_csv(info_path, sidecar)

                if should_save_periodic_debug:
                    generated_image.save(
                        generated_debug_dir / f"{img_name}_var{prompt_idx}_generated{SAVE_EXT}"
                    )

                completed_count += 1

            print(f"  -> completed {completed_count} valid variants + 1 real image")

        except Exception as e:
            print(f"  -> failed: {e}")
            detection_failure_rows.append([img_name, file_name, f"exception: {str(e)}"])
            try:
                fallback_image = Image.open(img_path).convert("RGB").resize(
                    FIXED_IMAGE_SIZE,
                    Image.Resampling.LANCZOS
                )
                save_failure_visual(split, img_name, fallback_image, "exception", str(e))
            except Exception:
                pass

    save_detection_failures_csv(detection_failure_csv, detection_failure_rows)
    print(f"\nSaved detection failure CSV to: {detection_failure_csv}")

print("\nDone.")