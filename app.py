# Web application for Leaf micro-CT Segmentation
# Author: Worasit Sanjgan
# Date: 7 April 2026

import sys

# Suppress Python 3.13 asyncio cleanup noise (Invalid file descriptor: -1)
def _suppress_invalid_fd(unraisable):
    if isinstance(unraisable.exc_value, ValueError) and "Invalid file descriptor" in str(unraisable.exc_value):
        return
    sys.__unraisablehook__(unraisable)
sys.unraisablehook = _suppress_invalid_fd

import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModel, AutoConfig
import os
import csv
import tempfile

# ── CONFIG ────────────────────────────────────────────────────────────────────
# For HuggingFace ZeroGPU deployment, uncomment:
# import spaces

MODEL_REPO  = "WorasitSangjan/Leaf-CT-Segmentation-Model"
DEVICE      = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else "cpu"
)
PATCH_SIZE  = 320
STRIDE      = 80
NUM_CLASSES = 5

CLASS_NAMES  = ["Background", "Epidermis", "Vascular_Region", "Mesophyll", "Air_Space"]
CLASS_COLORS = np.array([
    [0,   0,   0  ],   # Background — black
    [255, 100, 100],   # Epidermis  — red
    [100, 200, 100],   # Vascular   — green
    [100, 100, 255],   # Mesophyll  — blue
    [255, 230, 50 ],   # Air Space  — yellow
], dtype=np.uint8)

# 7-class scheme (new model): Bundle_Sheath and Vascular_Tissue are separate classes and the
# mesophyll is split into Palisade and Spongy mesophyll.
CLASS_NAMES_7  = ["Background", "Epidermis", "Bundle_Sheath", "Vascular_Tissue",
                  "Palisade_Mesophyll", "Spongy_Mesophyll", "Air_Space"]
CLASS_COLORS_7 = np.array([
    [0,   0,   0  ],   # Background         — black
    [255, 100, 100],   # Epidermis          — red
    [255, 165, 0  ],   # Bundle sheath      — orange
    [160, 70,  200],   # Vascular tissue    — purple
    [80,  180, 80 ],   # Palisade mesophyll — green
    [100, 100, 255],   # Spongy mesophyll   — blue
    [255, 230, 50 ],   # Air space          — yellow
], dtype=np.uint8)

EXAMPLE_PATHS = [
    ("examples/Lantana.png",  "Lantana"),
    ("examples/Olive.png",    "Olive"),
    ("examples/Pine.png",     "Pine"),
    ("examples/Viburnum.png", "Viburnum"),
    ("examples/Wheat.png",    "Wheat"),
]

STACK_EXAMPLE_PATHS = [
    ("examples/stacks/Arabidopsis.tif", "examples/stacks/Arabidopsis_preview.png", "Arabidopsis"),
    ("examples/stacks/Grape.tif",       "examples/stacks/Grape_preview.png",       "Grape"),
    ("examples/stacks/Oak.tif",         "examples/stacks/Oak_preview.png",         "Oak"),
]


# ── MODEL ARCHITECTURES ───────────────────────────────────────────────────────
# Every model exposes the same interface: forward (B, 1, H, W) → (B, NUM_CLASSES, H, W)
# semantic logits, so the patch-based inference loop can stay model-agnostic.

class EoMT_ViTL(nn.Module):
    """Encoder-only Mask Transformer with DINOv3 ViT-L/16 backbone."""
    def __init__(self, num_classes=5, num_queries=100):
        super().__init__()
        self.conv_in  = nn.Conv2d(1, 3, kernel_size=1)
        # Initialize architecture directly from local config to bypass gated HuggingFace download
        try:
            config_path = os.path.join(os.path.dirname(__file__), "config.json")
            config = AutoConfig.from_pretrained(config_path)
            self.backbone = AutoModel.from_config(config)
        except Exception:
            self.backbone = AutoModel.from_pretrained(
                "facebook/dinov3-vitl16-pretrain-lvd1689m", local_files_only=False
            )
        embed_dim        = self.backbone.config.hidden_size   # 1024
        self._embed_dim  = embed_dim
        self._patch_size = self.backbone.config.patch_size    # 16

        self.q          = nn.Embedding(num_queries, embed_dim)
        self.query_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=16, dropout=0.0, batch_first=True
        )
        self.class_head = nn.Linear(embed_dim, num_classes + 1)
        self.mask_head  = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, embed_dim // 2, kernel_size=2, stride=2),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim // 2, embed_dim, kernel_size=2, stride=2),
        )

    def forward(self, images):
        B, C, H, W = images.shape
        if C == 1:
            images = self.conv_in(images)
        hidden  = self.backbone(images).last_hidden_state
        spatial = hidden[:, 1:, :]   # skip CLS token

        q_tokens     = self.q.weight[None].expand(B, -1, -1)
        q_out, _     = self.query_attn(q_tokens, spatial, spatial)
        class_logits = self.class_head(q_out)

        grid_h, grid_w = H // self._patch_size, W // self._patch_size
        expected       = grid_h * grid_w
        if spatial.shape[1] > expected:
            spatial = spatial[:, :expected, :]
        spatial_up = self.upscale(
            spatial.transpose(1, 2).reshape(B, self._embed_dim, grid_h, grid_w)
        )
        mask_probs      = torch.einsum("bqd,bdhw->bqhw", self.mask_head(q_out), spatial_up).sigmoid()
        semantic_logits = torch.einsum("bqc,bqhw->bchw", class_logits[..., :-1], mask_probs)
        return F.interpolate(semantic_logits, size=(H, W), mode="bilinear", align_corners=False)


class SegFormerB4(nn.Module):
    """SegFormer-B4 (HuggingFace) with a 1→3 channel adapter; logits upsampled to input size."""
    def __init__(self, num_classes=5):
        super().__init__()
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        self.conv_in = nn.Conv2d(1, 3, kernel_size=1)
        config = SegformerConfig(
            num_labels=num_classes,
            depths=[3, 8, 27, 3],
            hidden_sizes=[64, 128, 320, 512],
            decoder_hidden_size=768,
        )
        self.model = SegformerForSemanticSegmentation(config)

    def forward(self, images):
        H, W   = images.shape[-2:]
        logits = self.model(pixel_values=self.conv_in(images)).logits   # (B, C, H/4, W/4)
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


class Mask2FormerSwinB(nn.Module):
    """Mask2Former (HuggingFace) with Swin-Base backbone; mask queries reduced to a semantic map."""
    def __init__(self, num_classes=5, num_queries=100):
        super().__init__()
        from transformers import Mask2FormerConfig, SwinConfig, Mask2FormerForUniversalSegmentation
        backbone = SwinConfig(
            embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32],
            window_size=12, image_size=384,
            out_features=["stage1", "stage2", "stage3", "stage4"],
        )
        config = Mask2FormerConfig(backbone_config=backbone, num_labels=num_classes, num_queries=num_queries)
        self.model = Mask2FormerForUniversalSegmentation(config)

    def forward(self, images):
        H, W = images.shape[-2:]
        out  = self.model(pixel_values=images.repeat(1, 3, 1, 1))
        cls  = out.class_queries_logits.softmax(dim=-1)[..., :-1]        # (B, Q, C), drop no-object
        mask = out.masks_queries_logits.sigmoid()                       # (B, Q, h, w)
        seg  = torch.einsum("bqc,bqhw->bchw", cls, mask)
        return F.interpolate(seg, size=(H, W), mode="bilinear", align_corners=False)


def _build_fpn(num_classes):
    import segmentation_models_pytorch as smp
    # Grayscale CT scans feed the MiT-B4 encoder directly (in_channels=1).
    return smp.FPN(encoder_name="mit_b4", encoder_weights=None, in_channels=1, classes=num_classes)


# ── MODEL REGISTRY ────────────────────────────────────────────────────────────
# The app has two segmentation schemes (5 classes and 7 classes); each has its own classes,
# colours and models. Display name → (HuggingFace Hub filename, builder). Upload the .pth files
# to MODEL_REPO with these exact filenames. Weights download lazily on first use.
MODELS = {
    "Mask2Former (Swin-B)": ("Mask2Former_SwinB.pth", lambda n: Mask2FormerSwinB(num_classes=n)),
    "SegFormer-B4":         ("Segformer_B4.pth",      lambda n: SegFormerB4(num_classes=n)),
    "EoMT (DINOv3 ViT-L)":  ("EoMT_DinoV3.pth",       lambda n: EoMT_ViTL(num_classes=n)),
    "FPN (MiT-B4)":         ("FPN_MiTB4.pth",         _build_fpn),
}
DEFAULT_MODEL = "Mask2Former (Swin-B)"

# Display labels shown in the radio buttons → internal model key
MODEL_LABELS = {
    "Mask2Former (Swin-B)": "Mask2Former (Swin-B)  —  Best overall · Mesophyll · Air Space ★",
    "SegFormer-B4":         "SegFormer-B4  —  Best for Vascular tissue",
    "EoMT (DINOv3 ViT-L)":  "EoMT (DINOv3 ViT-L)  —  Best for Epidermis",
    "FPN (MiT-B4)":         "FPN (MiT-B4)  —  Lightweight, comparable accuracy",
}
LABEL_TO_MODEL = {v: k for k, v in MODEL_LABELS.items()}

# 7-class model: Mask2Former (Swin-B), the best of the four architectures tested.
MODELS_7 = {
    "Mask2Former (Swin-B)": ("Mask2Former_SwinB_7class.pth", lambda n: Mask2FormerSwinB(num_classes=n)),
}
DEFAULT_MODEL_7 = "Mask2Former (Swin-B)"
MODEL_LABELS_7 = {
    "Mask2Former (Swin-B)": "Mask2Former (Swin-B)  —  7 classes ★",
}
LABEL_TO_MODEL_7 = {v: k for k, v in MODEL_LABELS_7.items()}

SCHEMES = {
    "5": dict(
        tab="5 Classes", num_classes=NUM_CLASSES, class_names=CLASS_NAMES, class_colors=CLASS_COLORS,
        models=MODELS, default=DEFAULT_MODEL, labels=MODEL_LABELS, label_to_model=LABEL_TO_MODEL,
        label_scale=50,        # class index × scale gives the grey-level label image
        norm="global",         # intensity normalisation: whole image
        check_keys=False,      # tolerate missing weights (kept as in the original release)
        combined=[], suffix="",
        blurb=("<p><strong>5 classes:</strong> Background, Epidermis, Vascular Region, Mesophyll, Air Space. "
               "Choose one of four models below.</p>"),
    ),
    "7": dict(
        tab="7 Classes", num_classes=len(CLASS_NAMES_7), class_names=CLASS_NAMES_7, class_colors=CLASS_COLORS_7,
        models=MODELS_7, default=DEFAULT_MODEL_7, labels=MODEL_LABELS_7, label_to_model=LABEL_TO_MODEL_7,
        label_scale=40,
        norm="patch",          # each 320×320 patch is normalised separately, as in training
        check_keys=True,       # refuse to run with random weights if the checkpoint does not match
        combined=[("Vein (Bundle_Sheath + Vascular_Tissue)", [2, 3])],
        suffix="-7",
        blurb=("<p><strong>7 classes:</strong> Background, Epidermis, Bundle Sheath, Vascular Tissue, "
               "Palisade Mesophyll, Spongy Mesophyll, Air Space.</p>"
               "<p><strong>Caution:</strong> Bundle Sheath and Vascular Tissue are often confused with each other, "
               "so their accuracy is lower when they are counted separately (IoU about 0.62 and 0.52 on held-out leaves). "
               "Counted together as one <em>Vein</em> class, the IoU rises to about 0.80. For measurements, use the "
               "combined <em>Vein</em> row in the statistics table.</p>"),
    ),
}

_MODEL_CACHE = {}

def get_model(name, scheme="5"):
    """Lazily download, build, and cache the selected model. Returns None on failure (demo mode)."""
    cfg = SCHEMES[scheme]
    if name not in cfg["models"]:
        name = cfg["default"]
    key = (scheme, name)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    filename, builder = cfg["models"][name]
    try:
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", filename)
        if os.path.exists(local_path):                      # local copy (e.g. for testing before upload)
            print(f"Using local weights {local_path}")
            model_path = local_path
        else:
            from huggingface_hub import hf_hub_download
            print(f"Downloading {filename} from {MODEL_REPO}...")
            model_path = hf_hub_download(repo_id=MODEL_REPO, filename=filename)
        model      = builder(cfg["num_classes"])
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        result     = model.load_state_dict(state_dict, strict=False)
        if result.missing_keys or result.unexpected_keys:
            print(f"  Note: {len(result.missing_keys)} missing / {len(result.unexpected_keys)} unexpected key(s).")
        if cfg["check_keys"]:
            real_missing = [k for k in result.missing_keys if "criterion" not in k]
            if real_missing:
                raise RuntimeError(f"{len(real_missing)} weight(s) missing from {filename}, e.g. {real_missing[:2]}")
        model.eval().to(DEVICE)
        print(f"'{name}' ({cfg['tab']}) loaded on {DEVICE}.")
        _MODEL_CACHE[key] = model
        return model
    except Exception as e:
        print(f"WARNING: Could not load '{name}' ({cfg['tab']}) — {e}. Running in demo mode.")
        return None


# ── IMAGE HELPERS ─────────────────────────────────────────────────────────────
_HIGH_BIT_MODES = ("F", "I", "I;16", "I;16L", "I;16B", "I;16N")

def to_gray8(image: Image.Image) -> np.ndarray:
    """Grayscale uint8 array. 16/32-bit images (raw micro-CT slices) are rescaled to 8-bit using the
    0.5–99.5 percentile range of the slice; ordinary 8-bit / RGB images are simply converted to grey."""
    if image.mode in _HIGH_BIT_MODES:
        a = np.array(image, dtype=np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        hi = max(float(hi), float(lo) + 1e-6)
        return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return np.array(image.convert("L"))

def _display_rgb(image: Image.Image) -> Image.Image:
    """RGB version of an image for display; 16/32-bit images are rescaled first."""
    if image.mode in _HIGH_BIT_MODES:
        return Image.fromarray(to_gray8(image)).convert("RGB")
    return image.convert("RGB")


# ── PATCH-BASED INFERENCE ─────────────────────────────────────────────────────
def _gaussian_kernel(size: int) -> torch.Tensor:
    ax = torch.linspace(-1, 1, size)
    g  = torch.exp(-ax**2 / 0.5)
    k  = torch.outer(g, g)
    return k / k.max()

def _grid_positions(size: int, patch: int, stride: int) -> list:
    """Return top-left coordinates for patch tiling (matches evaluate.py logic)."""
    pos = list(range(0, max(1, size - patch + 1), stride))
    if not pos or pos[-1] != max(0, size - patch):
        pos.append(max(0, size - patch))
    return pos

def _normalise(t: torch.Tensor) -> torch.Tensor:
    """Z-score over the non-zero pixels (zeros are padding / empty background)."""
    valid = t[t > 0]
    if len(valid):
        mean, std = valid.mean(), valid.std()
        if std > 1e-5:
            t = (t - mean) / std
    return t

def run_patch_inference(mdl, image: Image.Image, scheme: str = "5") -> np.ndarray:
    """Tile → infer → Gaussian-weighted stitch. Returns (H, W) class-index array."""
    cfg       = SCHEMES[scheme]
    nc        = cfg["num_classes"]
    per_patch = cfg["norm"] == "patch"

    img_np = to_gray8(image).astype(np.float32)
    H, W   = img_np.shape

    img_t = torch.tensor(img_np).unsqueeze(0)
    if not per_patch:
        img_t = _normalise(img_t)

    accum = torch.zeros(nc, H, W)
    count = torch.zeros(H, W)
    gk    = _gaussian_kernel(PATCH_SIZE)

    with torch.no_grad():
        for top in _grid_positions(H, PATCH_SIZE, STRIDE):
            for left in _grid_positions(W, PATCH_SIZE, STRIDE):
                ph, pw = min(PATCH_SIZE, H - top), min(PATCH_SIZE, W - left)
                patch  = img_t[:, top:top+ph, left:left+pw]
                if per_patch:
                    if ph < PATCH_SIZE or pw < PATCH_SIZE:
                        patch = F.pad(patch, (0, PATCH_SIZE - pw, 0, PATCH_SIZE - ph), value=0)
                    patch = _normalise(patch)
                elif ph < PATCH_SIZE or pw < PATCH_SIZE:
                    patch = F.pad(patch, (0, PATCH_SIZE - pw, 0, PATCH_SIZE - ph), mode="replicate")
                out    = mdl(patch.unsqueeze(0).to(DEVICE)).squeeze(0).float().cpu()
                weight = gk[:ph, :pw]
                accum[:, top:top+ph, left:left+pw] += out[:, :ph, :pw] * weight.unsqueeze(0)
                count[top:top+ph, left:left+pw]    += weight

    return (accum / count.clamp(min=1e-6).unsqueeze(0)).numpy().argmax(axis=0).astype(np.uint8)


# ── OUTPUT HELPERS ────────────────────────────────────────────────────────────
def mask_to_images(mask_array: np.ndarray, scheme: str = "5"):
    """Return (label_img, color_mask) as PIL RGB images."""
    cfg        = SCHEMES[scheme]
    color_mask = Image.fromarray(cfg["class_colors"][mask_array])
    label_img  = Image.fromarray((mask_array * cfg["label_scale"]).astype(np.uint8)).convert("RGB")
    return label_img, color_mask

def overlay_mask(original: Image.Image, mask: Image.Image, alpha: float = 0.5) -> Image.Image:
    return Image.blend(_display_rgb(original).resize(mask.size), mask, alpha=alpha)

def _stat_rows(counts, total, cfg) -> list:
    """[class, count, percentage] rows, followed by the combined rows (e.g. Vein) of the scheme."""
    rows = []
    for c, name in enumerate(cfg["class_names"]):
        n = int(counts[c])
        rows.append([name, f"{n:,}", f"{100 * n / total:.2f}%" if total > 0 else "0.00%"])
    for label, idxs in cfg["combined"]:
        n = int(sum(counts[i] for i in idxs))
        rows.append([label, f"{n:,}", f"{100 * n / total:.2f}%" if total > 0 else "0.00%"])
    return rows

def _na_rows(cfg) -> list:
    return [[name, "N/A", "N/A"] for name in cfg["class_names"] + [l for l, _ in cfg["combined"]]]

def area_stats(mask_array: np.ndarray, scheme: str = "5") -> list:
    cfg    = SCHEMES[scheme]
    counts = np.bincount(mask_array.ravel(), minlength=cfg["num_classes"])
    return _stat_rows(counts, mask_array.size, cfg)

def save_csv(rows: list, headers: list, filename: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), filename)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    return path

def save_image(img: Image.Image, filename: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), filename)
    img.save(path)
    return path

def save_tiff_stack(images: list, filename: str) -> str:
    path = os.path.join(tempfile.mkdtemp(), filename)
    images[0].save(path, save_all=True, append_images=images[1:])
    return path


# ── SINGLE IMAGE INFERENCE ────────────────────────────────────────────────────
# @spaces.GPU   # ← uncomment for HuggingFace ZeroGPU
def run_segmentation(input_image: Image.Image, model_label: str = None, scheme: str = "5"):
    cfg = SCHEMES[scheme]
    if input_image is None:
        raise gr.Error("Please upload a CT scan image first.")

    model = get_model(cfg["label_to_model"].get(model_label, cfg["default"]), scheme)
    if model is None:
        ph    = _display_rgb(input_image)
        ovl   = overlay_mask(input_image, ph)
        empty = _na_rows(cfg)
        return (
            ph, ph, ovl, empty,
            save_csv(empty, ["Class", "Pixels", "Percentage"], "area_stats.csv"),
            save_image(ph,  "label.png"),
            save_image(ph,  "color_mask.png"),
            save_image(ovl, "overlay.png"),
            "Demo mode — no model loaded.",
        )

    try:
        mask_array       = run_patch_inference(model, input_image, scheme)
        label_img, color = mask_to_images(mask_array, scheme)
        ovl              = overlay_mask(input_image, color)
        stats            = area_stats(mask_array, scheme)
        return (
            label_img, color, ovl, stats,
            save_csv(stats, ["Class", "Pixels", "Percentage"], "area_stats.csv"),
            save_image(label_img, "label.png"),
            save_image(color,     "color_mask.png"),
            save_image(ovl,       "overlay.png"),
            "Done — Segmentation complete.",
        )
    except Exception as e:
        raise gr.Error(f"Inference failed: {e}")


# ── STACK INFERENCE ───────────────────────────────────────────────────────────
def preview_and_store_stack(f):
    if f is None:
        return None, None, "Please upload a TIFF stack file."
    filepath = f.name if hasattr(f, "name") else f
    try:
        tiff = Image.open(filepath)
        n    = getattr(tiff, "n_frames", 1)
        tiff.seek(n // 2)
        return _display_rgb(tiff.copy()), filepath, f"Stack uploaded ({n} slices) — Click 'Run Segmentation on Stack'."
    except Exception as e:
        return None, None, f"Error reading stack: {e}"


# @spaces.GPU   # ← uncomment for HuggingFace ZeroGPU
def run_segmentation_stack(stack_path, model_label: str = None, scheme: str = "5"):
    cfg = SCHEMES[scheme]
    nc  = cfg["num_classes"]
    if stack_path is None:
        raise gr.Error("Please upload a stack image first.")
    try:
        tiff = Image.open(stack_path)
    except Exception as e:
        raise gr.Error(f"Could not open file: {e}")

    n_frames = getattr(tiff, "n_frames", 1)
    mid_idx  = n_frames // 2
    print(f"Stack: {n_frames} slice(s).")

    model = get_model(cfg["label_to_model"].get(model_label, cfg["default"]), scheme)
    if model is None:
        tiff.seek(mid_idx)
        ph    = _display_rgb(tiff.copy())
        empty = _na_rows(cfg)
        return (
            ph, ph, overlay_mask(ph, ph),
            empty,
            save_csv(empty, ["Class", "Voxels", "Volume %"],           "volume_stats.csv"),
            save_csv([],    ["Slice", "Class", "Pixels", "Percentage"], "area_per_slice.csv"),
            save_image(ph, "label.png"),
            save_image(ph, "color_mask.png"),
            save_image(ph, "overlay.png"),
            f"Demo mode — Showed slice {mid_idx + 1} of {n_frames}, no model loaded.",
        )

    try:
        total_counts = np.zeros(nc, dtype=np.int64)
        per_slice_data, label_imgs, color_masks, overlays = [], [], [], []
        mid_label = mid_color = mid_overlay = None

        for i in range(n_frames):
            tiff.seek(i)
            slice_img        = tiff.copy()
            mask_array       = run_patch_inference(model, slice_img, scheme)
            label_img, color = mask_to_images(mask_array, scheme)
            ovl              = overlay_mask(slice_img, color)

            total_counts += np.bincount(mask_array.ravel(), minlength=nc)
            per_slice_data.append((i, mask_array))
            label_imgs.append(label_img)
            color_masks.append(color)
            overlays.append(ovl)

            if i == mid_idx:
                mid_label, mid_color, mid_overlay = label_img, color, ovl

        stats = _stat_rows(total_counts, int(total_counts.sum()), cfg)
        perslice_rows = []
        for idx, ma in per_slice_data:
            counts = np.bincount(ma.ravel(), minlength=nc)
            for name, n_px, pct in _stat_rows(counts, ma.size, cfg):
                perslice_rows.append([idx + 1, name, n_px, pct])

        return (
            mid_label, mid_color, mid_overlay,
            stats,
            save_csv(stats,         ["Class", "Voxels", "Volume %"],           "volume_stats.csv"),
            save_csv(perslice_rows, ["Slice", "Class", "Pixels", "Percentage"], "area_per_slice.csv"),
            save_tiff_stack(label_imgs,  "label_all_slices.tif"),
            save_tiff_stack(color_masks, "color_all_slices.tif"),
            save_tiff_stack(overlays,    "overlay_all_slices.tif"),
            f"Done — {n_frames} slices processed, showing middle slice ({mid_idx + 1}).",
        )
    except Exception as e:
        raise gr.Error(f"Stack inference failed: {e}")


# ── CSS ───────────────────────────────────────────────────────────────────────
css = """
/* ══════════════════════════════════════════════════════════════════════════
   DESIGN TOKENS — change colors in one place, affects everything
   ══════════════════════════════════════════════════════════════════════════ */
:root {
    --c-bg:          #f7faf7;   /* page background          */
    --c-surface:     #ffffff;   /* card / block background  */
    --c-surface-alt: #f1f8f1;   /* subtle tinted surface    */
    --c-green:       #2e7d32;   /* primary accent           */
    --c-green-dark:  #1b5e20;   /* hover / heading          */
    --c-green-light: #c8e6c9;   /* borders, hover tint      */
    --c-green-pale:  #e8f5e9;   /* table header bg          */
    --c-green-row:   #f7fdf7;   /* table striped row        */
    --c-img-bg:      #a8bfaa;   /* image block background   */
    --c-border:      #a5d6a7;   /* input borders            */
    --c-text:        #111111;   /* body text                */
    --c-text-inv:    #ffffff;   /* text on dark backgrounds */
    --radius-sm:     8px;
    --radius-md:     10px;
}

/* ── GLOBAL RESET ────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box !important; }

body {
    background: var(--c-bg) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif !important;
    font-size: 1rem !important;
}
.gradio-container {
    background: var(--c-bg) !important;
    color: var(--c-text) !important;
    max-width: 100% !important;
    padding: 20px !important;
}
footer { display: none !important; }

/* ── TYPOGRAPHY ──────────────────────────────────────────────────────────── */
h1 { font-size: 2.2rem !important; color: var(--c-green-dark) !important; }
h2 { font-size: 1.9rem !important; color: var(--c-green) !important; }
h3, h4 { font-size: 1.3rem !important; color: var(--c-green) !important; }

/* color first = Firefox fallback; -webkit-text-fill-color = Chrome/Safari */
p, label, span, td, th, strong, b, em {
    font-size: 1.0625rem !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
}
.label-wrap span, label span {
    font-size: 1.0625rem !important;
    font-weight: 600 !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
}
button { font-size: 1.0625rem !important; font-weight: 600 !important; }

/* ── BLOCK / PANEL ───────────────────────────────────────────────────────── */
.block, .gr-box, .gr-panel, .gr-form, .gr-padded, .wrap {
    background: var(--c-surface) !important;
}
.block { overflow: hidden !important; border-radius: var(--radius-sm) !important; }

/* ── TABS ────────────────────────────────────────────────────────────────── */
button[role="tab"], .tab-nav button {
    color: var(--c-green) !important;
    background: var(--c-surface-alt) !important;
    border: 1px solid var(--c-green-light) !important;
    font-size: 1.0625rem !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
    -webkit-transition: background 0.15s ease, color 0.15s ease !important;
            transition: background 0.15s ease, color 0.15s ease !important;
}

/* Remove Gradio 6 orange accent line */
[role="tablist"]::before, [role="tablist"]::after,
.tab-nav::before, .tab-nav::after,
button[role="tab"]::before, button[role="tab"]::after,
button[role="tab"] span::after { display: none !important; content: "" !important; }
button[role="tab"] { border-bottom: none !important; box-shadow: none !important; outline: none !important; }
button[role="tab"][aria-selected="true"] {
    background: var(--c-green) !important; border-color: var(--c-green) !important;
    color: var(--c-text-inv) !important; -webkit-text-fill-color: var(--c-text-inv) !important;
    border-bottom: none !important; box-shadow: none !important;
}
button[role="tab"]:hover:not([aria-selected="true"]) {
    background: var(--c-green-pale) !important; border-color: var(--c-green) !important;
}

[role="tablist"], .tab-nav {
    background: var(--c-surface-alt) !important;
    border-bottom: none !important;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
    padding: 6px 6px 0 6px !important;
    display: -webkit-flex !important;
    display: flex !important;
    -webkit-flex-wrap: wrap !important;
    flex-wrap: wrap !important;
    gap: 4px !important;
}
[role="tablist"] > *, .tab-nav > * { margin: 2px !important; }
[role="tabpanel"], .tabitem { background: var(--c-surface) !important; padding: 12px !important; }

/* ── OUTER FRAMES ────────────────────────────────────────────────────────── */
#main-tabs {
    border: 2px solid var(--c-green) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    background: var(--c-surface) !important;
}
#result-col, #stack-result-col {
    border: 2px solid var(--c-green) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
    background: var(--c-surface) !important;
    padding: 0 !important;
    box-shadow: none !important;
}
#result-col [role="tablist"], #stack-result-col [role="tablist"] {
    border-bottom: 2px solid var(--c-green) !important;
    background: var(--c-surface-alt) !important;
    margin: 0 !important;
    padding: 6px 6px 0 6px !important;
    border-radius: 0 !important;
}
#result-col [role="tabpanel"], #stack-result-col [role="tabpanel"] {
    border: none !important;
    padding: 8px !important;
    background: var(--c-surface) !important;
}
#result-col .block, #stack-result-col .block, #result-tabs, #stack-result-tabs {
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    background: transparent !important;
}
#main-row, #stack-main-row {
    border: none !important;
    padding: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* ── STRIP THE SHARED OUTER FORM CONTAINER ───────────────────────────────── */
/* Gradio wraps all direct children in one <form> — remove its visual frame   */
/* so model-selector and status-block each appear as their own separate card. */
.gradio-container form,
.gradio-container .form {
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    padding: 0 !important;
    gap: 8px !important;
}
.gradio-container .gap {
    background: transparent !important;
}

/* ── MODEL SELECTOR ──────────────────────────────────────────────────────── */
#model-selector {
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 12px 16px !important;
    background: var(--c-surface) !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
}
/* Stack each radio option on its own line */
#model-selector .wrap {
    display: flex !important;
    flex-direction: column !important;
    gap: 6px !important;
}
/* "Segmentation Model" title — bold */
#model-selector > div > span,
#model-selector > span,
#model-selector .label-wrap span,
#model-selector label:not(:has(input)) span {
    font-weight: 700 !important;
    font-size: 1.0625rem !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
}
/* Each radio button pill — normal weight */
#model-selector label.svelte-1gfkyw6,
#model-selector label {
    width: 100% !important;
    background: var(--c-surface-alt) !important;
    border: 1px solid var(--c-green-light) !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 14px !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
    font-size: 1rem !important;
    font-weight: 400 !important;
    cursor: pointer !important;
    -webkit-transition: background 0.15s ease, border-color 0.15s ease !important;
            transition: background 0.15s ease, border-color 0.15s ease !important;
}
#model-selector label:has(input:checked) {
    background: var(--c-green-pale) !important;
    border-color: var(--c-green) !important;
    font-weight: 400 !important;
}
/* Force radio option spans to normal weight (overrides global label span rule) */
#model-selector label span,
#model-selector label:has(input) span {
    font-weight: 400 !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
}

/* ── STATUS ──────────────────────────────────────────────────────────────── */
#status-block {
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 8px 16px !important;
    background: var(--c-surface-alt) !important;
    margin-top: 12px !important;
}
.deployment-block {
    border: 1px solid var(--c-border) !important;
    border-radius: var(--radius-sm) !important;
    padding: 10px 16px !important;
    background: var(--c-surface-alt) !important;
    margin-top: 12px !important;
}
.deployment-block p {
    margin: 0 !important;
    font-size: 1.0625rem !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
}
.deployment-block strong {
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
}
.deployment-block a {
    color: var(--c-green-dark) !important;
    -webkit-text-fill-color: var(--c-green-dark) !important;
    text-decoration: underline !important;
}
#status-block label {
    font-weight: 700 !important;
    color: var(--c-green) !important;
    -webkit-text-fill-color: var(--c-green) !important;
}
#status-block textarea {
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
    background: var(--c-surface-alt) !important;
    font-size: 0.9375rem !important;
}

/* ── INPUTS ──────────────────────────────────────────────────────────────── */
textarea, input[type="text"] {
    background: var(--c-surface-alt) !important;
    border-color: var(--c-border) !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
}

/* ── BUTTONS — primary handled by theme; only secondary needs override ───── */
button.secondary {
    background: var(--c-surface-alt) !important;
    border: 1px solid var(--c-border) !important;
    color: var(--c-green) !important;
    -webkit-text-fill-color: var(--c-green) !important;
    transition: background 0.15s ease !important;
}
button.secondary:hover { background: var(--c-green-light) !important; }

/* ── TABLE / DATAFRAME ───────────────────────────────────────────────────── */
table, .table-wrap, .svelte-table, .gr-dataframe,
[data-testid="dataframe"], .dataframe-container { background: var(--c-surface) !important; }
.dataframe-container { margin-top: 2px !important; }
#area-table, #stack-area-table { margin-top: -20px !important; }
#dl-row, #stack-dl-row { margin-top: -20px !important; }

thead, thead tr, thead th, th,
[data-testid="dataframe"] thead th,
[data-testid="dataframe"] thead th span,
[data-testid="dataframe"] thead th *,
#area-table thead th, #area-table thead th span, #area-table thead th *,
#stack-area-table thead th, #stack-area-table thead th span, #stack-area-table thead th * {
    background: var(--c-green-pale) !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
    font-size: 0.9375rem !important;
    border-color: var(--c-green-light) !important;
}
tbody, tbody tr, tr { background: var(--c-surface) !important; }
tbody tr:nth-child(even), tr:nth-child(even) { background: var(--c-green-row) !important; }
td, tbody td, tr td,
#area-table td, #area-table td *, #area-table span, #area-table div,
#stack-area-table td, #stack-area-table td *, #stack-area-table span, #stack-area-table div {
    background: transparent !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
    border-color: var(--c-green-pale) !important;
    font-size: 0.9375rem !important;
}
/* Scope cell wrappers to dataframe only — avoids matching Gradio layout rows */
[data-testid="dataframe"] .cell-wrap,
[data-testid="dataframe"] [class*="cell"] {
    background: transparent !important;
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
}

/* ── IMAGE BLOCKS ────────────────────────────────────────────────────────── */
#img-file-upload, #img-file-upload .wrap, #img-file-upload > div,
#img-input, #img-input .wrap,
#img-label, #img-label .wrap,
#img-color, #img-color .wrap,
#img-overlay, #img-overlay .wrap,
#img-stack-preview, #img-stack-preview .wrap,
#img-stack-label, #img-stack-label .wrap,
#img-stack-color, #img-stack-color .wrap,
#img-stack-overlay, #img-stack-overlay .wrap,
#stack-file-upload, #stack-file-upload .wrap, #stack-file-upload > div {
    background-color: var(--c-img-bg) !important;
}

/* Image label badges */
#img-file-upload label, #img-file-upload label *, #img-input label, #img-input label *,
#img-label label, #img-label label *, #img-color label, #img-color label *,
#img-overlay label, #img-overlay label *,
#img-stack-preview label, #img-stack-preview label *, #img-stack-label label, #img-stack-label label *,
#img-stack-color label, #img-stack-color label *, #img-stack-overlay label, #img-stack-overlay label *,
#stack-file-upload label, #stack-file-upload label *,
#img-file-upload [data-testid="block-label"], #img-input [data-testid="block-label"],
#img-label [data-testid="block-label"], #img-color [data-testid="block-label"],
#img-overlay [data-testid="block-label"], #img-stack-preview [data-testid="block-label"],
#img-stack-label [data-testid="block-label"], #img-stack-color [data-testid="block-label"],
#img-stack-overlay [data-testid="block-label"], #stack-file-upload [data-testid="block-label"] {
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    font-size: 1.0625rem !important;
    font-weight: 600 !important;
}

/* Drop-zone text */
#img-file-upload .wrap p, #img-file-upload .wrap span,
#img-input .wrap p, #img-input .wrap span,
#img-stack-preview .wrap p, #img-stack-preview .wrap span,
#stack-file-upload .wrap p, #stack-file-upload .wrap span {
    color: var(--c-text-inv) !important;
    -webkit-text-fill-color: var(--c-text-inv) !important;
}

/* Image display — contain, not crop */
#img-input img, #img-label img, #img-color img, #img-overlay img,
#img-stack-preview img, #img-stack-label img, #img-stack-color img, #img-stack-overlay img {
    -o-object-fit: contain !important;
    object-fit: contain !important;
    width: 100% !important;
    height: 100% !important;
}

/* ── FILE UPLOAD DROP ZONES — compact ───────────────────────────────────── */
#img-file-upload, #img-file-upload > div, #img-file-upload .wrap,
#stack-file-upload, #stack-file-upload > div, #stack-file-upload .wrap {
    min-height: unset !important;
    height: 70px !important;
    max-height: 70px !important;
}
#img-file-upload .wrap, #stack-file-upload .wrap {
    -webkit-flex-direction: row !important; flex-direction: row !important;
    -webkit-align-items: center !important; align-items: center !important;
    -webkit-justify-content: center !important; justify-content: center !important;
    gap: 8px !important;
}
#img-file-upload .wrap p, #stack-file-upload .wrap p { margin: 0 !important; }
#img-file-upload .wrap svg, #stack-file-upload .wrap svg { width: 20px !important; height: 20px !important; }

/* Hide image input toolbar — class/attribute selectors only; no positional div:last-child
   (that selector hides the image preview after upload in Gradio 6) */
#img-input .source-selection, #img-input [data-testid="source-select"],
#img-input .icon-buttons, #img-input .toolbar { display: none !important; }

/* ── DOWNLOAD BLOCKS ─────────────────────────────────────────────────────── */
#dl-csv, #dl-label, #dl-color, #dl-overlay,
#stack-dl-csv, #stack-dl-perslice, #stack-dl-label, #stack-dl-color, #stack-dl-overlay {
    background: var(--c-surface) !important;
}
#dl-csv .upload-container, #dl-label .upload-container,
#dl-color .upload-container, #dl-overlay .upload-container,
#stack-dl-csv .upload-container, #stack-dl-perslice .upload-container,
#stack-dl-label .upload-container, #stack-dl-color .upload-container,
#stack-dl-overlay .upload-container,
#dl-csv [data-testid="file"], #dl-label [data-testid="file"],
#dl-color [data-testid="file"], #dl-overlay [data-testid="file"],
#stack-dl-csv [data-testid="file"], #stack-dl-perslice [data-testid="file"],
#stack-dl-label [data-testid="file"], #stack-dl-color [data-testid="file"],
#stack-dl-overlay [data-testid="file"],
#dl-csv > div, #dl-label > div, #dl-color > div, #dl-overlay > div,
#stack-dl-csv > div, #stack-dl-perslice > div, #stack-dl-label > div,
#stack-dl-color > div, #stack-dl-overlay > div {
    min-height: unset !important;
    height: 60px !important;
    max-height: 60px !important;
    overflow: hidden !important;
}
#dl-csv label, #dl-csv label *, #dl-label label, #dl-label label *,
#dl-color label, #dl-color label *, #dl-overlay label, #dl-overlay label *,
#stack-dl-csv label, #stack-dl-csv label *, #stack-dl-perslice label, #stack-dl-perslice label *,
#stack-dl-label label, #stack-dl-label label *, #stack-dl-color label, #stack-dl-color label *,
#stack-dl-overlay label, #stack-dl-overlay label *,
#dl-csv [data-testid="block-label"], #dl-label [data-testid="block-label"],
#dl-color [data-testid="block-label"], #dl-overlay [data-testid="block-label"],
#stack-dl-csv [data-testid="block-label"], #stack-dl-perslice [data-testid="block-label"],
#stack-dl-label [data-testid="block-label"], #stack-dl-color [data-testid="block-label"],
#stack-dl-overlay [data-testid="block-label"],
#dl-csv a, #dl-csv span, #dl-label a, #dl-label span,
#dl-color a, #dl-color span, #dl-overlay a, #dl-overlay span,
#stack-dl-csv a, #stack-dl-csv span, #stack-dl-perslice a, #stack-dl-perslice span,
#stack-dl-label a, #stack-dl-label span, #stack-dl-color a, #stack-dl-color span,
#stack-dl-overlay a, #stack-dl-overlay span {
    color: var(--c-text) !important;
    -webkit-text-fill-color: var(--c-text) !important;
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* ── EXAMPLES GALLERY (single image + stack) ─────────────────────────────── */
#example-gallery, #stack-example-gallery {
    gap: 10px !important; margin-top: 4px !important;
    flex-wrap: nowrap !important; overflow-x: auto !important;
    background: transparent !important;
}
#example-gallery .ex-card, #stack-example-gallery .stack-ex-card {
    padding: 10px 8px !important; border-radius: 8px !important; overflow: hidden !important;
    background: var(--c-img-bg) !important; border: 1px solid var(--c-green-light) !important;
    text-align: center !important;
    -webkit-transition: background-color 0.15s ease, border-color 0.15s ease !important;
            transition: background-color 0.15s ease, border-color 0.15s ease !important;
}
#example-gallery .ex-card:hover, #stack-example-gallery .stack-ex-card:hover {
    background: var(--c-green-pale) !important; border-color: var(--c-green) !important;
}
#example-gallery .ex-thumb img, #stack-example-gallery .stack-ex-thumb img {
    width: 100% !important; object-fit: contain !important; border-radius: 4px !important;
}
#example-gallery .ex-thumb, #example-gallery .ex-thumb .wrap, #example-gallery .ex-thumb > div,
#stack-example-gallery .stack-ex-thumb, #stack-example-gallery .stack-ex-thumb .wrap,
#stack-example-gallery .stack-ex-thumb > div {
    background: transparent !important; border: none !important; box-shadow: none !important;
}
#example-gallery .ex-thumb label, #example-gallery .ex-thumb [data-testid="block-label"],
#stack-example-gallery .stack-ex-thumb label,
#stack-example-gallery .stack-ex-thumb [data-testid="block-label"] { display: none !important; }
#example-gallery .ex-btn, #stack-example-gallery .stack-ex-btn {
    margin-top: 6px !important; width: 100% !important;
    background: transparent !important; border: none !important; box-shadow: none !important;
    font-size: 0.875rem !important; font-weight: 600 !important;
    color: var(--c-text) !important; -webkit-text-fill-color: var(--c-text) !important;
    padding: 0 !important; cursor: pointer !important;
}
#example-gallery .ex-btn:hover, #stack-example-gallery .stack-ex-btn:hover {
    color: var(--c-green) !important; -webkit-text-fill-color: var(--c-green) !important;
}

/* ── RESPONSIVE ──────────────────────────────────────────────────────────── */
@media (max-width: 900px) {
    #main-row, #stack-main-row {
        -webkit-flex-direction: column !important;
        flex-direction: column !important;
    }
    #result-col, #stack-result-col { width: 100% !important; }
}
@media (max-width: 600px) {
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.2rem !important; }
    .gradio-container { padding: 10px !important; }
    #stack-file-upload, #stack-file-upload > div, #stack-file-upload .wrap {
        height: auto !important;
        max-height: none !important;
    }
    #dl-row, #stack-dl-row { -webkit-flex-wrap: wrap !important; flex-wrap: wrap !important; }
}
"""


# ── UI HELPERS ────────────────────────────────────────────────────────────────
def load_uploaded_file(f):
    """Open an uploaded file into a PIL Image, forcing full load before temp cleanup."""
    if f is None:
        return None, gr.update()
    img = Image.open(f.name)
    img.load()   # force read into memory — temp file may be deleted after return
    if img.mode in _HIGH_BIT_MODES:                      # raw 16/32-bit slice: show it as 8-bit
        img = Image.fromarray(to_gray8(img))
    return img, "Image uploaded — Click 'Run Segmentation' to process."


# ── GRADIO UI ─────────────────────────────────────────────────────────────────
# Every element id used by the stylesheet. The 7-class section builds the same interface with the
# suffix "-7" on each id, and the stylesheet rules are duplicated for those ids (ids must be unique).
_UI_IDS = [
    "main-tabs", "main-row", "result-col", "result-tabs", "dl-row", "example-gallery", "model-selector",
    "status-block", "img-file-upload", "img-input", "img-label", "img-color", "img-overlay", "run-btn",
    "area-table", "dl-csv", "dl-label", "dl-color", "dl-overlay",
    "stack-file-upload", "img-stack-preview", "img-stack-label", "img-stack-color", "img-stack-overlay",
    "stack-run-btn", "stack-example-gallery", "stack-area-table", "stack-dl-row", "stack-dl-csv",
    "stack-dl-perslice", "stack-dl-label", "stack-dl-color", "stack-dl-overlay", "stack-main-row",
    "stack-result-col", "stack-result-tabs",
]

def _css_with_suffix_rules(css_text: str, suffix: str) -> str:
    """Append, right after every rule that targets one of the UI ids, a copy of that rule for the
    suffixed ids, so the second (7-class) interface is styled exactly like the first."""
    import re
    id_re = re.compile(r"#(" + "|".join(sorted(_UI_IDS, key=len, reverse=True)) + r")(?![\w-])")
    rule_re = re.compile(r"([^{}]+)\{([^{}]*)\}")

    def dup(m):
        selector, body = m.group(1), m.group(2)
        if not id_re.search(selector):
            return m.group(0)
        return m.group(0) + "\n" + id_re.sub(lambda i: "#" + i.group(1) + suffix, selector).strip() + " {" + body + "}"

    return rule_re.sub(dup, css_text)


theme_obj = gr.themes.Default(primary_hue=gr.themes.colors.green, neutral_hue=gr.themes.colors.gray)

goatcounter_head = """
<script data-goatcounter="https://leafctwebapp.goatcounter.com/count"
        data-goatcounter-settings='{"allow_frame": true}'
        async src="//gc.zgo.at/count.js"></script>
"""

blocks_kwargs = {
    "title": "Leaf CT Scan Segmentation",
    "theme": theme_obj,
    "css": _css_with_suffix_rules(css, SCHEMES["7"]["suffix"]),
    "head": goatcounter_head,
}

launch_kwargs = {
    "server_name": "0.0.0.0",
    "server_port": 7860,
    "ssr_mode": False,
    "debug": False,
}


def build_section(scheme: str):
    """Build the Single Image / Stack Image interface for one class scheme (5 or 7 classes)."""
    cfg         = SCHEMES[scheme]
    sfx         = cfg["suffix"]
    eid         = lambda name: name + sfx
    empty_table = [["—", "—", "—"]] * (cfg["num_classes"] + len(cfg["combined"]))

    # Event handlers bound to this scheme
    def _run(img, model_label):
        return run_segmentation(img, model_label, scheme)

    def _run_stack(path, model_label):
        return run_segmentation_stack(path, model_label, scheme)

    gr.HTML(cfg["blurb"])

    model_selector = gr.Radio(
        choices=list(cfg["labels"].values()),
        value=cfg["labels"][cfg["default"]],
        label="Segmentation Model",
        elem_id=eid("model-selector"),
    )

    status = gr.Textbox(
        label="Status",
        value="Ready — Please upload a CT scan and click Run.",
        interactive=False,
        elem_id=eid("status-block"),
    )

    with gr.Tabs(elem_id=eid("main-tabs")):

        # ── TAB 1: SINGLE IMAGE ───────────────────────────────────────────────
        with gr.Tab("Single Image"):
            with gr.Row(equal_height=True, elem_id=eid("main-row")):
                with gr.Column():
                    input_file = gr.File(
                        label="Upload CT Scan (.png / .jpg / .tif)",
                        file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff"],
                        height=90, elem_id=eid("img-file-upload"),
                    )
                    input_image = gr.Image(
                        type="pil", label="Preview",
                        interactive=False, height=400, elem_id=eid("img-input"),
                    )
                with gr.Column(elem_id=eid("result-col")):
                    with gr.Tabs(elem_id=eid("result-tabs")):
                        with gr.Tab("Class Label"):
                            output_label = gr.Image(type="pil", label="Class Label", interactive=False, height=500, elem_id=eid("img-label"))
                        with gr.Tab("Color Mask"):
                            output_color = gr.Image(type="pil", label="Color Mask", interactive=False, height=500, elem_id=eid("img-color"))
                        with gr.Tab("Overlay"):
                            output_overlay = gr.Image(type="pil", label="Overlay", interactive=False, height=500, elem_id=eid("img-overlay"))

            with gr.Row():
                submit_btn = gr.Button("Run Segmentation", variant="primary", elem_id=eid("run-btn"))
                clear_btn  = gr.Button("Clear", variant="secondary")

            gr.HTML("<h2>Example Images</h2><p>Click an image's name below to load it.</p>")
            with gr.Row(elem_id=eid("example-gallery")):
                example_btns = []
                for path, name in EXAMPLE_PATHS:
                    with gr.Column(scale=1, min_width=120, elem_classes=["ex-card"]):
                        gr.Image(value=path, interactive=False, show_label=False,
                                 height=130, elem_classes=["ex-thumb"])
                        example_btns.append(gr.Button(name, size="sm", elem_classes=["ex-btn"]))

            gr.HTML("<h2>Area Statistics</h2>")
            area_table = gr.Dataframe(
                headers=["Class", "Pixels", "Percentage"],
                value=empty_table, show_label=False, interactive=False, elem_id=eid("area-table"),
            )
            gr.HTML("<h2>Save Output</h2>")
            with gr.Row(elem_id=eid("dl-row")):
                dl_csv     = gr.File(label="1. Area Statistics", interactive=False, elem_id=eid("dl-csv"),     height=80)
                dl_label   = gr.File(label="2. Label Mask",      interactive=False, elem_id=eid("dl-label"),   height=80)
                dl_color   = gr.File(label="3. Color Mask",      interactive=False, elem_id=eid("dl-color"),   height=80)
                dl_overlay = gr.File(label="4. Overlay Image",   interactive=False, elem_id=eid("dl-overlay"), height=80)

            # ── Events
            input_file.upload(fn=load_uploaded_file, inputs=[input_file], outputs=[input_image, status])
            input_image.change(
                fn=lambda img: "Image loaded — Click 'Run Segmentation' to process." if img is not None else gr.update(),
                inputs=[input_image],
                outputs=[status],
            )
            run_event = submit_btn.click(
                fn=_run,
                inputs=[input_image, model_selector],
                outputs=[output_label, output_color, output_overlay, area_table, dl_csv, dl_label, dl_color, dl_overlay, status],
            )
            clear_btn.click(
                fn=lambda: (None, None, None, None, None, empty_table, None, None, None, None, "Ready — Please upload a CT scan and click Run."),
                outputs=[input_file, input_image, output_label, output_color, output_overlay, area_table, dl_csv, dl_label, dl_color, dl_overlay, status],
                cancels=[run_event],
            )
            for btn, (path, _) in zip(example_btns, EXAMPLE_PATHS):
                btn.click(
                    fn=lambda p=path: (Image.open(p).convert("L"), "Image loaded — Click 'Run Segmentation' to process."),
                    outputs=[input_image, status],
                )

        # ── TAB 2: TIFF STACK ─────────────────────────────────────────────────
        with gr.Tab("Stack Image"):
            stack_path_state = gr.State(None)
            with gr.Row(equal_height=True, elem_id=eid("stack-main-row")):
                with gr.Column():
                    stack_file = gr.File(
                        label="Upload CT Stack (.tif / .tiff)",
                        file_types=[".tif", ".tiff"], height=90, elem_id=eid("stack-file-upload"),
                    )
                    stack_preview = gr.Image(
                        type="pil", label="Stack Preview (Middle slice)",
                        interactive=False, height=350, elem_id=eid("img-stack-preview"),
                    )
                with gr.Column(elem_id=eid("stack-result-col")):
                    with gr.Tabs(elem_id=eid("stack-result-tabs")):
                        with gr.Tab("Class Label"):
                            stack_output_label = gr.Image(type="pil", label="Class Label (Middle slice)", interactive=False, height=500, elem_id=eid("img-stack-label"))
                        with gr.Tab("Color Mask"):
                            stack_output_color = gr.Image(type="pil", label="Color Mask (Middle slice)",  interactive=False, height=500, elem_id=eid("img-stack-color"))
                        with gr.Tab("Overlay"):
                            stack_output_overlay = gr.Image(type="pil", label="Overlay (Middle slice)",   interactive=False, height=500, elem_id=eid("img-stack-overlay"))

            with gr.Row():
                stack_submit_btn = gr.Button("Run Segmentation on Stack", variant="primary", elem_id=eid("stack-run-btn"))
                stack_clear_btn  = gr.Button("Clear", variant="secondary")

            gr.HTML("<h2>Example Stacks</h2><p>Click a stack'name below to load it.</p>")
            with gr.Row(elem_id=eid("stack-example-gallery")):
                stack_example_btns = []
                for _, preview, name in STACK_EXAMPLE_PATHS:
                    with gr.Column(scale=1, min_width=120, elem_classes=["stack-ex-card"]):
                        gr.Image(value=preview, interactive=False, show_label=False,
                                 height=100, elem_classes=["stack-ex-thumb"])
                        stack_example_btns.append(gr.Button(name, size="sm", elem_classes=["stack-ex-btn"]))

            gr.HTML("<h2>Volume Statistics (All slices)</h2>")
            stack_area_table = gr.Dataframe(
                headers=["Class", "Voxels", "Volume %"],
                value=empty_table, show_label=False, interactive=False, elem_id=eid("stack-area-table"),
            )
            gr.HTML("<h2>Save Output</h2>")
            with gr.Row(elem_id=eid("stack-dl-row")):
                stack_dl_csv          = gr.File(label="1. Volume Stats",    interactive=False, elem_id=eid("stack-dl-csv"),      height=80)
                stack_dl_perslice_csv = gr.File(label="2. Stats Per Slice", interactive=False, elem_id=eid("stack-dl-perslice"), height=80)
                stack_dl_label        = gr.File(label="3. Label Stack",     interactive=False, elem_id=eid("stack-dl-label"),    height=80)
                stack_dl_color        = gr.File(label="4. Color Stack",     interactive=False, elem_id=eid("stack-dl-color"),    height=80)
                stack_dl_overlay      = gr.File(label="5. Overlay Stack",   interactive=False, elem_id=eid("stack-dl-overlay"),  height=80)

            # ── Events
            stack_file.upload(
                fn=preview_and_store_stack,
                inputs=[stack_file],
                outputs=[stack_preview, stack_path_state, status],
            )
            stack_run_event = stack_submit_btn.click(
                fn=_run_stack,
                inputs=[stack_path_state, model_selector],
                outputs=[stack_output_label, stack_output_color, stack_output_overlay, stack_area_table,
                         stack_dl_csv, stack_dl_perslice_csv, stack_dl_label, stack_dl_color, stack_dl_overlay, status],
            )
            stack_clear_btn.click(
                fn=lambda: (None, None, None, None, None, None, empty_table, None, None, None, None, None, "Ready — Please upload a CT stack and click Run."),
                outputs=[stack_file, stack_preview, stack_path_state,
                         stack_output_label, stack_output_color, stack_output_overlay,
                         stack_area_table, stack_dl_csv, stack_dl_perslice_csv,
                         stack_dl_label, stack_dl_color, stack_dl_overlay, status],
                cancels=[stack_run_event],
            )
            for btn, (path, _preview, _name) in zip(stack_example_btns, STACK_EXAMPLE_PATHS):
                btn.click(
                    fn=lambda p=path: preview_and_store_stack(p),
                    outputs=[stack_preview, stack_path_state, status],
                )


with gr.Blocks(**blocks_kwargs) as demo:

    gr.HTML("""
        <img src="https://leafctwebapp.goatcounter.com/count?p=/huggingface-space&t=Leaf%20CT%20Segmentation%20Webapp"
             alt="" width="1" height="1" style="position:absolute; left:-9999px; width:1px; height:1px;">
        <h1>Leaf CT Scan Segmentation</h1>
        <p>Automatic leaf CT scan segmentation with transformer-based models, in two versions: <strong>5 classes</strong>
        (four models: EoMT/DINOv3, Mask2Former/Swin-B, SegFormer-B4, FPN/MiT-B4) and <strong>7 classes</strong>
        (Mask2Former/Swin-B, with Bundle Sheath, Vascular Tissue, Palisade and Spongy Mesophyll as separate classes). Pick a tab below.</p>
        <p><strong>Features:</strong> Upload a single image or a multi-slice TIFF stack to visualize tissue segmentation and export area statistics.
        &nbsp;·&nbsp; <a href="https://github.com/WorasitSangjan/WebApp-Leaf-microCT-Segmentation/blob/main/MANUAL.md" target="_blank">📖 User Manual</a></p>
        <div class="deployment-block">
            <p>
                <strong>Deployment Options:</strong> For accelerated processing, please run this application via:
                <a href="https://colab.research.google.com/github/WorasitSangjan/WebApp-Leaf-microCT-Segmentation/blob/main/run_colab.ipynb" target="_blank">Google Colab (GPU)</a> |
                <a href="https://github.com/WorasitSangjan/WebApp-Leaf-microCT-Segmentation#option-3--run-locally-python-310" target="_blank">Your Local Computer</a>
            </p>
        </div>
        <div class="deployment-block">
            <p>
                <strong>Dataset:</strong> Browse curated plant leaf X-ray micro-CT datasets, preview images, segmentation labels, metadata, and repository links via:
                <a href="https://leafcthub.github.io/" target="_blank">Leaf CT Hub</a>
            </p>
        </div>

    """)

    with gr.Tabs(elem_id="scheme-tabs"):
        with gr.Tab(SCHEMES["5"]["tab"]):
            build_section("5")
        with gr.Tab(SCHEMES["7"]["tab"]):
            build_section("7")

    gr.Markdown("Note: Processing time depends on image size and server load.")


# ── LAUNCH ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(**launch_kwargs)
