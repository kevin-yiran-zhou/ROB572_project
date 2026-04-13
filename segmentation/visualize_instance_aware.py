import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from tqdm import tqdm
from transformers import SegformerImageProcessor

from train_segformer_instance_aware import SegFormerInstanceAware


# ==========================================
# 1. Config
# ==========================================
IMG_DIR = "lars_v1.0.0_images/test/images"   # 改成 test/images 也可以
OUTPUT_DIR = "vis_instance_aware/test"
MODEL_PATH = "segformer_instance_aware_best.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_MAP = {
    "Static Obstacle": [255, 0, 0],
    "Water": [0, 0, 255],
    "Sky": [0, 255, 0],
    "Boat": [255, 128, 0],
    "Buoy": [255, 0, 255],
    "Swimmer": [255, 255, 0],
    "Animal": [128, 255, 255],
    "Float": [180, 80, 255],
    "Other": [255, 255, 255],
}


# ==========================================
# 2. Load model
# ==========================================
processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")

ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = ckpt["class_names"]
img_size = ckpt["img_size"]

model = SegFormerInstanceAware(
    backbone_name="nvidia/mit-b0",
    num_classes=len(class_names)
)
model.load_state_dict(ckpt["model"])
model.to(DEVICE)
model.eval()


# ==========================================
# 3. Helpers
# ==========================================
def semantic_to_color(mask):
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)

    for i, cls_name in enumerate(class_names):
        rgb = COLOR_MAP.get(cls_name, [255, 255, 255])
        color[mask == i] = rgb

    return color


def random_color(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=3)


# ==========================================
# 4. Predict
# ==========================================
@torch.no_grad()
def predict(image_rgb):
    h, w = image_rgb.shape[:2]

    inputs = processor(
        images=image_rgb,
        return_tensors="pt",
        do_resize=True,
        size={"height": img_size, "width": img_size}
    )
    pixel_values = inputs["pixel_values"].to(DEVICE)

    semantic_logits, boundary_logits = model(pixel_values)

    semantic_logits = F.interpolate(
        semantic_logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )

    boundary_logits = F.interpolate(
        boundary_logits,
        size=(h, w),
        mode="bilinear",
        align_corners=False
    )

    semantic_pred = torch.argmax(semantic_logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    boundary_prob = torch.sigmoid(boundary_logits).squeeze().cpu().numpy()

    return semantic_pred, boundary_prob


# ==========================================
# 5. Visualization
# ==========================================
image_files = sorted([
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

print(f"Start visualization on {len(image_files)} images...")

for filename in tqdm(image_files):
    img_path = os.path.join(IMG_DIR, filename)
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        continue

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    semantic_pred, boundary_prob = predict(image_rgb)

    color_sem = semantic_to_color(semantic_pred)
    overlay_sem = cv2.addWeighted(image_rgb, 0.7, color_sem, 0.3, 0)

    # ==========================================
    # Instance-aware view
    # 用 boundary_prob 在每个前景类内部切开粘连区域
    # ==========================================
    thing_ids = [i for i, name in enumerate(class_names) if name not in ["Water", "Sky"]]
    instance_overlay = image_rgb.copy().astype(np.float32)

    instances_to_draw = []
    seed = 0

    for cls_id in thing_ids:
        cls_mask = (semantic_pred == cls_id)
        cls_name = class_names[cls_id]

        # 用 boundary 把当前类别内部可能的实例边界挖掉
        split_mask = cls_mask & (boundary_prob < 0.4)

        num_labels, labels = cv2.connectedComponents(split_mask.astype(np.uint8), connectivity=8)

        for inst_id in range(1, num_labels):
            comp = (labels == inst_id)
            if comp.sum() < 30:
                continue

            color = random_color(seed)
            seed += 1
            instance_overlay[comp] = instance_overlay[comp] * 0.5 + color * 0.5

            ys, xs = np.where(comp)
            cx = int(xs.mean())
            cy = int(ys.mean())

            instances_to_draw.append({
                "mask": comp,
                "label": cls_name,
                "cx": cx,
                "cy": cy,
                "color": color,
            })

    instance_overlay = instance_overlay.astype(np.uint8)

    # ==========================================
    # 4-panel layout
    # ==========================================
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    panels = [
        image_rgb,
        color_sem,
        overlay_sem,
        instance_overlay,
    ]
    titles = [
        "1. Input RGB",
        "2. Semantic Pred",
        "3. Semantic Overlay",
        "4. Instance-aware View",
    ]

    for ax, img, title in zip(axes, panels, titles):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axis("off")

    # 只在最后一个 panel 上画实例轮廓和 label
    ax_inst = axes[3]
    for item in instances_to_draw:
        mask = item["mask"]
        label = item["label"]
        cx = item["cx"]
        cy = item["cy"]
        color = item["color"]

        ax_inst.contour(
            mask.astype(np.uint8),
            levels=[0.5],
            colors=[color / 255.0],
            linewidths=1.5
        )
        ax_inst.text(
            cx, cy, label,
            color="white",
            fontsize=10,
            ha="center",
            va="center",
            bbox=dict(facecolor="black", alpha=0.8, pad=2, edgecolor="none")
        )

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, f"instance_aware_{filename}")
    plt.savefig(save_path, bbox_inches="tight", dpi=200)
    plt.close(fig)

print(f"Saved to: {OUTPUT_DIR}")
