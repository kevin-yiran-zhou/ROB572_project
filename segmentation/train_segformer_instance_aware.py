import os
import json
import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp
from transformers import SegformerModel, SegformerImageProcessor


# ==========================================
# 1. Config
# ==========================================
TRAIN_IMG_DIR = "lars_v1.0.0_images/train/images"
VAL_IMG_DIR = "lars_v1.0.0_images/val/images"

TRAIN_PANOPTIC_MASK_DIR = "lars_v1.0.0_annotations/train/panoptic_masks"
VAL_PANOPTIC_MASK_DIR = "lars_v1.0.0_annotations/val/panoptic_masks"

TRAIN_PANOPTIC_JSON = "lars_v1.0.0_annotations/train/panoptic_annotations.json"
VAL_PANOPTIC_JSON = "lars_v1.0.0_annotations/val/panoptic_annotations.json"

BEST_SAVE_PATH = "segformer_instance_aware_best.pth"
LAST_SAVE_PATH = "segformer_instance_aware_last.pth"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 768
BATCH_SIZE = 8
NUM_WORKERS = 4
NUM_EPOCHS = 20
LR = 6e-5
IGNORE_INDEX = 255
SEED = 42

# 合并版类别
CLASS_NAMES = [
    "Static Obstacle",
    "Water",
    "Sky",
    "Boat",
    "Buoy",
    "Swimmer",
    "Animal",
    "Float",
    "Other",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES)

# 原始 panoptic category id -> 合并后的类别
CATEGORY_ID_TO_CLASS_NAME = {
    1: "Static Obstacle",
    3: "Water",
    5: "Sky",
    11: "Boat",   # Boat/ship
    12: "Boat",   # Row boats
    13: "Boat",   # Paddle board
    14: "Buoy",
    15: "Swimmer",
    16: "Animal",
    17: "Float",
    19: "Other",
}

# 提高小目标和稀有类权重
# 顺序必须与 CLASS_NAMES 对齐
SEMANTIC_CLASS_WEIGHTS = [
    1.0,  # Static Obstacle
    1.0,  # Water
    1.0,  # Sky
    3.0,  # Boat
    4.0,  # Buoy
    5.0,  # Swimmer
    4.0,  # Animal
    4.0,  # Float
    2.0,  # Other
]


# ==========================================
# 2. Utils
# ==========================================
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)


def rgb2id(color):
    color = np.asarray(color, dtype=np.int64)
    return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]


def build_boundary_map(instance_id_map):
    """
    instance_id_map: [H, W], int
    输出: [H, W], float32, 0/1 boundary
    注意：这里不做 dilation，让 boundary 更细
    """
    h, w = instance_id_map.shape
    boundary = np.zeros((h, w), dtype=np.uint8)

    boundary[1:, :] |= (instance_id_map[1:, :] != instance_id_map[:-1, :])
    boundary[:-1, :] |= (instance_id_map[:-1, :] != instance_id_map[1:, :])
    boundary[:, 1:] |= (instance_id_map[:, 1:] != instance_id_map[:, :-1])
    boundary[:, :-1] |= (instance_id_map[:, :-1] != instance_id_map[:, 1:])

    return boundary.astype(np.float32)


# ==========================================
# 3. Dataset
# ==========================================
class LaRSPanopticDataset(Dataset):
    def __init__(self, img_dir, panoptic_mask_dir, panoptic_json_path, processor, img_size=640):
        self.img_dir = img_dir
        self.panoptic_mask_dir = panoptic_mask_dir
        self.processor = processor
        self.img_size = img_size

        self.image_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

        with open(panoptic_json_path, "r") as f:
            panoptic_data = json.load(f)

        self.categories = {c["id"]: c for c in panoptic_data["categories"]}
        self.ann_by_file = {a["file_name"]: a for a in panoptic_data["annotations"]}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base = os.path.splitext(img_name)[0]
        panoptic_name = base + ".png"

        img_path = os.path.join(self.img_dir, img_name)
        panoptic_path = os.path.join(self.panoptic_mask_dir, panoptic_name)

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        panoptic_rgb = np.array(Image.open(panoptic_path).convert("RGB"))
        panoptic_ids = rgb2id(panoptic_rgb)

        ann = self.ann_by_file[panoptic_name]

        semantic_mask = np.full(panoptic_ids.shape, IGNORE_INDEX, dtype=np.uint8)

        for seg in ann["segments_info"]:
            seg_id = seg["id"]
            cat_id = seg["category_id"]

            if cat_id not in CATEGORY_ID_TO_CLASS_NAME:
                continue

            cls_name = CATEGORY_ID_TO_CLASS_NAME[cat_id]
            cls_idx = CLASS_TO_IDX[cls_name]
            semantic_mask[panoptic_ids == seg_id] = cls_idx

        boundary_map = build_boundary_map(panoptic_ids)

        inputs = self.processor(
            images=image,
            return_tensors="pt",
            do_resize=True,
            size={"height": self.img_size, "width": self.img_size},
        )

        pixel_values = inputs["pixel_values"].squeeze(0)

        semantic_mask = cv2.resize(
            semantic_mask,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST
        )
        boundary_map = cv2.resize(
            boundary_map,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_NEAREST
        )

        semantic_mask = torch.from_numpy(semantic_mask).long()
        boundary_map = torch.from_numpy(boundary_map).float()

        return {
            "pixel_values": pixel_values,
            "labels": semantic_mask,
            "boundary": boundary_map,
            "image_name": img_name,
        }


# ==========================================
# 4. Model
# ==========================================
class SegFormerInstanceAware(nn.Module):
    def __init__(self, backbone_name="nvidia/mit-b0", num_classes=9):
        super().__init__()
        self.backbone = SegformerModel.from_pretrained(
            backbone_name,
            use_safetensors=True,
        )
        hidden_dim = self.backbone.config.hidden_sizes[-1]

        self.semantic_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)
        self.boundary_head = nn.Conv2d(hidden_dim, 1, kernel_size=1)

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        feat = outputs.last_hidden_state

        semantic_logits = self.semantic_head(feat)
        boundary_logits = self.boundary_head(feat)

        return semantic_logits, boundary_logits


# ==========================================
# 5. Loss
# ==========================================
class InstanceAwareLoss(nn.Module):
    def __init__(
        self,
        ignore_index=255,
        ce_weight=0.5,
        dice_weight=0.5,
        boundary_weight=1.0,
        class_weights=None
    ):
        super().__init__()

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        self.ce = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            weight=class_weights
        )
        self.dice = smp.losses.DiceLoss(mode="multiclass", ignore_index=ignore_index)
        self.bce = nn.BCEWithLogitsLoss()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.boundary_weight = boundary_weight

    def forward(self, semantic_logits, boundary_logits, labels, boundary_targets):
        semantic_logits_up = F.interpolate(
            semantic_logits,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        boundary_logits_up = F.interpolate(
            boundary_logits,
            size=boundary_targets.shape[-2:],
            mode="bilinear",
            align_corners=False
        ).squeeze(1)

        loss_ce = self.ce(semantic_logits_up, labels)
        loss_dice = self.dice(semantic_logits_up, labels)
        loss_boundary = self.bce(boundary_logits_up, boundary_targets)

        total = (
            self.ce_weight * loss_ce +
            self.dice_weight * loss_dice +
            self.boundary_weight * loss_boundary
        )

        return total, {
            "ce": loss_ce.item(),
            "dice": loss_dice.item(),
            "boundary": loss_boundary.item(),
        }


# ==========================================
# 6. Train / Eval
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Train", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        boundary = batch["boundary"].to(device)

        optimizer.zero_grad()

        semantic_logits, boundary_logits = model(pixel_values)
        loss, loss_dict = criterion(semantic_logits, boundary_logits, labels, boundary)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(
            total=f"{loss.item():.4f}",
            ce=f"{loss_dict['ce']:.4f}",
            dice=f"{loss_dict['dice']:.4f}",
            bnd=f"{loss_dict['boundary']:.4f}",
        )

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Val", leave=False)
    for batch in pbar:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        boundary = batch["boundary"].to(device)

        semantic_logits, boundary_logits = model(pixel_values)
        loss, _ = criterion(semantic_logits, boundary_logits, labels, boundary)

        total_loss += loss.item()

    return total_loss / len(loader)


# ==========================================
# 7. Main
# ==========================================
def main():
    print("Using device:", DEVICE)
    print("Classes:", CLASS_NAMES)
    print("Class weights:", SEMANTIC_CLASS_WEIGHTS)

    processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")

    train_ds = LaRSPanopticDataset(
        TRAIN_IMG_DIR,
        TRAIN_PANOPTIC_MASK_DIR,
        TRAIN_PANOPTIC_JSON,
        processor,
        img_size=IMG_SIZE
    )
    val_ds = LaRSPanopticDataset(
        VAL_IMG_DIR,
        VAL_PANOPTIC_MASK_DIR,
        VAL_PANOPTIC_JSON,
        processor,
        img_size=IMG_SIZE
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if DEVICE == "cuda" else False,
    )

    model = SegFormerInstanceAware(
        backbone_name="nvidia/mit-b0",
        num_classes=NUM_CLASSES
    ).to(DEVICE)

    criterion = InstanceAwareLoss(
        ignore_index=IGNORE_INDEX,
        ce_weight=0.5,
        dice_weight=0.5,
        boundary_weight=1.0,
        class_weights=SEMANTIC_CLASS_WEIGHTS,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss = evaluate(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f}")

        ckpt = {
            "model": model.state_dict(),
            "class_names": CLASS_NAMES,
            "img_size": IMG_SIZE,
            "category_id_to_class_name": CATEGORY_ID_TO_CLASS_NAME,
            "semantic_class_weights": SEMANTIC_CLASS_WEIGHTS,
        }

        torch.save(ckpt, LAST_SAVE_PATH)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(ckpt, BEST_SAVE_PATH)
            print(f"Saved best model to: {BEST_SAVE_PATH}")

    print("\nTraining done.")
    print("Best val loss:", best_val_loss)


if __name__ == "__main__":
    main()
