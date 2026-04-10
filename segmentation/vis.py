import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# ==========================================
# 1. 配置路径与参数
# ==========================================
IMG_DIR = 'lars_v1.0.0_images/val/images'   # 验证集图片文件夹
OUTPUT_DIR = 'vis_results_segformer_pred'   # 结果保存路径
MODEL_PATH = 'segformer_baseline.pth'       # 训练好的 SegFormer 权重
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 颜色映射
# 0: obstacle -> 红
# 1: water    -> 蓝
# 2: sky      -> 绿
COLOR_MAP = {
    0: [255, 0, 0],
    1: [0, 0, 255],
    2: [0, 255, 0],
    255: [0, 0, 0]
}

def label_to_color(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in COLOR_MAP.items():
        color_mask[mask == label] = color
    return color_mask

# ==========================================
# 2. 加载模型与 Processor
# ==========================================
processor = SegformerImageProcessor.from_pretrained("nvidia/mit-b0")

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/mit-b0",
    num_labels=3,
    ignore_mismatched_sizes=True
)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# 如果你的 pth 直接就是 state_dict，用这一句
model.load_state_dict(checkpoint)

# 如果上面报错，再改成这一句：
# model.load_state_dict(checkpoint['model_state_dict'])

model.to(DEVICE)
model.eval()

# ==========================================
# 3. 遍历整个验证集
# ==========================================
image_files = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(('.jpg', '.png'))])

print(f"开始生成可视化结果，共 {len(image_files)} 张图片...")

for filename in tqdm(image_files):
    img_path = os.path.join(IMG_DIR, filename)

    # 读取原图
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        print(f"Warning: failed to read {img_path}")
        continue

    image_src = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_src.shape[:2]

    # ==========================================
    # 4. 模型推理
    # ==========================================
    inputs = processor(images=image_src, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # 上采样回原图尺寸
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=(h, w),
            mode='bilinear',
            align_corners=False
        )

        pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # ==========================================
    # 5. 生成彩色 prediction 和 overlay
    # ==========================================
    color_pred = label_to_color(pred_mask)
    overlay_pred = cv2.addWeighted(image_src, 0.7, color_pred, 0.3, 0)

    # ==========================================
    # 6. 3-panel 可视化
    # ==========================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    titles = ["1. Input RGB", "2. Prediction Mask", "3. Prediction Overlay"]
    images = [image_src, color_pred, overlay_pred]

    for i in range(3):
        axes[i].imshow(images[i])
        axes[i].set_title(titles[i], fontsize=14, pad=10, fontweight='bold')
        axes[i].axis('off')

    plt.tight_layout()

    # 保存结果
    save_path = os.path.join(OUTPUT_DIR, f"vis_segformer_{os.path.splitext(filename)[0]}.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=200)
    plt.close(fig)

print(f"Saved to: {OUTPUT_DIR}")
