"""
Image Preprocessing and Augmentation Pipeline for Strawberry AI.

Choice Rationale:
- ViT models require normalized input matching ImageNet statistics:
  mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225].
- Image dimensions resized to 224x224 to match standard 16x16 patch ViT grids (14x14 = 196 patches).
- Training augmentations: RandomHorizontalFlip, RandomRotation(15), ColorJitter(brightness=0.1, contrast=0.1)
  to prevent overfitting on subtle pathology leaf/fruit differences without distorting characteristic lesions.
- Validation/Test transforms: Strict deterministic Resize((224, 224)) and Normalize.
"""

from typing import Tuple, Dict, Optional
from PIL import Image
import numpy as np
import torchvision.transforms as T

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 224) -> T.Compose:
    """Data augmentation pipeline for training."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_transforms(image_size: int = 224) -> T.Compose:
    """Deterministic preprocessing for validation and test evaluation."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_inference_transforms(image_size: int = 224) -> T.Compose:
    """Preprocessing for single-image web app and CLI inference."""
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_image_safely(image_path: str) -> Image.Image:
    """Open image file, ensure RGB format, and close file descriptor cleanly."""
    with Image.open(image_path) as img:
        return img.convert("RGB")


def remove_strawberry_leaves(
    pil_image: Image.Image,
    background_mode: str = "white",
    dilation_kernel_size: int = 5,
) -> Tuple[Image.Image, np.ndarray, Dict]:
    """
    Isolate the strawberry fruit flesh by detecting and removing the leafy calyx, stems, and foliage.
    
    Args:
        pil_image: Input PIL Image
        background_mode: 'white' (default), 'neutral' (#f0f0f0), 'blur', or 'black'
        dilation_kernel_size: Kernel size for morphological dilation to cover fine leaf serrations
        
    Returns:
        fruit_only_image: PIL Image with leaves/calyx removed
        leaf_mask: Binary NumPy array (H, W) where 255 indicates leaf pixels
        stats: Dictionary containing leaf area percentage, fruit area percentage, etc.
    """
    rgb = np.array(pil_image.convert("RGB"))
    h, w = rgb.shape[:2]

    if not HAS_CV2:
        # Fallback if cv2 is not available
        return pil_image, np.zeros((h, w), dtype=np.uint8), {"leaf_area_pct": 0.0, "leaves_detected": False}

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    # 1. Detect Green Calyx / Foliage in HSV
    lower_green = np.array([25, 30, 30])
    upper_green = np.array([92, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # 2. Check Green Dominance in RGB
    r = rgb[:, :, 0].astype(np.int32)
    g = rgb[:, :, 1].astype(np.int32)
    b = rgb[:, :, 2].astype(np.int32)
    green_dom = (g > r * 0.95) & (g > b * 0.95) & (g > 35)

    # 3. Protect Red Strawberry Fruit Flesh from accidental masking
    # Red strawberries have strong red channel dominance
    red_fruit_mask = (r > g * 1.2) & (r > b * 1.2) & (r > 60)

    # Combine green leaf masks and exclude protected red fruit flesh
    combined_leaf_mask = (green_mask > 0) | green_dom
    combined_leaf_mask[red_fruit_mask] = False
    combined_leaf_mask = combined_leaf_mask.astype(np.uint8) * 255

    # 4. Morphological dilation to smoothly enclose serrated leaf margins and small stems
    k_size = max(3, dilation_kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    dilated_leaf_mask = cv2.dilate(combined_leaf_mask, kernel, iterations=2)
    # Re-apply red fruit protection so berry flesh is never eaten away
    dilated_leaf_mask[red_fruit_mask] = 0

    # 5. Apply chosen background replacement
    fruit_only = rgb.copy()
    if background_mode == "white":
        bg_val = np.array([255, 255, 255], dtype=np.uint8)
        fruit_only[dilated_leaf_mask > 0] = bg_val
    elif background_mode == "neutral":
        bg_val = np.array([242, 242, 244], dtype=np.uint8)
        fruit_only[dilated_leaf_mask > 0] = bg_val
    elif background_mode == "black":
        bg_val = np.array([0, 0, 0], dtype=np.uint8)
        fruit_only[dilated_leaf_mask > 0] = bg_val
    elif background_mode == "blur":
        blurred_bg = cv2.GaussianBlur(rgb, (51, 51), 0)
        fruit_only[dilated_leaf_mask > 0] = blurred_bg[dilated_leaf_mask > 0]
    else:
        fruit_only[dilated_leaf_mask > 0] = np.array([255, 255, 255], dtype=np.uint8)

    # 6. Compute statistics
    total_pixels = h * w
    leaf_pixel_count = int(np.sum(dilated_leaf_mask > 0))
    leaf_area_pct = round(float(leaf_pixel_count / max(total_pixels, 1) * 100), 2)
    fruit_pixel_count = int(np.sum(red_fruit_mask))
    fruit_area_pct = round(float(fruit_pixel_count / max(total_pixels, 1) * 100), 2)

    stats = {
        "leaf_area_pct": leaf_area_pct,
        "fruit_area_pct": fruit_area_pct,
        "leaves_detected": leaf_area_pct > 1.5,
        "leaf_pixels": leaf_pixel_count,
    }

    return Image.fromarray(fruit_only), dilated_leaf_mask, stats

