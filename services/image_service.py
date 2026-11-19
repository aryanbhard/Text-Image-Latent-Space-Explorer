"""
Image handling, sample collection management, and preprocessing service.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import io
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image, ImageDraw, ImageFont
from utils.config import SAMPLE_IMAGES_DIR


def validate_image_file(uploaded_file, max_size_mb: float = 5.0) -> Tuple[bool, str, Optional[Image.Image]]:
    """
    Validate uploaded image format and file size.
    Returns:
        (is_valid, message, image_or_none)
    """
    if uploaded_file is None:
        return False, "No file uploaded.", None

    # Check size
    size_bytes = uploaded_file.size
    if size_bytes > max_size_mb * 1024 * 1024:
        return False, f"File size ({size_bytes / (1024*1024):.2f} MB) exceeds maximum allowed limit ({max_size_mb} MB).", None

    try:
        image = Image.open(uploaded_file)
        # Verify image integrity
        image.verify()
        # Re-open because verify() closes the file descriptor
        uploaded_file.seek(0)
        image = Image.open(uploaded_file).convert("RGB")
        return True, "Valid image.", image
    except Exception as e:
        return False, f"Invalid or corrupted image format: {str(e)}", None


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert PIL Image to base64 data string."""
    buffered = io.BytesIO()
    image.save(buffered, format=format)
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{img_str}"


def create_curated_sample_images() -> List[Dict[str, Any]]:
    """
    Generate or load a curated set of clean semantic demo images
    spanning diverse categories: fashion, nature, objects, technology.
    """
    SAMPLE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    categories = [
        # Fashion items (relates directly to Fashion-MNIST domain)
        ("T-Shirt Top", (230, 80, 80), "T-Shirt", "fashion"),
        ("Sneaker Shoe", (60, 160, 240), "Sneaker", "fashion"),
        ("Leather Handbag", (180, 100, 40), "Handbag", "fashion"),
        ("Winter Coat", (50, 70, 90), "Coat", "fashion"),
        # Nature & Environment
        ("Deep Ocean Waves", (20, 80, 160), "Ocean", "nature"),
        ("Lush Forest Tree", (34, 139, 34), "Forest", "nature"),
        ("Golden Sunset", (240, 140, 30), "Sunset", "nature"),
        ("Blooming Sunflower", (255, 215, 0), "Sunflower", "nature"),
        # Objects & Technology
        ("Laptop Computer", (120, 130, 140), "Laptop", "tech"),
        ("Coffee Mug", (140, 70, 20), "Coffee", "objects"),
        ("Soccer Football", (230, 230, 230), "Soccer", "sports"),
        ("Commercial Airplane", (160, 190, 220), "Airplane", "transport"),
    ]

    sample_items = []
    for label, bg_color, short_name, category in categories:
        filename = f"{short_name.lower().replace(' ', '_')}.png"
        filepath = SAMPLE_IMAGES_DIR / filename

        if not filepath.exists():
            # Create a 224x224 synthetic image with geometric features and clear label
            img = Image.new("RGB", (224, 224), color=bg_color)
            draw = ImageDraw.Draw(img)

            # Draw distinguishing visual shapes
            draw.rectangle([20, 20, 204, 204], outline=(255, 255, 255), width=3)
            draw.ellipse([50, 50, 174, 174], outline=(255, 255, 255), width=2)
            draw.rectangle([70, 70, 154, 154], fill=(255, 255, 255, 60))

            # Draw text banner
            draw.rectangle([25, 95, 199, 135], fill=(0, 0, 0))
            draw.text((35, 105), short_name.upper(), fill=(255, 255, 255))

            img.save(filepath, "PNG")

        loaded_img = Image.open(filepath).convert("RGB")
        sample_items.append({
            "name": label,
            "short_name": short_name,
            "category": category,
            "path": filepath,
            "image": loaded_img,
        })

    return sample_items
