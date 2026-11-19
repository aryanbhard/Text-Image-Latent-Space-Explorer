"""
Prepare initial demo data: curated images and sample texts.
Text & Image Latent-Space Explorer (Project P6 - Roll Number: 06)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.image_service import create_curated_sample_images
from utils.config import DEFAULT_TEXT_SAMPLES, SAMPLE_IMAGES_DIR


def main():
    print("[Demo Data] Preparing curated sample images...")
    images = create_curated_sample_images()
    print(f"[Demo Data] Created/Verified {len(images)} sample images in {SAMPLE_IMAGES_DIR}")
    print(f"[Demo Data] Verified {len(DEFAULT_TEXT_SAMPLES)} default semantic sentences across 5 domains.")


if __name__ == "__main__":
    main()
