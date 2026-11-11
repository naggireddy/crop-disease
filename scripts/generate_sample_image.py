"""
Sample Diagnostic Image Generator for Pipeline Testing
Generates a synthetic plant leaf image with simulated lesion characteristics
to test leaf detection, quality gate validation, and inference mechanics
without fabricating benchmark datasets.
"""

import cv2
import numpy as np
from pathlib import Path

def generate_test_leaf_image(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    size = 400
    # Background: neutral light gray desk/table
    img = np.ones((size, size, 3), dtype=np.uint8) * 190

    # Leaf shape (ellipse rotated)
    center = (200, 200)
    axes = (140, 75)
    angle = 35

    # Draw vibrant green leaf blade
    cv2.ellipse(img, center, axes, angle, 0, 360, (38, 145, 45), -1)

    # Leaf texture and secondary tone
    overlay = img.copy()
    cv2.ellipse(overlay, center, (120, 60), angle, 0, 360, (50, 175, 60), -1)
    img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)

    # Main vein (midrib)
    cv2.line(img, (90, 120), (310, 280), (30, 110, 35), 3)

    # Side veins
    cv2.line(img, (140, 155), (130, 200), (35, 125, 40), 1)
    cv2.line(img, (180, 185), (200, 150), (35, 125, 40), 1)
    cv2.line(img, (220, 215), (210, 260), (35, 125, 40), 1)
    cv2.line(img, (260, 245), (280, 210), (35, 125, 40), 1)

    # Simulated fungal necrotic lesion (concentric brown rings - Early Blight style)
    lesion_center = (210, 175)
    cv2.circle(img, lesion_center, 22, (20, 60, 95), -1)   # dark necrotic brown
    cv2.circle(img, lesion_center, 14, (15, 45, 75), -1)   # inner ring
    cv2.circle(img, lesion_center, 6, (10, 30, 55), -1)    # center
    # Chlorotic yellow halo around lesion
    halo_overlay = img.copy()
    cv2.circle(halo_overlay, lesion_center, 30, (50, 200, 220), -1)
    img = cv2.addWeighted(halo_overlay, 0.25, img, 0.75, 0)

    # Add realistic photographic camera sensor noise
    noise = np.random.normal(0, 4, img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    cv2.imwrite(str(output_path), noisy_img)
    print(f"[+] Synthetic diagnostic test image saved to: {output_path}")

if __name__ == "__main__":
    out_file = Path(__file__).resolve().parents[1] / "data" / "sample_test_leaf.jpg"
    generate_test_leaf_image(out_file)

