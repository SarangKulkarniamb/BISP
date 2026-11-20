"""Basic image-quality checks; these heuristics are not clinically validated."""

from dataclasses import asdict, dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class QualityResult:
    accepted: bool
    reasons: list[str]
    width: int
    height: int
    mean_brightness: float
    sharpness: float

    def to_dict(self) -> dict:
        return asdict(self)


def assess_image(image: Image.Image, min_dimension: int = 160,
                 min_brightness: float = 25.0, max_brightness: float = 245.0,
                 min_sharpness: float = 20.0) -> QualityResult:
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    reasons = []
    if min(width, height) < min_dimension:
        reasons.append("image_too_small")
    if brightness < min_brightness:
        reasons.append("too_dark")
    elif brightness > max_brightness:
        reasons.append("too_bright")
    if sharpness < min_sharpness:
        reasons.append("possibly_blurry")
    return QualityResult(not reasons, reasons, width, height, brightness, sharpness)
