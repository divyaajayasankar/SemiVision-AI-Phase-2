from __future__ import annotations
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance


def load_gray(path, size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_mask(path, size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.NEAREST)
    return (np.asarray(img) > 0).astype(np.uint8)


def apply_perturbation(arr: np.ndarray, condition: str, rng=None) -> np.ndarray:
    rng = np.random.default_rng(42) if rng is None else rng
    x = np.clip(arr, 0, 1)
    if condition == "clean":
        return x
    if condition == "brightness_+0.10":
        return np.clip(x + 0.10, 0, 1)
    if condition == "brightness_-0.10":
        return np.clip(x - 0.10, 0, 1)
    if condition == "contrast_0.70":
        mean = float(x.mean())
        return np.clip((x - mean) * 0.70 + mean, 0, 1)
    if condition == "gamma_1.50":
        return np.clip(np.power(x, 1.50), 0, 1)
    if condition.startswith("noise_"):
        sigma = float(condition.split("_")[-1])
        return np.clip(x + rng.normal(0, sigma, x.shape), 0, 1).astype(np.float32)
    if condition.startswith("blur_"):
        sigma = float(condition.split("_")[-1])
        im = Image.fromarray((x * 255).astype(np.uint8), mode="L")
        return np.asarray(im.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.float32) / 255.0
    raise ValueError(f"Unknown drift condition: {condition}")
