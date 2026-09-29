import numpy as np


RGB_TO_YUV = np.array([
    [0.299, 0.587, 0.114],
    [-0.14713, -0.28886, 0.436],
    [0.615, -0.51499, -0.10001],
], dtype=np.float64)

YUV_TO_RGB = np.linalg.inv(RGB_TO_YUV)
YUV_NOISE_SCALES = np.linalg.norm(RGB_TO_YUV, axis=1)


def _validate_color_image(image, name):
    image = np.asarray(image, dtype=np.float64)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"{name} must have shape (height, width, 3)")
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(image).all():
        raise ValueError(f"{name} must contain only finite values")
    return image


def rgb_to_yuv(rgb):
    """Convert a channel-last RGB image to signed, full-resolution YUV."""
    rgb = _validate_color_image(rgb, "rgb")
    if np.min(rgb) < 0. or np.max(rgb) > 255.:
        raise ValueError("rgb values must be in the range [0, 255]")
    return rgb @ RGB_TO_YUV.T


def yuv_to_rgb(yuv):
    """Convert a channel-last, full-resolution YUV image back to RGB."""
    yuv = _validate_color_image(yuv, "yuv")
    return yuv @ YUV_TO_RGB.T


def rgb_sigma_to_yuv(sigma):
    """Return marginal Y/U/V sigmas for independent equal-variance RGB noise."""
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.:
        raise ValueError("sigma must be a finite positive number")
    return sigma * YUV_NOISE_SCALES
