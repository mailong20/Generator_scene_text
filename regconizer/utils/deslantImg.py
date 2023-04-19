""" Created by MrBBS """
# 1/5/2022
# -*-encoding:utf-8-*-

from collections import namedtuple
from typing import Tuple

import cv2
import numpy as np

DeslantRes = namedtuple('DeslantRes', 'img, shear_val, candidates')
Candidate = namedtuple('Candidate', 'shear_val, score')


def _get_shear_vals(lower_bound: float,
                    upper_bound: float,
                    step: float) -> Tuple[float]:
    """Compute shear values in given range."""
    return tuple(np.arange(lower_bound, upper_bound + step, step))


def _shear_img(img: np.ndarray,
               s: float, bg_color: int,
               interpolation=cv2.INTER_NEAREST) -> np.ndarray:
    """Shears image by given shear value."""
    h, w = img.shape
    offset = h * s
    w = w + int(abs(offset))
    tx = max(-offset, 0)

    shear_transform = np.asarray([[1, s, tx], [0, 1, 0]], dtype=float)
    img_sheared = cv2.warpAffine(img, shear_transform, (w, h), flags=interpolation, borderValue=bg_color)

    return img_sheared


def _compute_score(img_binary: np.ndarray, s: float) -> float:
    """Compute score, with higher score values corresponding to more and longer vertical lines."""
    img_sheared = _shear_img(img_binary, s, 0)
    h = img_sheared.shape[0]

    img_sheared_mask = img_sheared > 0
    first_fg_px = np.argmax(img_sheared_mask, axis=0)
    last_fg_px = h - np.argmax(img_sheared_mask[::-1], axis=0)
    num_fg_px = np.sum(img_sheared_mask, axis=0)

    dist_fg_px = last_fg_px - first_fg_px
    col_mask = np.bitwise_and(num_fg_px > 0, dist_fg_px == num_fg_px)
    masked_dist_fg_px = dist_fg_px[col_mask]

    score = sum(masked_dist_fg_px ** 2)
    return score


def deslant_img(img: np.ndarray,
                lower_bound: float = -2,
                upper_bound: float = 2,
                num_steps: int = 20,
                bg_color=255) -> DeslantRes:
    """
    Deslants the image by applying a shear transform.
    The function searches for a shear transform that yields many long connected vertical lines.
    Args:
        img: The image to be deslanted with text in black and background in white.
        lower_bound: Lower bound of shear values to be considered by optimizer.
        upper_bound: Upper bound of shear values to be considered by optimizer.
        num_steps: Number of grid points if optim_algo is 'grid'.
        bg_color: Color that is used to fill the gaps of the returned sheared image.
    Returns:
        Object of DeslantRes, holding the deslanted image and (only for optim_algo 'grid') the candidates
        with shear value and score.
    """
    assert img.ndim == 2
    assert img.dtype == np.uint8
    assert lower_bound < upper_bound

    # apply Otsu's threshold method to inverted input image
    img_binary = cv2.threshold(255 - img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] // 255
    # compute scores on grid points
    step = (upper_bound - lower_bound) / num_steps
    shear_vals = _get_shear_vals(lower_bound, upper_bound, step)
    candidates = [Candidate(s, _compute_score(img_binary, s)) for s in shear_vals]
    best_shear_val = sorted(candidates, key=lambda c: c.score, reverse=True)[0].shear_val

    res_img = _shear_img(img, best_shear_val, bg_color, cv2.INTER_LINEAR)
    return DeslantRes(res_img, best_shear_val, candidates)
