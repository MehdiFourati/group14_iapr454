"""
Classical computer-vision UNO card classifier.

Usage
-----
from uno_card_classifier import UnoCardClassifier

clf = UnoCardClassifier(reference_dir=r"C:/path/to/reference_images")
label = clf.classify(cropped_card_bgr)              # e.g. "r_5", "b_skip", "wild_draw4"
details = clf.classify(cropped_card_bgr, return_details=True)

Expected reference images inside reference_dir
---------------------------------------------
L1000765.jpg  -> 1, 2, 3 cards
L1000766.jpg  -> 4, 5, 6 cards
L1000767.jpg  -> 7, 8, 9 + wild cards
L1000768.jpg  -> 0 + reverse/skip/+2 cards

Input assumption
----------------
The input to classify(...) should be a crop containing one full UNO card.
It can be BGR OpenCV image, RGB image, grayscale image, or path to an image.
The crop should be reasonably tight around the card. Perspective-corrected crops work best.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np

ImageInput = Union[str, Path, np.ndarray]


# -----------------------------------------------------------------------------
# Global constants
# -----------------------------------------------------------------------------

CARD_W = 160
CARD_H = 250

CENTER_DIGIT_ROI = (0.25, 0.23, 0.75, 0.77)
ACTION_ROI = (0.08, 0.08, 0.92, 0.92)
BLACK_SYMBOL_ROI = (0.12, 0.12, 0.88, 0.88)

TEMPLATE_SIZE = (64, 96)          # width, height
ACTION_TEMPLATE_SIZE = (96, 96)   # width, height

ORIGINAL_W = 2048
ORIGINAL_H = 1362

REF_FILENAMES = {
    "1_2_3": "L1000765.jpg",
    "4_5_6": "L1000766.jpg",
    "7_8_9": "L1000767.jpg",
    "0_actions": "L1000768.jpg",
}

# Manual central-digit crops, written for 2048x1362 images.
# They are automatically scaled if your local reference images have another size.
TEMPLATE_CROPS_ORIGINAL = [
    ("0", "0_actions", 1030, 910, 120, 170),
    ("1", "1_2_3",     700, 1030, 130, 160),
    ("2", "1_2_3",     700,  730, 120, 140),
    ("3", "1_2_3",     680,  410, 110, 150),
    ("4", "4_5_6",     875, 1035, 130, 160),
    ("5", "4_5_6",     850,  735, 130, 140),
    ("6", "4_5_6",     835,  410, 120, 150),
    ("7", "7_8_9",    1000, 1050, 130, 170),
    ("8", "7_8_9",     965,  760, 140, 140),
    ("9", "7_8_9",     980,  430, 120, 150),
]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def load_bgr(image_input: ImageInput, input_color: str = "BGR") -> np.ndarray:
    """
    Load/convert an image to BGR uint8.

    Parameters
    ----------
    image_input:
        Path or numpy image.
    input_color:
        Only used when image_input is a numpy array.
        Use "BGR" for OpenCV images, "RGB" for PIL/matplotlib images, "GRAY" for grayscale.
    """
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_input}")
        return img

    img = np.asarray(image_input)

    if img.ndim == 2:
        return cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    if img.ndim != 3 or img.shape[2] not in (3, 4):
        raise ValueError("image_input must be a path, grayscale image, RGB image, or BGR image.")

    if img.shape[2] == 4:
        img = img[:, :, :3]

    img = img.astype(np.uint8)

    if input_color.upper() == "RGB":
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if input_color.upper() == "BGR":
        return img.copy()

    raise ValueError("input_color must be 'BGR', 'RGB', or 'GRAY'.")


def crop_relative(img: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
    """Crop using relative coordinates: (x0, y0, x1, y1)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi

    x0_i = int(round(x0 * w))
    y0_i = int(round(y0 * h))
    x1_i = int(round(x1 * w))
    y1_i = int(round(y1 * h))

    return img[y0_i:y1_i, x0_i:x1_i].copy()


def crop_card_roi(card: np.ndarray, roi: Tuple[float, float, float, float]) -> np.ndarray:
    """Alias for crop_relative, used for readability."""
    return crop_relative(card, roi)


def crop_center_digit(card: np.ndarray) -> np.ndarray:
    """Crop the central UNO digit/symbol region from a normalized card image."""
    return crop_relative(card, CENTER_DIGIT_ROI)


def rotate_patch_keep_size(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate an image while keeping the same output size."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    rotated = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated


def normalize_card_crop(card: np.ndarray) -> np.ndarray:
    """
    Normalize a cropped UNO card to portrait CARD_W x CARD_H.

    If the crop is landscape, rotate it once to portrait. A possible 180-degree
    ambiguity is handled later by rotated templates and the 6/9 underline logic.
    """
    if card.ndim == 2:
        card = cv2.cvtColor(card, cv2.COLOR_GRAY2BGR)

    h, w = card.shape[:2]

    if w > h:
        card = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)

    card = cv2.resize(card, (CARD_W, CARD_H), interpolation=cv2.INTER_AREA)
    return card


# -----------------------------------------------------------------------------
# Perspective helpers used only to build reference action templates
# -----------------------------------------------------------------------------

def expand_rotated_rect(rect: tuple, scale: float = 1.03) -> tuple:
    """Slightly enlarge a cv2.minAreaRect rectangle."""
    (cx, cy), (w, h), angle = rect
    return ((cx, cy), (w * scale, h * scale), angle)


def order_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1).reshape(-1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_warp(img: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Warp a rotated card to a top-down rectangular crop."""
    rect = order_points(pts)
    tl, tr, br, bl = rect

    width_bottom = np.linalg.norm(br - bl)
    width_top = np.linalg.norm(tr - tl)
    max_width = max(int(round(max(width_bottom, width_top))), 1)

    height_right = np.linalg.norm(tr - br)
    height_left = np.linalg.norm(tl - bl)
    max_height = max(int(round(max(height_right, height_left))), 1)

    dst = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, matrix, (max_width, max_height))
    return warped


def edge_mask_for_cards(img: np.ndarray, canny_low: int = 30, canny_high: int = 90) -> np.ndarray:
    """Detect physical card borders using edges/shadows."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

    edges = cv2.Canny(blur, canny_low, canny_high)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=2,
    )
    return edges


def detect_card_candidates(
    img: np.ndarray,
    canny_low: int = 30,
    canny_high: int = 90,
    min_area_frac: float = 0.004,
    max_area_frac: float = 0.04,
    aspect_min: float = 1.25,
    aspect_max: float = 1.95,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Detect full-card candidates in a reference image."""
    h, w = img.shape[:2]
    image_area = h * w
    mask = edge_mask_for_cards(img, canny_low=canny_low, canny_high=canny_high)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: List[Dict[str, Any]] = []

    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), _ = rect

        if rw < 1 or rh < 1:
            continue

        rect_area = rw * rh
        aspect = max(rw, rh) / max(min(rw, rh), 1)

        if not (min_area_frac * image_area <= rect_area <= max_area_frac * image_area):
            continue

        if not (aspect_min <= aspect <= aspect_max):
            continue

        rect_expanded = expand_rotated_rect(rect, scale=1.03)
        box_expanded = cv2.boxPoints(rect_expanded).astype(np.float32)
        warped = four_point_warp(img, box_expanded)
        warped = normalize_card_crop(warped)

        detections.append(
            {
                "center": (float(cx), float(cy)),
                "area": float(rect_area),
                "aspect": float(aspect),
                "box": box_expanded,
                "card": warped,
            }
        )

    detections = remove_duplicate_detections(detections)
    detections = sorted(detections, key=lambda d: (d["center"][1], d["center"][0]))
    return detections, mask


def remove_duplicate_detections(detections: List[Dict[str, Any]], min_center_dist: float = 80) -> List[Dict[str, Any]]:
    """Remove duplicate detections of the same physical card."""
    detections_sorted = sorted(detections, key=lambda d: d.get("area", 0.0), reverse=True)
    kept: List[Dict[str, Any]] = []

    for det in detections_sorted:
        cx, cy = det["center"]
        is_duplicate = False

        for kept_det in kept:
            kx, ky = kept_det["center"]
            if np.sqrt((cx - kx) ** 2 + (cy - ky) ** 2) < min_center_dist:
                is_duplicate = True
                break

        if not is_duplicate:
            kept.append(det)

    return sorted(kept, key=lambda d: (d["center"][1], d["center"][0]))


# -----------------------------------------------------------------------------
# Binary feature extraction
# -----------------------------------------------------------------------------

def digit_binary(
    patch: np.ndarray,
    crop_to_content: bool = True,
    out_size: Tuple[int, int] = TEMPLATE_SIZE,
) -> np.ndarray:
    """
    Extract a clean binary mask for an UNO central digit.

    The mask keeps the central colored/dark digit and removes most border/background.
    Output is white foreground on black background.
    """
    if patch.ndim == 2:
        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
    else:
        patch_bgr = patch

    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    _, s, v = cv2.split(hsv)

    colored = ((s > 55) & (v > 70)).astype(np.uint8) * 255
    dark = (v < 95).astype(np.uint8) * 255
    mask = cv2.bitwise_or(colored, dark)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    h, w = mask.shape[:2]
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask)
    cx_ref = w / 2
    cy_ref = h / 2
    candidates = []

    for i in range(1, num):
        x, y, cw, ch, area = stats[i]
        cx, cy = centroids[i]

        if area < 80:
            continue

        touches_border = x <= 2 or y <= 2 or x + cw >= w - 2 or y + ch >= h - 2
        if touches_border:
            continue

        dist = np.sqrt((cx - cx_ref) ** 2 + (cy - cy_ref) ** 2)
        score = area - 2.0 * dist
        candidates.append((score, i))

    if candidates:
        _, best_id = max(candidates, key=lambda item: item[0])
        cleaned[labels == best_id] = 255
    else:
        cleaned = mask

    mask = cv2.morphologyEx(
        cleaned,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )

    if crop_to_content:
        ys, xs = np.where(mask > 0)
        if len(xs) > 0 and len(ys) > 0:
            pad = 8
            x0 = max(0, xs.min() - pad)
            x1 = min(mask.shape[1], xs.max() + pad + 1)
            y0 = max(0, ys.min() - pad)
            y1 = min(mask.shape[0], ys.max() + pad + 1)
            mask = mask[y0:y1, x0:x1]

    mask = cv2.resize(mask, out_size, interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8) * 255


def action_edge_binary(
    patch: np.ndarray,
    out_size: Tuple[int, int] = ACTION_TEMPLATE_SIZE,
) -> np.ndarray:
    """Extract action-card symbols using edges."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(gray_eq, (3, 3), 0)

    edges = cv2.Canny(blur, 40, 120)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
    edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    ys, xs = np.where(edges > 0)
    if len(xs) > 0:
        pad = 6
        x0 = max(0, xs.min() - pad)
        x1 = min(edges.shape[1], xs.max() + pad + 1)
        y0 = max(0, ys.min() - pad)
        y1 = min(edges.shape[0], ys.max() + pad + 1)
        edges = edges[y0:y1, x0:x1]

    edges = cv2.resize(edges, out_size, interpolation=cv2.INTER_NEAREST)
    return (edges > 0).astype(np.uint8) * 255


# -----------------------------------------------------------------------------
# Classifier
# -----------------------------------------------------------------------------

class UnoCardClassifier:
    """Classical CV UNO classifier for one cropped card at a time."""

    def __init__(
        self,
        reference_dir: Union[str, Path],
        action_min_score: float = 0.34,
        black_threshold: float = 0.13,
    ) -> None:
        self.reference_dir = Path(reference_dir)
        self.action_min_score = action_min_score
        self.black_threshold = black_threshold

        self.ref_images = self._load_reference_images()
        self.raw_template_crops = self._build_raw_digit_template_crops()
        self.digit_templates = self._build_digit_templates()
        self.upright_6_9_templates = self._build_upright_6_9_templates()
        self.colored_action_templates = self._build_colored_action_templates_from_reference()

    def classify(
        self,
        cropped_card: ImageInput,
        input_color: str = "BGR",
        return_details: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """
        Classify one cropped UNO card.

        Returns a label by default:
            r_5, b_skip, y_reverse, g_+2, wild, wild_draw4

        If return_details=True, returns a dictionary with label, color, symbol,
        kind, score, and intermediate candidates.
        """
        card = load_bgr(cropped_card, input_color=input_color)
        card = normalize_card_crop(card)
        result = self._classify_uno_card(card)

        if return_details:
            result = dict(result)
            result["card"] = card
            return result

        return result["label"]

    def classify_many(
        self,
        cropped_cards: List[ImageInput],
        input_color: str = "BGR",
        return_details: bool = False,
    ) -> List[Union[str, Dict[str, Any]]]:
        """Classify a list of cropped UNO cards."""
        return [self.classify(card, input_color=input_color, return_details=return_details) for card in cropped_cards]

    # ------------------------------------------------------------------
    # Reference/template building
    # ------------------------------------------------------------------

    def _load_reference_images(self) -> Dict[str, np.ndarray]:
        ref_images = {}
        missing = []

        for key, filename in REF_FILENAMES.items():
            path = self.reference_dir / filename
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                missing.append(str(path))
            else:
                ref_images[key] = img

        if missing:
            raise FileNotFoundError(
                "Missing reference image(s):\n" + "\n".join(missing)
            )

        return ref_images

    def _build_raw_digit_template_crops(self) -> List[Tuple[str, np.ndarray]]:
        crops: List[Tuple[str, np.ndarray]] = []

        for label, image_key, x, y, w, h in TEMPLATE_CROPS_ORIGINAL:
            img = self.ref_images[image_key]
            img_h, img_w = img.shape[:2]

            sx = img_w / ORIGINAL_W
            sy = img_h / ORIGINAL_H

            x_s = int(round(x * sx))
            y_s = int(round(y * sy))
            w_s = int(round(w * sx))
            h_s = int(round(h * sy))

            patch = img[y_s:y_s + h_s, x_s:x_s + w_s].copy()
            crops.append((label, patch))

        return crops

    def _build_digit_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        templates: Dict[str, List[Dict[str, Any]]] = {str(i): [] for i in range(10)}

        for label, patch in self.raw_template_crops:
            base = digit_binary(patch)

            for angle in [0, 180, -8, 8]:
                rotated = rotate_patch_keep_size(base, angle)
                templates[label].append({"angle": angle, "image": rotated})

        return templates

    def _build_upright_6_9_templates(self) -> Dict[str, List[Dict[str, Any]]]:
        upright_templates: Dict[str, List[Dict[str, Any]]] = {"6": [], "9": []}

        for label, patch in self.raw_template_crops:
            if label not in ["6", "9"]:
                continue

            normalized_patch, _ = self._normalize_digit_crop_using_underline(patch)
            base = digit_binary(normalized_patch)

            for angle in [0, -8, 8]:
                rotated = rotate_patch_keep_size(base, angle)
                upright_templates[label].append({"angle": angle, "image": rotated})

        return upright_templates

    def _build_colored_action_templates_from_reference(self) -> Dict[str, List[Dict[str, Any]]]:
        """Build +2, skip, reverse templates from the 0_actions reference image."""
        img = self.ref_images["0_actions"]
        img_h, img_w = img.shape[:2]
        detections, _ = detect_card_candidates(img)

        groups: Dict[str, List[Dict[str, Any]]] = {
            "+2": [],
            "skip": [],
            "reverse": [],
        }

        for det in detections:
            cx, cy = det["center"]

            # +2 cards are bottom-left.
            if cx < 0.42 * img_w and cy > 0.58 * img_h:
                groups["+2"].append(det)

            # Reverse cards are upper-right.
            elif cx > 0.42 * img_w and cy < 0.42 * img_h:
                groups["reverse"].append(det)

            # Skip cards are middle-right.
            elif cx > 0.42 * img_w and 0.42 * img_h <= cy <= 0.70 * img_h:
                groups["skip"].append(det)

        action_templates: Dict[str, List[Dict[str, Any]]] = {}

        for label, dets in groups.items():
            if not dets:
                raise RuntimeError(f"No reference examples found for action label: {label}")

            action_templates[label] = []

            for det in dets:
                crop = crop_card_roi(det["card"], ACTION_ROI)
                base = action_edge_binary(crop)

                for angle in [0, 180, -10, 10, 170, 190]:
                    rotated = rotate_patch_keep_size(base, angle)
                    action_templates[label].append({"angle": angle, "image": rotated})

        return action_templates

    # ------------------------------------------------------------------
    # Color / black-card logic
    # ------------------------------------------------------------------

    def _black_card_score(self, card: np.ndarray) -> float:
        hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)
        black_mask = (v < 75).astype(np.uint8)
        return float(black_mask.mean())

    def _is_black_special_card(self, card: np.ndarray) -> Tuple[bool, float]:
        ratio = self._black_card_score(card)
        return ratio > self.black_threshold, ratio

    def _classify_card_color(self, card: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        is_black, black_ratio = self._is_black_special_card(card)

        if is_black:
            return "black", {"black_ratio": black_ratio}

        hsv = cv2.cvtColor(card, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        color_mask = (s > 60) & (v > 80)
        hue = h[color_mask]

        if len(hue) == 0:
            return "unknown", {"black_ratio": black_ratio, "counts": {}}

        counts = {
            "r": int(np.sum((hue < 10) | (hue > 165))),
            "y": int(np.sum((hue >= 18) & (hue <= 40))),
            "g": int(np.sum((hue >= 45) & (hue <= 85))),
            "b": int(np.sum((hue >= 90) & (hue <= 115))),
        }

        color = max(counts, key=counts.get)
        return color, {"black_ratio": black_ratio, "counts": counts}

    def _classify_black_special(self, card: np.ndarray) -> Dict[str, Any]:
        """Distinguish wild vs wild_draw4 using colored component structure."""
        crop = crop_card_roi(card, BLACK_SYMBOL_ROI)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        _, s, v = cv2.split(hsv)

        colored = ((s > 55) & (v > 70)).astype(np.uint8) * 255
        colored = cv2.morphologyEx(
            colored,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )
        colored = cv2.morphologyEx(
            colored,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            iterations=1,
        )

        colored_ratio = float((colored > 0).mean())
        num, _, stats, _ = cv2.connectedComponentsWithStats(colored, connectivity=8)

        component_areas = []
        for i in range(1, num):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area > 40:
                component_areas.append(area)

        component_count = len(component_areas)
        largest_component_ratio = 0.0

        if component_areas:
            largest_component_ratio = float(max(component_areas) / max(sum(component_areas), 1))

        if component_count >= 3 and largest_component_ratio < 0.75:
            label = "wild_draw4"
        else:
            label = "wild"

        return {
            "label": label,
            "score": float(max(colored_ratio, largest_component_ratio)),
            "colored_ratio": colored_ratio,
            "component_count": component_count,
            "largest_component_ratio": largest_component_ratio,
        }

    # ------------------------------------------------------------------
    # Template matching
    # ------------------------------------------------------------------

    def _classify_digit_crop(self, digit_crop: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        query = digit_binary(digit_crop)
        scores = []

        for label, variants in self.digit_templates.items():
            for variant in variants:
                score = cv2.matchTemplate(query, variant["image"], cv2.TM_CCOEFF_NORMED)[0, 0]
                scores.append((float(score), label, variant["angle"]))

        scores = sorted(scores, reverse=True)
        return {
            "label": scores[0][1],
            "score": scores[0][0],
            "angle": scores[0][2],
            "top": scores[:top_k],
            "query_binary": query,
        }

    def _classify_colored_action_crop(self, action_crop: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        query = action_edge_binary(action_crop)
        scores = []

        for label, variants in self.colored_action_templates.items():
            for variant in variants:
                score = cv2.matchTemplate(query, variant["image"], cv2.TM_CCOEFF_NORMED)[0, 0]
                scores.append((float(score), label, variant["angle"]))

        scores = sorted(scores, reverse=True)
        return {
            "label": scores[0][1],
            "score": scores[0][0],
            "angle": scores[0][2],
            "top": scores[:top_k],
            "query_binary": query,
        }

    # ------------------------------------------------------------------
    # 6/9 underline logic
    # ------------------------------------------------------------------

    def _detect_underline_position(self, digit_crop: np.ndarray) -> Dict[str, Any]:
        """Detect whether UNO 6/9 underline is above or below the digit."""
        crop = cv2.resize(digit_crop, (128, 192), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(hsv)

        dark = (v < 120).astype(np.uint8) * 255
        dark = cv2.morphologyEx(
            dark,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

        h, w = dark.shape[:2]
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(0.20 * w), 3))
        horiz = cv2.morphologyEx(dark, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

        num, _, stats, centroids = cv2.connectedComponentsWithStats(horiz, connectivity=8)
        candidates = []

        for i in range(1, num):
            x, y, cw, ch, area = stats[i]
            cx, cy = centroids[i]
            aspect = cw / max(ch, 1)
            near_top_or_bottom = cy < 0.40 * h or cy > 0.60 * h

            if cw >= 0.18 * w and aspect >= 3.5 and area >= 20 and near_top_or_bottom:
                candidates.append({"x": x, "y": y, "w": cw, "h": ch, "area": area, "cx": cx, "cy": cy})

        if not candidates:
            return {"found": False, "position": None, "line_y": None, "body_center_y": None}

        best = max(candidates, key=lambda c: c["w"] * c["area"])
        line_y = float(best["cy"])

        body = dark.copy()
        body[horiz > 0] = 0
        ys, _ = np.where(body > 0)

        if len(ys) == 0:
            ys, _ = np.where(dark > 0)

        body_center_y = float((ys.min() + ys.max()) / 2)
        position = "above" if line_y < body_center_y else "below"

        return {
            "found": True,
            "position": position,
            "line_y": line_y,
            "body_center_y": body_center_y,
        }

    def _normalize_digit_crop_using_underline(self, digit_crop: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """If underline is above the digit, rotate by 180 degrees."""
        underline = self._detect_underline_position(digit_crop)
        normalized = digit_crop.copy()
        rotated = False

        if underline["found"] and underline["position"] == "above":
            normalized = cv2.rotate(normalized, cv2.ROTATE_180)
            rotated = True

        return normalized, {"rotated": rotated, "underline": underline}

    def _classify_6_9_with_underline(self, digit_crop: np.ndarray, top_k: int = 5) -> Dict[str, Any]:
        """Classify 6 vs 9 after orientation normalization using underline."""
        normalized_crop, norm_info = self._normalize_digit_crop_using_underline(digit_crop)
        query = digit_binary(normalized_crop)
        scores = []

        for label, variants in self.upright_6_9_templates.items():
            for variant in variants:
                score = cv2.matchTemplate(query, variant["image"], cv2.TM_CCOEFF_NORMED)[0, 0]
                scores.append((float(score), label, variant["angle"]))

        scores = sorted(scores, reverse=True)
        return {
            "label": scores[0][1],
            "score": scores[0][0],
            "angle": scores[0][2],
            "top": scores[:top_k],
            "norm_info": norm_info,
            "query_binary": query,
        }

    def _classify_card_number_with_underline_fix(self, card: np.ndarray) -> Dict[str, Any]:
        """Classify number card; use underline-normalized logic for 6/9."""
        digit_crop = crop_center_digit(card)
        normal_result = self._classify_digit_crop(digit_crop)

        final_label = normal_result["label"]
        final_score = normal_result["score"]
        six_nine_result = None

        if normal_result["label"] in ["6", "9"]:
            six_nine_result = self._classify_6_9_with_underline(digit_crop)
            if six_nine_result["score"] >= 0.20:
                final_label = six_nine_result["label"]
                final_score = six_nine_result["score"]

        return {
            "label": final_label,
            "score": final_score,
            "raw_label": normal_result["label"],
            "raw_score": normal_result["score"],
            "center_result": normal_result,
            "six_nine_result": six_nine_result,
            "digit_crop": digit_crop,
        }

    # ------------------------------------------------------------------
    # Final card classification
    # ------------------------------------------------------------------

    def _classify_uno_card(self, card: np.ndarray) -> Dict[str, Any]:
        color, color_info = self._classify_card_color(card)

        # Black cards: wild or wild_draw4.
        if color == "black":
            black_result = self._classify_black_special(card)
            return {
                "label": black_result["label"],
                "color": "black",
                "symbol": black_result["label"],
                "score": black_result["score"],
                "kind": "black_special",
                "color_info": color_info,
                "result": black_result,
            }

        # Colored cards: action or number.
        action_crop = crop_card_roi(card, ACTION_ROI)
        action_result = self._classify_colored_action_crop(action_crop)
        number_result = self._classify_card_number_with_underline_fix(card)

        if action_result["score"] >= self.action_min_score:
            symbol = action_result["label"]
            return {
                "label": f"{color}_{symbol}",
                "color": color,
                "symbol": symbol,
                "score": action_result["score"],
                "kind": "colored_action",
                "color_info": color_info,
                "result": action_result,
                "number_candidate": number_result,
            }

        symbol = number_result["label"]
        return {
            "label": f"{color}_{symbol}",
            "color": color,
            "symbol": symbol,
            "score": number_result["score"],
            "kind": "number",
            "color_info": color_info,
            "result": number_result,
            "action_candidate": action_result,
        }


_CLASSIFIER_CACHE: Dict[Tuple[str, float, float], UnoCardClassifier] = {}


def classify_uno_card(
    cropped_card: ImageInput,
    reference_dir: Union[str, Path],
    input_color: str = "BGR",
    return_details: bool = False,
    action_min_score: float = 0.34,
    black_threshold: float = 0.13,
) -> Union[str, Dict[str, Any]]:
    """
    Example
    -------
    label = classify_uno_card(card_crop, reference_dir="reference_images")
    # label: "r_5", "b_skip", "wild", "wild_draw4", ...
    """
    key = (str(Path(reference_dir).resolve()), action_min_score, black_threshold)

    if key not in _CLASSIFIER_CACHE:
        _CLASSIFIER_CACHE[key] = UnoCardClassifier(
            reference_dir=reference_dir,
            action_min_score=action_min_score,
            black_threshold=black_threshold,
        )

    return _CLASSIFIER_CACHE[key].classify(
        cropped_card,
        input_color=input_color,
        return_details=return_details,
    )
