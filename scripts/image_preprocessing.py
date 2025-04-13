# Updated coffee cup analysis pipeline with improved circle detection logic and stitching

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import sys
import os

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")

# ---- 0. MiDaS Import Setup ----
sys.path.append(os.path.join(os.path.dirname(__file__), "MiDaS"))
from midas.dpt_depth import DPTDepthModel  # MiDaS local version

# ---- 1. Load MiDaS Model ----
def load_midas_model():
    model_path = "models/dpt_large-midas-2f21e586.pt"
    midas = DPTDepthModel(path=model_path, backbone="vitl16_384", non_negative=True)
    midas.eval()
    transform = Compose([
        Resize((384, 384)),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return midas, transform

# ---- 2. Depth Estimation ----
def estimate_depth(image, model, transform, original_shape):
    input_batch = transform(image).unsqueeze(0)
    with torch.no_grad():
        prediction = model(input_batch)[0]
    depth = prediction.squeeze().cpu().numpy()
    depth_resized = cv2.resize(depth, (original_shape[1], original_shape[0]))
    depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return depth_normalized

# ---- 3. Segment Cup from Depth ----
def segment_cup_from_depth(depth_map):
    inverted = cv2.bitwise_not(depth_map)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 51, -10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(depth_map)

    largest = max(contours, key=cv2.contourArea)
    cup_mask = np.zeros_like(depth_map)
    cv2.drawContours(cup_mask, [largest], -1, color=255, thickness=-1)
    return cup_mask

# ---- 4. Improved Circle Detection from Cup Mask ----
def fit_circle_to_cup_mask(cup_mask):
    blurred = cv2.medianBlur(cup_mask, 5)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=10,
        maxRadius=min(cup_mask.shape) // 2
    )
    circle_mask = np.zeros_like(cup_mask)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        cv2.circle(circle_mask, (x, y), r, 255, -1)
        return (x, y, r), circle_mask
    else:
        return None, circle_mask

# ---- 5. Split Bottom and Walls ----
def split_bottom_and_walls(depth_map, inside_cup_mask, depth_tolerance=15):
    h, w = depth_map.shape
    ys, xs = np.where(inside_cup_mask > 0)
    if len(xs) == 0:
        return np.zeros_like(depth_map), np.zeros_like(depth_map)

    center_x, center_y = int(np.mean(xs)), int(np.mean(ys))
    radius = min(h, w) // 10
    temp_mask = np.zeros_like(inside_cup_mask)
    cv2.circle(temp_mask, (center_x, center_y), radius, 255, -1)
    central_depths = depth_map[(temp_mask > 0) & (inside_cup_mask > 0)]
    if len(central_depths) == 0:
        return np.zeros_like(depth_map), np.zeros_like(depth_map)

    center_depth = np.median(central_depths)
    bottom_candidate = cv2.inRange(depth_map, center_depth - depth_tolerance, center_depth + depth_tolerance)
    bottom_mask = cv2.bitwise_and(bottom_candidate, inside_cup_mask)
    wall_mask = cv2.subtract(inside_cup_mask, bottom_mask)
    return bottom_mask, wall_mask

# ---- 6. Generate Inner Circle from Area Ratio ----
def generate_inner_circle_from_area_ratio(cup_mask, outer_circle, bottom_mask, max_area_ratio=0.4, shrink_factor=0.5):
    A_outer = np.count_nonzero(cup_mask)
    A_bottom = np.count_nonzero(bottom_mask)

    if A_outer == 0 or A_bottom == 0:
        return None, np.zeros_like(cup_mask)

    area_ratio = min(A_bottom / A_outer, max_area_ratio)
    scale = np.sqrt(area_ratio) * shrink_factor

    x, y, r = outer_circle
    inner_r = int(r * scale)
    inner_mask = np.zeros_like(cup_mask)
    cv2.circle(inner_mask, (x, y), inner_r, 255, -1)
    return (x, y, inner_r), inner_mask

# ---- 7. Unwrap Cup Wall ----
def unwrap_cup_wall(img_rgb, wall_mask, center, r_inner, r_outer, output_height=200, output_width=400):
    x0, y0 = center
    theta_vals = np.linspace(0, 2 * np.pi, output_width)
    r_vals = np.linspace(r_inner, r_outer, output_height)

    map_x = np.zeros((output_height, output_width), dtype=np.float32)
    map_y = np.zeros((output_height, output_width), dtype=np.float32)

    for i, r in enumerate(r_vals):
        for j, theta in enumerate(theta_vals):
            x = x0 + r * np.cos(theta)
            y = y0 + r * np.sin(theta)
            map_x[i, j] = x
            map_y[i, j] = y

    unwrapped = cv2.remap(img_rgb, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    unwrapped_mask = cv2.remap(wall_mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
    unwrapped[unwrapped_mask == 0] = 0

    pad_x = output_width // 10
    left = unwrapped[:, -pad_x:]
    right = unwrapped[:, :pad_x]
    unwrapped_padded = np.concatenate([left, unwrapped, right], axis=1)

    return unwrapped_padded

# ---- 8. Full Pipeline ----
def process_image(image_path):
    midas, transform = load_midas_model()
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}.")
        return

    original_shape = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    depth_map = estimate_depth(pil_img, midas, transform, original_shape)
    cup_mask = segment_cup_from_depth(depth_map)
    outer_circle, inside_cup_mask = fit_circle_to_cup_mask(cup_mask)
    depth_bottom_mask, _ = split_bottom_and_walls(depth_map, inside_cup_mask)
    inner_circle, bottom_mask = generate_inner_circle_from_area_ratio(cup_mask, outer_circle, depth_bottom_mask, shrink_factor=0.65)
    wall_mask = cv2.subtract(inside_cup_mask, bottom_mask)

    only_bottom_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=bottom_mask)
    only_walls_rgb = cv2.bitwise_and(img_rgb, img_rgb, mask=wall_mask)

    circle_overlay = img.copy()
    if outer_circle:
        cv2.circle(circle_overlay, (outer_circle[0], outer_circle[1]), outer_circle[2], (0, 255, 0), 2)
    if inner_circle:
        cv2.circle(circle_overlay, (inner_circle[0], inner_circle[1]), inner_circle[2], (255, 0, 0), 2)

    if outer_circle and inner_circle:
        center = (outer_circle[0], outer_circle[1])
        r_inner = inner_circle[2]
        r_outer = outer_circle[2]
        unwrapped_wall = unwrap_cup_wall(img_rgb, wall_mask, center, r_inner, r_outer)

        x, y, r = inner_circle
        diameter = 2 * r
        square = np.zeros((diameter, diameter, 3), dtype=np.uint8)

        x1 = x - r
        y1 = y - r
        x2 = x + r
        y2 = y + r

        img_h, img_w = img_rgb.shape[:2]
        crop_x1 = max(x1, 0)
        crop_y1 = max(y1, 0)
        crop_x2 = min(x2, img_w)
        crop_y2 = min(y2, img_h)

        src_crop = img_rgb[crop_y1:crop_y2, crop_x1:crop_x2]

        offset_x = crop_x1 - x1
        offset_y = crop_y1 - y1
        square[offset_y:offset_y + src_crop.shape[0], offset_x:offset_x + src_crop.shape[1]] = src_crop

        wall_height = unwrapped_wall.shape[0]
        resized_bottom = cv2.resize(square, (wall_height, wall_height), interpolation=cv2.INTER_AREA)

        stitched = np.hstack((unwrapped_wall, resized_bottom))
        stitched_bgr = cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR)
        save_path = "stitched_output.jpg"
        cv2.imwrite(save_path, stitched_bgr)
        print(f"Saved stitched image to {save_path}")

# ---- Example Call ----
process_image("sonja_kafe.jpg")
#process_image("kafe_test.jpg", shrink_factor=0.6)