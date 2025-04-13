import os
import sys
import cv2
import math
import json
import torch
import random
import requests
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm.models.layers")

from io import BytesIO

# ===============================
# == MiDaS Depth + Preprocessing
# ===============================

sys.path.append(os.path.join(os.path.dirname(__file__), "MiDaS"))
from midas.dpt_depth import DPTDepthModel

def load_midas_model():
    model_path = "models/dpt_large-midas-2f21e586.pt"
    midas = DPTDepthModel(path=model_path, backbone="vitl16_384", non_negative=True)
    midas.eval()
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    return midas, transform

def estimate_depth(image, model, transform, original_shape):
    try:
        input_batch = transform(image).unsqueeze(0)
        with torch.no_grad():
            prediction = model(input_batch)[0]
        depth = prediction.squeeze().cpu().numpy()
        depth_resized = cv2.resize(depth, (original_shape[1], original_shape[0]))
        depth_normalized = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return depth_normalized
    except Exception:
        print("Bad image quality. Try taking image with better lighting or better angle!")
        sys.exit(1)

def segment_cup_from_depth(depth_map):
    if np.count_nonzero(depth_map) < 1000:
        print("Bad image quality. Try taking image with better lighting or better angle!")
        sys.exit(1)
    inverted = cv2.bitwise_not(depth_map)
    blurred = cv2.GaussianBlur(inverted, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 51, -10)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Bad image quality. Try taking image with better lighting or better angle!")
        sys.exit(1)
    largest = max(contours, key=cv2.contourArea)
    cup_mask = np.zeros_like(depth_map)
    cv2.drawContours(cup_mask, [largest], -1, color=255, thickness=-1)
    return cup_mask

def fit_circle_to_cup_mask(cup_mask):
    blurred = cv2.medianBlur(cup_mask, 5)
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50, param1=100, param2=30, minRadius=10, maxRadius=min(cup_mask.shape) // 2)
    circle_mask = np.zeros_like(cup_mask)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        x, y, r = circles[0][0]
        cv2.circle(circle_mask, (x, y), r, 255, -1)
        return (x, y, r), circle_mask
    else:
        print("Bad image quality. Try taking image with better lighting or better angle!")
        sys.exit(1)

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

def generate_inner_circle_from_area_ratio(cup_mask, outer_circle, bottom_mask, max_area_ratio=0.4, shrink_factor=0.65):
    A_outer = np.count_nonzero(cup_mask)
    A_bottom = np.count_nonzero(bottom_mask)
    if A_outer == 0 or A_bottom == 0:
        print("Bad image quality. Try taking image with better lighting or better angle!")
        sys.exit(1)
    area_ratio = min(A_bottom / A_outer, max_area_ratio)
    scale = np.sqrt(area_ratio) * shrink_factor
    x, y, r = outer_circle
    inner_r = int(r * scale)
    inner_mask = np.zeros_like(cup_mask)
    cv2.circle(inner_mask, (x, y), inner_r, 255, -1)
    return (x, y, inner_r), inner_mask

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
    return np.concatenate([left, unwrapped, right], axis=1)

def preprocess_and_stitch(image_path, output_path="stitched_output.jpg"):
    midas, transform = load_midas_model()
    img = cv2.imread(image_path)
    if img is None:
        print("Bad image quality. Try taking image with better lighting or better angle!")
        sys.exit(1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img_rgb.shape[:2]
    depth_map = estimate_depth(Image.fromarray(img_rgb), midas, transform, original_shape)
    cup_mask = segment_cup_from_depth(depth_map)
    outer_circle, inside_cup_mask = fit_circle_to_cup_mask(cup_mask)
    depth_bottom_mask, _ = split_bottom_and_walls(depth_map, inside_cup_mask)
    inner_circle, bottom_mask = generate_inner_circle_from_area_ratio(cup_mask, outer_circle, depth_bottom_mask)
    wall_mask = cv2.subtract(inside_cup_mask, bottom_mask)
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
    cv2.imwrite(output_path, cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR))

# ===============================
# == Character CNN + Detection
# ===============================

class SimpleCharCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 3, padding=1), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 7 * 7, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

CHAR_MEANINGS = {
    "A": "start of something new", "B": "conflict", "C": "children and protection",
    "D": "home", "E": "unexpected news", "F": "friendship", "G": "joy", "H": "pride", "I": "surprise",
    "J": "love", "K": "expansion", "L": "lies", "M": "healing", "N": "rejection", "O": "success",
    "P": "pregnancy", "Q": "success with children", "R": "relief", "S": "happiness", "T": "timeliness",
    "U": "material desire", "V": "victory", "W": "spiritual warning", "X": "crossroads",
    "Y": "Crossroads, options in decisions, or help coming from a friend", "Z": "ending",
    "!": "danger or excitement", "?": "concern", ":": "good luck",
    "0": "Things repeating themselves", "1": "Start of something new", "2": "Choice", "3": "meeting",
    "4": "jealousy", "5": "success", "6": "sex", "7": "new love", "8": "Double luck", "9": "an offer"
}

def map_position_to_zone(x, y, image_shape, part="wall"):
    h, w = image_shape
    if part == "wall":
        if y < h / 3: return "past"
        elif y < 2 * h / 3: return "present"
        else: return "future"
    else:
        cx, cy = w // 2, h // 2
        angle_deg = np.degrees(np.arctan2(y - cy, x - cx)) % 360
        if 45 <= angle_deg < 135: return "future"
        elif 135 <= angle_deg < 225: return "home"
        elif 225 <= angle_deg < 315: return "present"
        else: return "love"

def load_model(path="char_symbol_cnn.pt"):
    checkpoint = torch.load(path, map_location="cpu")
    class_names = checkpoint["class_names"]
    model = SimpleCharCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names

def preprocess_patch(patch):
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (28, 28))
    tensor = transforms.ToTensor()(resized)
    return tensor.unsqueeze(0)

def predict_char(patch, model, class_names):
    tensor = preprocess_patch(patch)
    with torch.no_grad():
        logits = model(tensor)
        pred = torch.argmax(logits, dim=1).item()
    return class_names[pred]

def detect_and_interpret(image_path, model_path="char_symbol_cnn.pt"):
    model, class_names = load_model(model_path)
    image = cv2.imread(image_path)
    if image is None:
        print("Bad image quality. Try taking image with better lighting or better angle!")
        sys.exit(1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    results = {}
    positions = {}
    drawn_image = image_rgb.copy()
    count_valid = 0

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if 10 <= bw <= 60 and 10 <= bh <= 60:
            patch = image_rgb[y:y+bh, x:x+bw]
            cx, cy = x + bw // 2, y + bh // 2
            try:
                char = predict_char(patch, model, class_names).upper()
                meaning = CHAR_MEANINGS.get(char, "unknown")
                zone = map_position_to_zone(cx, cy, (h, w), part="wall")
                results[char] = (meaning, zone)
                if char not in positions:
                    positions[char] = (cx, cy)
                count_valid += 1
            except:
                continue

    if count_valid == 0:
        print("Bad image quality. Try taking image with better lighting or better angle!")
        sys.exit(1)

    # Filter to ensure selected characters are far apart
    def is_far_enough(candidate_pos, chosen_positions, min_dist=50):
        for pos in chosen_positions:
            dist = math.hypot(candidate_pos[0] - pos[0], candidate_pos[1] - pos[1])
            if dist < min_dist:
                return False
        return True

    candidates = list(positions.items())
    random.shuffle(candidates)
    chosen = []
    chosen_positions = []

    for char, (cx, cy) in candidates:
        if is_far_enough((cx, cy), chosen_positions, min_dist=50):
            chosen.append((char, (cx, cy)))
            chosen_positions.append((cx, cy))
        if len(chosen) >= 6:
            break

    for char, (cx, cy) in chosen:
        cv2.circle(drawn_image, (cx, cy), 14, (255, 0, 0), 2)
        cv2.putText(drawn_image, char, (cx + 15, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imwrite("result_image.jpg", cv2.cvtColor(drawn_image, cv2.COLOR_RGB2BGR))

    output_data = []
    for char, (meaning, zone) in sorted(results.items()):
        output_data.append({
            "character": char,
            "meaning": meaning,
            "zone": zone
        })

    with open("fortune_results.json", "w") as f:
        json.dump(output_data, f, indent=4)

    #print("\n🔮 Fortune Telling Result:\n")
    for entry in output_data:
        print(f"'{entry['character']}' : ({entry['meaning']}, {entry['zone']})")

    # print("\n🔮 Fortune Telling Result:\n")
    # for char, (meaning, zone) in sorted(results.items()):
    #     print(f"'{char}' : ({meaning}, {zone})")


# === Continue with imgbb upload and final result saving ===
def upload_image_to_imgbb(image_path, api_key='63db8ca1d56f9c50ecbcee756b05f668', expiration=600):
    url = "https://api.imgbb.com/1/upload"
    with open(image_path, "rb") as file:
        payload = {
            "key": api_key,
            "expiration": str(expiration)
        }
        files = {
            "image": file
        }
        try:
            response = requests.post(url, data=payload, files=files)
            data = response.json()
            if data.get("success"):
                print("✅ Image uploaded:", data["data"]["url"])
                return data["data"]["url"]
            else:
                print("❌ Upload failed:", data)
                return None
        except Exception as e:
            print("❌ Error uploading image:", str(e))
            return None

# ==== Final main logic ====
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_url>")
        sys.exit(1)

    image_url = sys.argv[1]

    try:
        response = requests.get(image_url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        temp_input_path = "temp_input.jpg"
        image.save(temp_input_path)

        preprocess_and_stitch(temp_input_path)
        detect_and_interpret("stitched_output.jpg")

        # Upload result image
        image_link = upload_image_to_imgbb("result_image.jpg")

        # Load fortune results
        with open("fortune_results.json", "r") as f:
            results_json = json.load(f)

        # Combine and save to final output
        final_output = {
            "image_link": image_link,
            "results": results_json
        }

        with open("final_output.json", "w") as f:
            json.dump(final_output, f, indent=4)

        print("\n✅ Final output saved to final_output.json")
        print(f"🔗 Link: {image_link}")

    except Exception as e:
        print(f"❌ Failed to process image from URL: {e}")
