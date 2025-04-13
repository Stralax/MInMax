import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import os

# -------------------------
# 1. CNN Architecture
# -------------------------
class SimpleCharCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------
# 2. Mappings
# -------------------------
CHAR_MEANINGS = {
    "A": "start of something new", "B": "conflict", "C": "children and protection",
    "D": "home", "E": "unexpected news", "F": "friendship", "G": "joy",
    "H": "pride", "I": "surprise", "J": "love", "K": "expansion", "L": "lies",
    "M": "healing", "N": "rejection", "O": "success", "P": "pregnancy", "Q": "success with children",
    "R": "relief", "S": "happiness", "T": "timeliness", "U": "material desire", "V": "victory", "W": "spiritual warning", "X": "crossroads", "Y" : "Crossroads, options in decisions, or help coming from a friend", "Z": "ending",
    "!": "danger or excitement", "?": "concern", ":": "good luck",
    "0" : "Things repeating them selves, going in circles", "1" : "Start of something new", "2" : "Choice(s) or indecision between two things",
    "3": "meeting", "4": "jealousy", "5": "success", "6" : "Sex",
    "7": "new love", "8": "Double luck", "9": "an offer", 
}

def map_position_to_zone(x, y, image_shape, part="wall"):
    h, w = image_shape
    if part == "wall":
        if y < h / 3:
            return "past"
        elif y < 2 * h / 3:
            return "present"
        else:
            return "future"
    elif part == "bottom":
        cx, cy = w // 2, h // 2
        dx, dy = x - cx, y - cy
        angle = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle) % 360
        if 45 <= angle_deg < 135:
            return "future"
        elif 135 <= angle_deg < 225:
            return "home"
        elif 225 <= angle_deg < 315:
            return "present"
        else:
            return "love"
    return "unknown"

# -------------------------
# 3. Load Model + Class Names
# -------------------------
def load_model(path="char_symbol_cnn.pt"):
    checkpoint = torch.load(path, map_location="cpu")
    class_names = checkpoint["class_names"]
    model = SimpleCharCNN(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, class_names

# -------------------------
# 4. Preprocess and Predict
# -------------------------
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

# -------------------------
# 5. Full Detection Pipeline
# -------------------------
def detect_and_interpret(image_path, model_path="char_symbol_cnn.pt"):
    zone_priority = {"past": 0, "present": 1, "future": 2}
    model, class_names = load_model(model_path)
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = image.shape[:2]
    results = {}
    drawn_image = image_rgb.copy()

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if 10 <= bw <= 60 and 10 <= bh <= 60:
            patch = image_rgb[y:y+bh, x:x+bw]
            cx, cy = x + bw // 2, y + bh // 2
            try:
                char = predict_char(patch, model, class_names)
            except:
                continue
            meaning = CHAR_MEANINGS.get(char.upper(), "unknown")
            zone = map_position_to_zone(cx, cy, (h, w), part="wall")

            # Draw a red circle at the center of the detected patch
            cv2.circle(drawn_image, (cx, cy), 8, (255, 0, 0), 2)  # red circle, thickness 2

            # Only update if char is new or zone has higher priority
            if char not in results or zone_priority[zone] > zone_priority[results[char][1]]:
                results[char] = (meaning, zone)

    return results, drawn_image


# -------------------------
# 6. Run the Script
# -------------------------
if __name__ == "__main__":
    IMAGE_PATH = "sonja_kafe_out.jpg"
    predictions, annotated_image = detect_and_interpret(IMAGE_PATH)

    print("\n🔮 Fortune Telling Result:\n")
    for char, (meaning, zone) in sorted(predictions.items()):
        print(f"'{char}' → ({meaning}, {zone})")

    # Show or save image with circles
    annotated_bgr = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("sonja_result.jpg", annotated_bgr)
    cv2.imshow("Detected Symbols", annotated_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()