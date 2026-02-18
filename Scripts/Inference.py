from ultralytics import YOLO
import cv2
import numpy as np
import torch
import argparse
from mobilefacenet_arcface import MobileFaceNet

# --------------------------------------------------
# Args
# --------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--source", default="0")
parser.add_argument("--face_weights", default="mobilefacenet.pt")
args = parser.parse_args()

# --------------------------------------------------
# Load YOLOv8 Face Model
# --------------------------------------------------
yolo = YOLO("../Model/weights/cerberusface_yolov8n.pt")

# --------------------------------------------------
# Load MobileFaceNet
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

face_model = MobileFaceNet(embedding_size=128).to(device)
state = torch.load(args.face_weights, map_location=device)
face_model.load_state_dict(state, strict=True)
face_model.eval()

print("✅ MobileFaceNet loaded correctly")

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def safe_crop(img, x1, y1, x2, y2):
    h, w, _ = img.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return img[y1:y2, x1:x2]


def preprocess(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (112, 112))
    face = (face.astype(np.float32) / 127.5) - 1.0
    face = np.transpose(face, (2, 0, 1))
    return torch.from_numpy(face).unsqueeze(0)


# --------------------------------------------------
# Video
# --------------------------------------------------
src = int(args.source) if args.source.isdigit() else args.source
cap = cv2.VideoCapture(src)

# --------------------------------------------------
# Main Loop
# --------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    results = yolo.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.4,
        iou=0.5
    )[0]

    if results.boxes:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tid = int(box.id[0]) if box.id is not None else -1

            face = safe_crop(frame, x1, y1, x2, y2)
            if face.size == 0 or (x2 - x1) < 40:
                continue

            face_tensor = preprocess(face).to(device)

            with torch.no_grad():
                emb = face_model(face_tensor)[0]
                norm = torch.norm(emb).item()

            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                display,
                f"ID:{tid} emb_norm:{norm:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2
            )

    cv2.imshow("Face Tracking + Embedding", display)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
