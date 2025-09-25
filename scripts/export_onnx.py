import os
from ultralytics import YOLO

best = "runs/classify/train/weights/best.pt"
assert os.path.exists(best), "best.pt not found. Train first."

YOLO(best).export(format="onnx")  # writes alongside best.pt
os.makedirs("artifacts", exist_ok=True)
src = "runs/classify/train/weights/best.onnx"
dst = "artifacts/best.onnx"
os.replace(src, dst)
open("artifacts/labels.txt","w").write("\n".join([
    "battery", "biological", "brown-glass", "cardboard", "clothes",
    "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"
]))
print("[OK] Exported -> artifacts/best.onnx + labels.txt")
