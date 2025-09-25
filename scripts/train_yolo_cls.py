import os, sys
from ultralytics import YOLO

DATA = "waste_cls"
MODEL = "yolov8n-cls.pt"
EPOCHS = int(os.getenv("EPOCHS", "30"))
IMGSZ = int(os.getenv("IMGSZ", "224"))
BATCH = int(os.getenv("BATCH", "32"))
LR0   = float(os.getenv("LR0", "0.003"))

if not os.path.isdir(DATA):
    print("[ERR] Missing waste_cls/ splits. Run `make split` first."); sys.exit(1)

print("[INFO] Starting training...")
m = YOLO(MODEL)
m.train(task="classify", data=DATA, epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        lr0=LR0, augment=True, mixup=0.1, device="cpu")
print("[OK] Training finished. Weights in runs/classify/train/weights/best.pt")
