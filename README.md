# Smart Waste Segregation – 12-Class Classifier

## Scope
Single-item classification on conveyor: battery, biological, brown-glass, cardboard, clothes, green-glass, metal, paper, plastic, shoes, trash, white-glass.
Trained on ASUS TUF GPU using YOLOv8-Cls → exported to ONNX for RPi4 runtime.

## 🚀 Quick Start Commands

### 1. Initial Setup
```bash
# Create virtual environment and install dependencies
make setup

# Or manually:
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r env/requirements.txt
```

### 2. Data Preparation
```bash
# Scan data for bad/duplicate images
python scripts/sanity_scan.py

# Split data into train/val/test sets (70/20/10)
make split
# Or: python scripts/split_data.py
```

### 3. Model Training
```bash
# Train YOLOv8 classification model (30 epochs)
make train
# Or: python scripts/train_yolo_cls.py

# Validate trained model
make val
# Or: yolo task=classify mode=val model=runs/classify/train/weights/best.pt data=waste_cls
```

### 4. Model Export & Optimization
```bash
# Export to ONNX format
make export
# Or: python scripts/export_onnx.py

# Optional: Quantize model for faster inference
make quant
# Or: python scripts/quantize_onnx.py
```

### 5. Testing & Validation
```bash
# Test inference on test dataset
make dryrun
# Or: python scripts/dryrun_infer.py

# Test confidence scores on sample images
python scripts/test_confidence.py

# Run webcam simulator for real-time testing
make sim
# Or: python scripts/webcam_simulator.py

# Generate evaluation report
make report
# Or: python scripts/eval_report.py
```

### 6. Debug & Analysis
```bash
# Debug confidence issues with webcam
python scripts/debug_confidence.py

# Check model performance on test images
python scripts/test_confidence.py
```

## Artifacts

- `artifacts/best.onnx` (+ optional `best-int8.onnx`)
- `artifacts/labels.txt` (class order)
- `artifacts/confusion_matrix.png`
- `artifacts/simulator_screenshot.jpg`

## 📁 Project Structure
```
smart-waste/
├── data/                    # Raw class folders (populate with images)
│   ├── battery/
│   ├── biological/
│   ├── brown-glass/
│   ├── cardboard/
│   ├── clothes/
│   ├── green-glass/
│   ├── metal/
│   ├── paper/
│   ├── plastic/
│   ├── shoes/
│   ├── trash/
│   └── white-glass/
├── waste_cls/               # Auto-generated train/val/test splits
├── scripts/                 # All Python scripts
├── artifacts/              # Final models and outputs
├── runs/                   # Training outputs
├── env/requirements.txt    # Dependencies
├── Makefile               # Build commands
└── README.md              # This file
```

## ⚙️ Configuration & Thresholds

- **Confidence threshold:** 0.50 (below → "unknown" bin)
- **Temporal smoothing:** majority vote window = 15, classify every 5th frame
- **Training epochs:** 30 (increased from 15 for better accuracy)
- **Image size:** 224x224 pixels
- **Batch size:** 32

## Operational Assumptions

- Fixed, diffuse lighting above belt
- Single item in ROI at a time
- ROI tuned to belt sweet spot

## Next Step (RPi4)

Install python3-opencv, onnxruntime, gpiozero; drop in best.onnx, labels.txt,
and the Pi runtime (app.py). Map GPIO pins to diverters, keep the same thresholds.

## 🔧 Troubleshooting

### Common Issues & Solutions

**1. NumPy Compatibility Error**
```bash
# Error: AttributeError: _ARRAY_API not found
# Solution: Downgrade NumPy
pip install "numpy<2"
```

**2. CUDA Not Available**
```bash
# Error: Invalid CUDA 'device=0' requested
# Solution: Model automatically uses CPU (device="cpu")
# No action needed - training will work on CPU
```

**3. Low Model Accuracy**
```bash
# Solutions:
# 1. Increase training epochs
export EPOCHS=50
python scripts/train_yolo_cls.py

# 2. Lower confidence threshold
# Edit scripts/webcam_simulator.py: CONF_T=0.40

# 3. Add more training data to confused classes
```

**4. Webcam Not Working**
```bash
# Error: can't grab frame
# Solutions:
# 1. Check camera permissions
# 2. Try different camera index: cap=cv2.VideoCapture(1)
# 3. Use debug script: python scripts/debug_confidence.py
```

**5. ONNX Export Issues**
```bash
# Error: onnxslim compatibility
# Solution: Export still succeeds despite warnings
# Check artifacts/best.onnx exists
```

## 🖥️ System Requirements

### Minimum Requirements
- **OS:** Windows 10/11, Linux, macOS
- **Python:** 3.8+ (tested with 3.11.9)
- **RAM:** 8GB+ (16GB recommended)
- **Storage:** 5GB free space
- **CPU:** Multi-core processor (training on CPU)

### Recommended for Training
- **GPU:** NVIDIA GPU with CUDA support
- **RAM:** 16GB+
- **Storage:** SSD with 10GB+ free space

### For Raspberry Pi Deployment
- **Model:** Raspberry Pi 4B (4GB+ RAM)
- **OS:** Raspberry Pi OS
- **Storage:** 32GB+ microSD card

## Deliverable Definition of Done (pre-Pi)

- ✅ Dataset cleaned + split
- ✅ `runs/classify/.../best.pt` produced
- ✅ `artifacts/best.onnx` (and `best-int8.onnx`) present
- ✅ Dry-run accuracy printed ≥ 0.90
- ✅ Confusion matrix & simulator screenshot in `artifacts/`
- ✅ README updated with exact commands + measured metrics
