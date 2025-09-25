import cv2, numpy as np, onnxruntime as ort, glob, os

MODEL="artifacts/best.onnx"
LABELS=[l.strip() for l in open("artifacts/labels.txt")]

sess=ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp=sess.get_inputs()[0].name

def prep(img, size=224):
    h,w=img.shape[:2]; s=min(h,w); y=(h-s)//2; x=(w-s)//2
    img=img[y:y+s,x:x+s]; img=cv2.resize(img,(size,size))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    img=(img-[0.485,0.456,0.406])/[0.229,0.224,0.225]
    return np.transpose(img,(2,0,1))[None,...].astype(np.float32)

print("=== TESTING MODEL CONFIDENCE ON SAMPLE IMAGES ===\n")

# Test a few images from each class
for class_name in LABELS:
    print(f"--- {class_name.upper()} SAMPLES ---")
    class_path = f"waste_cls/test/{class_name}"
    if os.path.exists(class_path):
        images = glob.glob(f"{class_path}/*.*")[:3]  # Test first 3 images
        for img_path in images:
            img = cv2.imread(img_path)
            if img is not None:
                x = prep(img)
                probs = sess.run(None, {inp: x})[0][0]
                
                max_idx = np.argmax(probs)
                max_conf = probs[max_idx]
                predicted = LABELS[max_idx]
                
                print(f"  {os.path.basename(img_path):20} -> {predicted:12} ({max_conf:.3f})")
                
                # Show all confidence scores
                for i, (label, prob) in enumerate(zip(LABELS, probs)):
                    if prob > 0.1:  # Only show significant scores
                        print(f"    {label:12}: {prob:.3f}")
    print()

print("=== CONFIDENCE THRESHOLD ANALYSIS ===")
print("Current threshold: 0.70")
print("If confidence < 0.70, model returns 'UNKNOWN'")
print("\nTo fix the paper detection issue:")
print("1. Lower confidence threshold to 0.50 or 0.60")
print("2. Add more diverse paper samples to training data")
print("3. Retrain with more epochs")
