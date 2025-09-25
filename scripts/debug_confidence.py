import cv2, numpy as np, onnxruntime as ort
import os

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

# Test with webcam
cap=cv2.VideoCapture(0)
print("Press SPACE to capture and analyze, ESC to exit")

while True:
    ok,frame=cap.read()
    if not ok: break
    
    h,w=frame.shape[:2]
    # ROI: middle band
    y1=int(0.33*h); y2=int(0.67*h); x1=int(0.25*w); x2=int(0.75*w)
    roi=frame[y1:y2,x1:x2]
    
    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),2)
    cv2.putText(frame,"Press SPACE to analyze",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    cv2.imshow("Debug Analyzer", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == 32:  # SPACE
        # Analyze the ROI
        x = prep(roi)
        probs = sess.run(None, {inp: x})[0][0]
        
        print("\n=== CONFIDENCE ANALYSIS ===")
        for i, (label, prob) in enumerate(zip(LABELS, probs)):
            print(f"{label:12}: {prob:.4f} ({prob*100:.1f}%)")
        
        max_idx = np.argmax(probs)
        max_conf = probs[max_idx]
        predicted = LABELS[max_idx]
        
        print(f"\nPredicted: {predicted}")
        print(f"Confidence: {max_conf:.4f} ({max_conf*100:.1f}%)")
        print(f"Threshold: 0.70 ({70:.1f}%)")
        print(f"Result: {'CONFIDENT' if max_conf >= 0.70 else 'UNKNOWN'}")

cap.release()
cv2.destroyAllWindows()
