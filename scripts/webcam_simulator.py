import cv2, numpy as np, onnxruntime as ort
from collections import deque

MODEL="artifacts/best.onnx"     # or best-int8.onnx
LABELS=[l.strip() for l in open("artifacts/labels.txt")]

CONF_T=0.50       # below this -> "unknown"
EVERY_N=5         # classify every N frames
VOTE_WIN=15       # majority vote window size

sess=ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
inp=sess.get_inputs()[0].name

def prep(img, size=224):
    h,w=img.shape[:2]; s=min(h,w); y=(h-s)//2; x=(w-s)//2
    img=img[y:y+s,x:x+s]; img=cv2.resize(img,(size,size))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    img=(img-[0.485,0.456,0.406])/[0.229,0.224,0.225]
    return np.transpose(img,(2,0,1))[None,...].astype(np.float32)

cap=cv2.VideoCapture(0) # choose webcam
win=deque(maxlen=VOTE_WIN)
i=0
while True:
    ok,frame=cap.read()
    if not ok: break
    h,w=frame.shape[:2]
    # ROI: middle band (adjust once on belt)
    y1=int(0.33*h); y2=int(0.67*h); x1=int(0.25*w); x2=int(0.75*w)
    roi=frame[y1:y2,x1:x2]
    if i % EVERY_N == 0:
        p=sess.run(None,{inp:prep(roi)})[0][0]
        idx=int(np.argmax(p)); conf=float(np.max(p))
        lab=LABELS[idx] if conf>=CONF_T else "unknown"
        win.append(lab)
    # vote
    if win:
        vals,cts=np.unique(list(win),return_counts=True)
        j=int(np.argmax(cts)); voted,ratio=vals[j],cts[j]/len(win)
    else:
        voted,ratio="unknown",0.0

    cv2.rectangle(frame,(x1,y1),(x2,y2),(255,255,255),1)
    cv2.putText(frame,f"{voted.upper()} vote={ratio:.2f}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0) if voted!="unknown" else (0,0,255),2)
    cv2.imshow("Laptop Belt Simulator", frame)
    i+=1
    if cv2.waitKey(1)&0xFF==27: break
cap.release(); cv2.destroyAllWindows()
