import onnxruntime as ort, cv2, numpy as np, glob, os, time
MODEL = "artifacts/best.onnx"   # or artifacts/best-int8.onnx
LABELS = [l.strip() for l in open("artifacts/labels.txt")]
sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
name = sess.get_inputs()[0].name

def prep(img, size=224):
    h,w=img.shape[:2]; s=min(h,w); y=(h-s)//2; x=(w-s)//2
    img=img[y:y+s,x:x+s]; img=cv2.resize(img,(size,size))
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)/255.
    img=(img-[0.485,0.456,0.406])/[0.229,0.224,0.225]
    return np.transpose(img,(2,0,1))[None,...].astype(np.float32)

ts = glob.glob("waste_cls/test/**/*.*", recursive=True)
ok, total = 0, 0
t0=time.time()
for p in ts:
    if not p.lower().endswith(('.jpg','.jpeg','.png')): continue
    img=cv2.imread(p); x=prep(img)
    probs=sess.run(None,{name:x})[0][0]
    i=int(np.argmax(probs)); conf=float(np.max(probs))
    pred = LABELS[i]
    true = p.split(os.sep)[-2]
    total += 1
    ok += int(pred==true)
print(f"[ACC] {ok}/{total} = {ok/total:.3f}  | runtime: {time.time()-t0:.2f}s")
