import os, glob, cv2, hashlib

ROOT = "data"
bad = []
hashes = set()

for p in glob.glob(f"{ROOT}/**/*.*", recursive=True):
    if not p.lower().endswith((".jpg",".jpeg",".png")): continue
    img = cv2.imread(p)
    if img is None or min(img.shape[:2]) < 80:
        bad.append(p); continue
    h = hashlib.md5(open(p,'rb').read()).hexdigest()
    if h in hashes:
        bad.append(p); continue
    hashes.add(h)

print(f"[INFO] Files scanned: {len(hashes)}")
print("[WARN] Bad/dupe/small images:")
for b in bad: print(b)
