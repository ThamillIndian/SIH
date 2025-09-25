import shutil, random, os
random.seed(42)
SRC="data"; OUT="waste_cls"; splits=["train","val","test"]; ratios=[0.7,0.2,0.1]

classes = [d for d in os.listdir(SRC) if os.path.isdir(os.path.join(SRC,d))]
for s in splits:
    for c in classes:
        os.makedirs(os.path.join(OUT,s,c), exist_ok=True)

for c in classes:
    src = os.path.join(SRC,c)
    files=[os.path.join(src,f) for f in os.listdir(src)
           if f.lower().endswith(('.jpg','.jpeg','.png'))]
    random.shuffle(files)
    n=len(files); a=int(.7*n); b=int(.9*n)
    buckets={"train": files[:a], "val": files[a:b], "test": files[b:]}
    for s,chunk in buckets.items():
        for f in chunk:
            shutil.copy(f, os.path.join(OUT,s,c,os.path.basename(f)))

print("[OK] Split complete -> waste_cls/{train,val,test}/<class>")
