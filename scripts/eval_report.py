import json, glob, os, shutil
# Locate last results.json
cands = glob.glob("runs/classify/**/results.json", recursive=True)
assert cands, "Run validation with save_json=True first."
latest = max(cands, key=os.path.getmtime)
stats = json.load(open(latest))
print("[REPORT] Keys:", list(stats.keys()))
# Optionally copy allied plots
for png in glob.glob(os.path.join(os.path.dirname(latest),"*.png")):
    shutil.copy(png, "artifacts/")
print("[OK] Artifacts synced to artifacts/")
