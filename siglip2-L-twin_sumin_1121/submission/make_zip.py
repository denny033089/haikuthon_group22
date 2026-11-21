"""제출 ZIP 만들기:  python tools/make_zip.py <가중치 폴더> <zip 이름>

ZIP 구성 (채점 서버 규격)
  model.py
  models/config.json            {"weights": "<폴더 이름>"}
  models/<폴더 이름>/...         config.json, model.safetensors, tokenizer*, processor_config.json
"""
import json
import os
import sys
import zipfile

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
wdir, name = os.path.abspath(sys.argv[1]), sys.argv[2]
wname = os.path.basename(wdir.rstrip("/\\"))
out = os.path.abspath(name)
os.makedirs(os.path.dirname(out), exist_ok=True)
need = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
missing = need - set(os.listdir(wdir))
assert not missing, f"가중치 폴더에 없는 파일: {missing}"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    z.write(os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.py"), "model.py")
    z.writestr("models/config.json", json.dumps({"weights": wname}))
    for f in sorted(os.listdir(wdir)):
        p = os.path.join(wdir, f)
        if os.path.isfile(p):
            ct = zipfile.ZIP_STORED if f.endswith(".safetensors") else zipfile.ZIP_DEFLATED
            z.write(p, f"models/{wname}/{f}", compress_type=ct)
with zipfile.ZipFile(out) as z:
    for i in z.infolist():
        print(f"  {i.filename:<55s} {i.file_size / 1e6:10.1f} MB")
print(f"→ {out}  ({os.path.getsize(out) / 1e9:.2f} GB, 제한 5 GB)")
