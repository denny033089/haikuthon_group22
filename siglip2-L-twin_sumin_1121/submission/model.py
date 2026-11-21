"""HAIKUTHON 제출: SigLIP2-large (쌍둥이 학습) 텍스트 → 이미지 검색.

구조
  load_model : ZIP 안 models/<weights> 폴더에서 오프라인으로 모델·토크나이저 로드 (fp16)
  predict    : 1) 후보 이미지 전체를 CPU 워커 4개로 디코딩 → GPU 에서 벡터화
               2) 문장 전체를 벡터화
               3) 코사인 유사도 상위 5개 반환
전처리는 학습 노트북(siglip2-L-twin_sumin_1121)과 동일: 384x384 BICUBIC 리사이즈, [0,1] → (x-0.5)/0.5, 문장은 소문자 + 64토큰.
각 문장은 다른 문장의 결과를 쓰지 않고 독립적으로 순위를 매김 (규정 준수).
"""
import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

DEFAULT_CFG = {
    "weights": "siglip2L_twin",  # models/ 아래 폴더 이름
    "res": 384,
    "max_len": 64,
    "lowercase": True,
    "img_bs": 64,
    "txt_bs": 256,
    "num_workers": 4,
}


def _load_cfg(model_dir):
    cfg = dict(DEFAULT_CFG)
    p = os.path.join(model_dir, "models", "config.json")
    if os.path.exists(p):
        with open(p) as f:
            cfg.update(json.load(f))
    if os.environ.get("HAIKU_CFG"):  # 로컬 실험용
        cfg.update(json.loads(os.environ["HAIKU_CFG"]))
    return cfg


class _ImgDS(torch.utils.data.Dataset):
    def __init__(self, paths, res):
        self.paths, self.res = paths, res

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB").resize((self.res, self.res), Image.BICUBIC)
            return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)  # uint8 [3,H,W]


def load_model(model_dir, device):
    from transformers import AutoModel, AutoTokenizer

    cfg = _load_cfg(model_dir)
    path = cfg["weights"] if os.path.isabs(cfg["weights"]) else os.path.join(model_dir, "models", cfg["weights"])
    is_cuda = str(device).startswith("cuda")
    model = AutoModel.from_pretrained(path, attn_implementation="sdpa", local_files_only=True)
    model = model.to(device)
    model = model.half() if is_cuda else model.float()
    model.eval()
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if is_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return {"model": model, "tok": tok, "cfg": cfg, "device": device,
            "dtype": torch.float16 if is_cuda else torch.float32}


def _batches(paths, cfg, dev):
    """uint8 이미지 배치를 순서대로 내보냄. Linux: fork 프로세스 워커 / 그 외(Windows 등): 스레드 풀."""
    ds = _ImgDS(paths, cfg["res"])
    nw = min(cfg["num_workers"], os.cpu_count() or 1)
    import multiprocessing as mp
    if nw > 0 and "fork" in mp.get_all_start_methods():
        yield from torch.utils.data.DataLoader(
            ds, batch_size=cfg["img_bs"], num_workers=nw, multiprocessing_context="fork",
            pin_memory=str(dev).startswith("cuda"), prefetch_factor=4)
        return
    from concurrent.futures import ThreadPoolExecutor
    bs = cfg["img_bs"]
    with ThreadPoolExecutor(max(1, nw)) as ex:
        futs = [ex.submit(lambda s: torch.stack([ds[i] for i in range(s, min(s + bs, len(ds)))]), s)
                for s in range(0, len(ds), bs)]
        for f in futs:
            yield f.result()


@torch.no_grad()
def _encode_images(m, paths):
    cfg, dev, dtype = m["cfg"], m["device"], m["dtype"]
    out = []
    for x in _batches(paths, cfg, dev):
        x = x.to(dev, non_blocking=True).float().div_(255).sub_(0.5).div_(0.5).to(dtype)
        e = m["model"].vision_model(pixel_values=x).pooler_output
        out.append(F.normalize(e.float(), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def _encode_texts(m, texts):
    cfg, dev = m["cfg"], m["device"]
    if cfg["lowercase"]:
        texts = [t.strip().lower() for t in texts]
    out = []
    for s in range(0, len(texts), cfg["txt_bs"]):
        ids = m["tok"](texts[s:s + cfg["txt_bs"]], padding="max_length", max_length=cfg["max_len"],
                       truncation=True, return_tensors="pt")["input_ids"].to(dev)
        e = m["model"].text_model(input_ids=ids).pooler_output
        out.append(F.normalize(e.float(), dim=-1))
    return torch.cat(out)


@torch.no_grad()
def predict(model, queries, candidates, top_k=5):
    m = model
    ids = [c["image_id"] for c in candidates]
    img = _encode_images(m, [c["image_path"] for c in candidates])  # [N, D]
    txt = _encode_texts(m, [q["prompt"] for q in queries])         # [Q, D]
    k = min(top_k, img.shape[0])
    preds = {}
    for s in range(0, txt.shape[0], 1024):                          # 문장 1024개씩 독립적으로 상위 k
        top = torch.topk(txt[s:s + 1024] @ img.T, k=k, dim=1).indices.cpu().tolist()
        for q, row in zip(queries[s:s + 1024], top):
            preds[q["prompt_id"]] = [ids[j] for j in row]
    return preds
