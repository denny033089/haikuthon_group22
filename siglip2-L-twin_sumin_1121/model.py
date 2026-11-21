"""HAIKUTHON final submission: two-stage text -> image retrieval.

stage 1 : twin-trained SigLIP2-L (models/<s1>)  -> cosine similarity over all candidates -> top-k_max per query
stage 2 : Qwen3-VL-Reranker-2B, first 16 LLM layers, LoRA-trained on twin images (models/<rr>), 256px images
          "does this image match the description?" yes-minus-no logit, fused with the stage-1 score.
          Anytime schedule: first ranks 1..k_base for EVERY query, then rank k_base+1, k_base+2, ... for every
          query while time allows; reranking stops at budget_s seconds after predict() starts (limit is 15 min).
Each query only uses its own text and its own candidates' scores (no cross-query information).
"""
import json
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

DEFAULT_CFG = {
    "s1": "siglip2L_twin", "rr": "reranker16",
    "res": 384, "rr_res": 256, "max_len": 64,
    "img_bs": 64, "txt_bs": 256, "num_workers": 4,
    "k_base": 3,        # ranks every query gets reranked
    "k_max": 10,        # deepest rank reranked if time is left (test: 70% of answers are in the twin model's top-10)
    "rr_bs": 24,        # pairs per reranker forward
    "alpha": 4.0,       # final = z(stage-1) + alpha * z(reranker) inside the reranked set
    "budget_s": 780,    # stop reranking 13 min after predict() starts
}
JUDGE_SYS = ('Judge whether the Document meets the requirements based on the Query and the Instruct provided. '
             'Note that the answer can only be "yes" or "no".')
INSTR = "<Instruct>: Retrieve images that match the user's description exactly.<Query>:"


def _load_cfg(model_dir):
    cfg = dict(DEFAULT_CFG)
    p = os.path.join(model_dir, "models", "config.json")
    if os.path.exists(p):
        with open(p) as f:
            cfg.update(json.load(f))
    if os.environ.get("HAIKU_CFG"):
        cfg.update(json.loads(os.environ["HAIKU_CFG"]))
    return cfg


class _Imgs(torch.utils.data.Dataset):
    """returns (stage-1 image [3,res,res] uint8, reranker image [rr_res,rr_res,3] uint8), both BICUBIC as in training"""
    def __init__(self, paths, res, rr_res):
        self.paths, self.res, self.rr_res = paths, res, rr_res

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            im = im.convert("RGB")
            a = np.asarray(im.resize((self.res, self.res), Image.BICUBIC)).copy()
            b = np.asarray(im.resize((self.rr_res, self.rr_res), Image.BICUBIC)).copy()
        return torch.from_numpy(a).permute(2, 0, 1), torch.from_numpy(b)


def _batches(paths, cfg):
    ds = _Imgs(paths, cfg["res"], cfg["rr_res"])
    nw = min(cfg["num_workers"], os.cpu_count() or 1)
    import multiprocessing as mp
    if nw > 0 and "fork" in mp.get_all_start_methods():
        yield from torch.utils.data.DataLoader(ds, batch_size=cfg["img_bs"], num_workers=nw,
                                               multiprocessing_context="fork", prefetch_factor=4)
        return
    from concurrent.futures import ThreadPoolExecutor
    bs = cfg["img_bs"]
    def load(s):
        items = [ds[i] for i in range(s, min(s + bs, len(ds)))]
        return torch.stack([x for x, _ in items]), torch.stack([y for _, y in items])
    with ThreadPoolExecutor(max(1, nw)) as ex:
        for f in [ex.submit(load, s) for s in range(0, len(ds), bs)]:
            yield f.result()


def load_model(model_dir, device):
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

    cfg = _load_cfg(model_dir)
    root = os.path.join(model_dir, "models")
    p1 = cfg["s1"] if os.path.isabs(cfg["s1"]) else os.path.join(root, cfg["s1"])
    p2 = cfg["rr"] if os.path.isabs(cfg["rr"]) else os.path.join(root, cfg["rr"])
    s1 = AutoModel.from_pretrained(p1, attn_implementation="sdpa", local_files_only=True).to(device).half().eval()
    tok = AutoTokenizer.from_pretrained(p1, local_files_only=True)
    rr = Qwen3VLForConditionalGeneration.from_pretrained(p2, dtype=torch.float16, attn_implementation="sdpa",
                                                         local_files_only=True).to(device).eval()
    proc = AutoProcessor.from_pretrained(p2, local_files_only=True)
    proc.tokenizer.padding_side = "left"                     # last position = answer slot for every row
    voc = proc.tokenizer.get_vocab()
    yes = [voc[w] for w in ("yes", "Yes") if w in voc]
    no = [voc[w] for w in ("no", "No") if w in voc]
    W = rr.lm_head.weight[yes + no].detach().float()
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
    return {"s1": s1, "tok": tok, "rr": rr, "proc": proc, "W": W, "n_yes": len(yes), "cfg": cfg, "device": device}


def _conv(text, img):
    return [{"role": "system", "content": [{"type": "text", "text": JUDGE_SYS}]},
            {"role": "user", "content": [{"type": "text", "text": INSTR}, {"type": "text", "text": text},
                                         {"type": "text", "text": "\n<Document>:"}, {"type": "image", "image": img}]}]


@torch.no_grad()
def _score_pairs(m, texts, imgs):
    """reranker yes-minus-no logit for each (text, PIL image) pair"""
    inp = m["proc"].apply_chat_template([_conv(t, im) for t, im in zip(texts, imgs)], tokenize=True,
                                        add_generation_prompt=True, return_dict=True, padding=True,
                                        return_tensors="pt").to(m["device"])
    h = m["rr"].model(**inp, use_cache=False).last_hidden_state[:, -1].float()
    lg = h @ m["W"].T
    n = m["n_yes"]
    return (torch.logsumexp(lg[:, :n], 1) - torch.logsumexp(lg[:, n:], 1)).cpu()


def _z(x):
    return (x - x.mean()) / (x.std() + 1e-6)


@torch.no_grad()
def predict(model, queries, candidates, top_k=5):
    t0 = time.time()
    m, cfg, dev = model, model["cfg"], model["device"]
    ids = [c["image_id"] for c in candidates]

    # ---- stage 1: SigLIP2-L embeddings (images also kept at rr_res for the reranker)
    emb, small = [], []
    for big, sm in _batches([c["image_path"] for c in candidates], cfg):
        x = big.to(dev, non_blocking=True).half().div_(255).sub_(0.5).div_(0.5)
        emb.append(F.normalize(m["s1"].vision_model(pixel_values=x).pooler_output.float(), dim=-1))
        small.append(sm)
    img = torch.cat(emb)
    small = torch.cat(small)                                  # [N, rr_res, rr_res, 3] uint8, CPU
    txt = []
    texts = [q["prompt"].strip().lower() for q in queries]   # SigLIP2 text: lowercase, 64 tokens
    for s in range(0, len(texts), cfg["txt_bs"]):
        tid = m["tok"](texts[s:s + cfg["txt_bs"]], padding="max_length", max_length=cfg["max_len"],
                       truncation=True, return_tensors="pt")["input_ids"].to(dev)
        txt.append(F.normalize(m["s1"].text_model(input_ids=tid).pooler_output.float(), dim=-1))
    txt = torch.cat(txt)
    kmax = min(max(cfg["k_max"], top_k), len(ids))
    TV, TI = [], []
    for s in range(0, len(queries), 1024):
        v, i = (txt[s:s + 1024] @ img.T).topk(kmax, dim=1)
        TV.append(v.cpu()); TI.append(i.cpu())
    TV, TI = torch.cat(TV), torch.cat(TI)
    t_s1 = time.time() - t0

    # ---- stage 2: rerank, anytime (all queries ranks 1..k_base, then one more rank at a time)
    Q = len(queries)
    RS = torch.full((Q, kmax), float("nan"))
    kb = min(cfg["k_base"], kmax)
    order = [(a, r) for a in range(Q) for r in range(kb)] + [(a, r) for r in range(kb, kmax) for a in range(Q)]
    deadline = t0 + cfg["budget_s"]
    done = 0
    raw = [q["prompt"].strip() for q in queries]             # reranker reads the original text
    for s in range(0, len(order), cfg["rr_bs"]):
        if time.time() > deadline:
            break
        ch = order[s:s + cfg["rr_bs"]]
        sc = _score_pairs(m, [raw[a] for a, _ in ch], [Image.fromarray(small[int(TI[a, r])].numpy()) for a, r in ch])
        for (a, r), v in zip(ch, sc.tolist()):
            RS[a, r] = v
        done += len(ch)

    # ---- fuse: scored ranks form a prefix of each query's stage-1 list
    out = {}
    for a, q in enumerate(queries):
        c = int((~torch.isnan(RS[a])).sum())
        rank = list(range(kmax))
        if c >= 2:
            f = _z(TV[a, :c]) + cfg["alpha"] * _z(RS[a, :c])
            rank = f.argsort(descending=True).tolist() + list(range(c, kmax))
        out[q["prompt_id"]] = [ids[int(TI[a, j])] for j in rank][:top_k]
    model["stats"] = {"stage1_s": round(t_s1, 1), "pairs_reranked": done, "pairs_planned": len(order),
                      "total_s": round(time.time() - t0, 1)}
    return out
