# HAIKUTHON - SigLIP2-large fine-tuning on Kaggle
# Notebook settings: Accelerator = GPU T4 x2 (uses 1), Internet = On, Add Input = 2026-haikuthon competition data.
# Run 1: TRAIN_ON_ALL=False -> read "best epoch". Run 2: TRAIN_ON_ALL=True, EPOCHS=best epoch -> final weights.

# %% settings
CKPT = "google/siglip2-large-patch16-384"   # 256 / 384 / 512 exist; 512 = finer detail, ~2x slower
RES = 384                                   # must match CKPT
N_TRAIN_LAYERS = 6                          # last N blocks of each tower are trained, the rest frozen
EPOCHS, BS, LR, WARMUP = 8, 32, 1e-5, 50
TRAIN_ON_ALL = False                        # True = no validation split, train on all 4000 pairs
OUT = "/kaggle/working/siglip2-ft"

# %% setup
import glob, os, re, random, math, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoTokenizer

dev = "cuda"
torch.manual_seed(0); random.seed(0); np.random.seed(0)
csv_path = glob.glob("/kaggle/input/**/prompts.csv", recursive=True)[0]
root = os.path.dirname(csv_path)
df = pd.read_csv(csv_path, dtype={"id": str})
P = df["prompt"].str.lower().str.strip().tolist()   # SigLIP2 was trained on lowercased text
print(root, len(P))

# %% images: decode once into RAM as uint8 (4000x3x384x384 ~ 1.8GB). No flip aug: prompts say left/right.
def load(i):
    im = Image.open(f"{root}/images/{i}.png").convert("RGB").resize((RES, RES), Image.BICUBIC)
    return torch.from_numpy(np.array(im)).permute(2, 0, 1)  # np.array copies -> writable, no warning
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    IMGS = torch.stack(list(ex.map(load, df["id"])))
print(f"images loaded {tuple(IMGS.shape)} in {time.time() - t0:.0f}s")
def norm(x):  # = SiglipImageProcessor: rescale to [0,1], mean=std=0.5
    return (x.to(dev, non_blocking=True).float() / 255 - 0.5) / 0.5

# %% text: model only sees 64 tokens, so long prompts are packed into <=64-token sentence chunks;
# a prompt's embedding = mean of its chunk embeddings (same code must be used in model.py).
tok = AutoTokenizer.from_pretrained(CKPT)
ntok = lambda s: len(tok(s)["input_ids"])
def pack(p):
    chunks = []
    for s in re.split(r"(?<=\.)\s+", p):
        if chunks and ntok(chunks[-1] + " " + s) <= 64: chunks[-1] += " " + s
        else: chunks.append(s)
    return tok(chunks, padding="max_length", max_length=64, truncation=True, return_tensors="pt")["input_ids"]
CH = [pack(p) for p in P]
print("prompts needing >1 chunk:", sum(len(c) > 1 for c in CH), "/", len(CH))

# %% rule-based hard negative captions: swap two attributes of the same kind, or flip a relation.
# The test pool is full of images that differ from the prompt in exactly this way.
GROUPS = [
    "red orange yellow green blue purple pink brown black white gray grey beige silver gold".split(),
    "striped polka-dotted solid-colored plaid checkered floral".split(),
    "metal plastic wooden wood glass ceramic leather fabric paper stone".split(),
    "round square oval rectangular triangular".split(),
    "open closed".split(),
    "standing sitting lying".split(),
]
FLIPS = [("left", "right"), ("above", "below")]
def swap(p, a, b): return re.sub(rf"\b({re.escape(a)}|{re.escape(b)})\b", lambda m: b if m.group(0) == a else a, p)
def perturb(p):
    opts = []
    for g in GROUPS:
        present = [w for w in g if re.search(rf"\b{re.escape(w)}\b", p)]
        if len(present) >= 2: opts.append(tuple(random.sample(present, 2)))
    opts += [f for f in FLIPS if re.search(rf"\b({f[0]}|{f[1]})\b", p)]
    if not opts: return None
    q = swap(p, *random.choice(opts))
    return q if q != p else None

# %% split: prompts with identical content words (attribute-swap twins) stay on the same side
STOP = set("one two three four the a an is are and in of with from to it its scene shows set viewed view image under rendered as which that appears alongside present".split())
key = pd.Series([" ".join(sorted(set(re.findall(r"[a-z-]+", p)) - STOP)) for p in P])
gkeys = key.unique().tolist(); random.shuffle(gkeys)
val_keys = set() if TRAIN_ON_ALL else set(gkeys[: len(gkeys) // 4])
val_idx = np.where(key.isin(val_keys))[0]; tr_idx = np.where(~key.isin(val_keys))[0]
print("train", len(tr_idx), "val", len(val_idx))

# %% model: AutoModel -> SiglipModel; freeze all but the last N blocks + heads/norms/logit scale & bias
model = AutoModel.from_pretrained(CKPT, attn_implementation="sdpa").to(dev)  # fp32 master weights
for n, p in model.named_parameters():
    m = re.search(r"encoder\.layers\.(\d+)\.", n)
    p.requires_grad = (int(m.group(1)) >= 24 - N_TRAIN_LAYERS) if m else ("embeddings" not in n)
print("trainable params (M):", round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6))

def enc_img(x): return F.normalize(model.vision_model(pixel_values=x).pooler_output.float(), dim=-1)
def enc_txt(chs):  # chs: list of [k,64] id tensors, one per prompt
    ids = torch.cat(chs).to(dev)
    grp = torch.repeat_interleave(torch.arange(len(chs)), torch.tensor([len(c) for c in chs])).to(dev)
    e = F.normalize(model.text_model(input_ids=ids).pooler_output.float(), dim=-1)
    return F.normalize(torch.zeros(len(chs), e.shape[1], device=dev).index_add_(0, grp, e), dim=-1)

@torch.no_grad()
def embed(idx, bs=64):
    model.eval(); I, T = [], []
    with torch.autocast("cuda", torch.float16):
        for s in range(0, len(idx), bs):
            b = idx[s:s + bs]
            I.append(enc_img(norm(IMGS[b]))); T.append(enc_txt([CH[i] for i in b]))
    return torch.cat(I), torch.cat(T)

def score_from_ranks(r):  # r = 1-based rank of the correct image; competition metric
    r = np.asarray(r, float)
    return float(np.mean(((r == 1) + np.where(r <= 5, 1 / np.log2(r + 1), 0)) / 2))

def evaluate(idx):
    I, T = embed(idx)
    sim = T @ I.T                                   # query i's correct image is column i
    r = ((sim > sim.diag()[:, None]).sum(1) + 1).cpu().numpy()
    return score_from_ranks(r), float((r == 1).mean())

if len(val_idx): print("zero-shot val (score, R@1):", evaluate(val_idx))

# %% hard batches: random seed prompt + its nearest neighbours by text embedding, so near-duplicate
# prompts/images are in-batch negatives. Recomputed every epoch with the current model.
def hard_batches(idx):
    _, T = embed(idx); sim = (T @ T.T).cpu()
    left = set(range(len(idx))); order = list(range(len(idx))); random.shuffle(order); out = []
    for s in order:
        if s not in left: continue
        nb = [j for j in sim[s].argsort(descending=True).tolist() if j in left][:BS]
        left -= set(nb); out.append(idx[nb])
    random.shuffle(out); return out   # ponytail: O(n^2) greedy, fine for 4k prompts

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
total = EPOCHS * math.ceil(len(tr_idx) / BS)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1))))
scaler = torch.amp.GradScaler()
save = lambda: (model.save_pretrained(OUT, state_dict={k: v.half() for k, v in model.state_dict().items()}), tok.save_pretrained(OUT))
best, best_ep = -1, -1

for ep in range(EPOCHS):
    t0 = time.time(); batches = hard_batches(tr_idx)
    model.train(); tot = 0
    for b in batches:
        negs = [(k, perturb(P[i])) for k, i in enumerate(b)]
        negs = [(k, q) for k, q in negs if q]
        with torch.autocast("cuda", torch.float16):
            I = enc_img(norm(IMGS[b]))
            T = enc_txt([CH[i] for i in b] + [pack(q) for _, q in negs])
        logits = T @ I.T * model.logit_scale.exp() + model.logit_bias   # rows: texts (pos then neg), cols: images
        labels = -torch.ones_like(logits); labels[: len(b)].fill_diagonal_(1)
        loss = -F.logsigmoid(labels * logits).sum() / len(b)            # SigLIP sigmoid loss
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item()
    msg = f"epoch {ep + 1}/{EPOCHS}: loss {tot / len(batches):.3f}  {time.time() - t0:.0f}s"
    if len(val_idx):
        s, r1 = evaluate(val_idx); msg += f"  val score {s:.4f}  R@1 {r1:.4f}"
        if s > best: best, best_ep = s, ep + 1; save()
    print(msg)

if not len(val_idx): save(); print("saved (all data) ->", OUT)
else: print(f"best val score {best:.4f} at epoch {best_ep} -> {OUT}  (for run 2 set EPOCHS={best_ep}, TRAIN_ON_ALL=True)")
