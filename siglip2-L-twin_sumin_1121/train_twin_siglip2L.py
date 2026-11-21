# %% [markdown]
# # HAIKUTHON — SigLIP2-large 쌍둥이(twin) 학습
# - 기반: `siglip2-L/train_siglip2_kaggle.py` (2번 팀원) — sigmoid 손실, 마지막 N층만 학습, 규칙 기반 오답 문장
# - 바꾼 점
#   1. **쌍둥이 묶음**(핵심 단어 70% 이상 겹치는 문장끼리 연결)을 노트북 안에서 계산
#   2. **배치 = 쌍둥이 묶음 통째로** 채움 → 같은 배치에 쌍둥이 그림·문장이 서로의 '헷갈리는 오답'이 됨
#   3. **검증 = 쌍둥이 묶음 단위 분할** (묶음 전체가 학습 또는 검증 한쪽에만) → 실제 테스트처럼 쌍둥이가 갤러리에 같이 있음
#   4. 조기 종료(검증 점수가 PATIENCE epoch 연속 안 오르면 중단), WiSE-FT(학습 전·후 가중치 섞기) 비교
#   5. 텍스트는 공식 방식(소문자, 64토큰 max_length 패딩) — 문장 나눠 평균은 zero-shot에서 효과 없어서 뺌
# - 1차(TRAIN_ON_ALL=False): 최적 epoch·α 찾기 / 2차(TRAIN_ON_ALL=True): 4,000쌍 전부로 다시 학습해 가중치 저장
# - Kaggle: GPU T4 (1장 사용), Internet On, Input = 2026 HAIKUTHON 대회 데이터

# %% 셀 1. 설정
CKPT = "google/siglip2-large-patch16-384"
RES, MAX_LEN = 384, 64
N_TRAIN_LAYERS = 6            # 이미지·텍스트 각각 마지막 N층만 학습 (24층 중)
EPOCHS, PATIENCE = 8, 2       # 최대 epoch, 조기 종료 인내
BS = 32                       # 배치 = 쌍둥이 묶음 여러 개
LR, WD, WARMUP = 1e-5, 0.01, 50
NEG_PER_SAMPLE = 1            # 규칙 기반 오답 문장: 문장당 최대 1개
VAL_FRAC = 0.2                # 묶음의 20%를 검증으로
TWIN_JACCARD = 0.7            # 핵심 단어 겹침 비율 기준
ALPHAS = [0.3, 0.5, 0.7, 1.0] # WiSE-FT: 0=학습 전, 1=학습 결과 그대로
TRAIN_ON_ALL = False          # 2차 실행에서 True
FINAL_EPOCHS, FINAL_ALPHA = 5, 1.0   # 1차 결과: best epoch 5, α=1.0 (val score 0.5433 → 0.6399)
SEED = 0
OUT = "/kaggle/working/siglip2L_twin"

# %% 셀 2. 환경·데이터
import glob, os, re, json, math, random, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, torch, torch.nn.functional as F
import transformers
from PIL import Image
from transformers import AutoModel, AutoTokenizer

os.makedirs(OUT, exist_ok=True)
dev = "cuda:0"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
print("torch", torch.__version__, "| transformers", transformers.__version__, "| GPU", torch.cuda.get_device_name(0), flush=True)

csv_path = sorted(glob.glob("/kaggle/input/**/prompts.csv", recursive=True))[0]
root = os.path.dirname(csv_path)
df = pd.read_csv(csv_path, dtype={"id": str})
P = df["prompt"].str.strip().str.lower().tolist()      # SigLIP2는 소문자로 학습됨
print(root, len(P), flush=True)

# %% 셀 3. 쌍둥이 묶음 계산 (로컬 .twin_groups.json 과 같은 규칙: 가장 비슷한 문장과 70% 이상 겹치면 연결)
STOP = set("one a an is the in with and of to from scene view lighting are as that which there it "
           "image shown shows viewed rendered set setting also present appears appear".split())
words = [set(w for w in re.findall(r"[a-z\-]+", p) if w not in STOP) for p in P]
vocab = {w: i for i, w in enumerate(sorted(set().union(*words)))}
M = torch.zeros(len(P), len(vocab))
for i, ws in enumerate(words):
    M[i, [vocab[w] for w in ws]] = 1
inter = M @ M.T
sz = M.sum(1)
jac = inter / (sz[:, None] + sz[None, :] - inter)
jac.fill_diagonal_(0)
best_v, best_j = jac.max(1)
parent = list(range(len(P)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]; x = parent[x]
    return x
for i in range(len(P)):
    if best_v[i] >= TWIN_JACCARD:
        parent[find(i)] = find(int(best_j[i]))
groups = {}
for i in range(len(P)):
    groups.setdefault(find(i), []).append(i)
groups = list(groups.values())
GID = np.zeros(len(P), dtype=int)
for k, g in enumerate(groups):
    GID[g] = k
from collections import Counter
print("쌍둥이 묶음", len(groups), "| 크기 분포", sorted(Counter(len(g) for g in groups).items()), flush=True)

# %% 셀 4. 묶음 단위 분할
rng = random.Random(SEED)
gs = groups[:]; rng.shuffle(gs)
n_val = 0 if TRAIN_ON_ALL else int(len(gs) * VAL_FRAC)
val_groups, tr_groups = gs[:n_val], gs[n_val:]
val_idx = sorted(i for g in val_groups for i in g)
tr_idx = sorted(i for g in tr_groups for i in g)
vs = set(val_idx)
twin_in_val = sum(1 for i in val_idx if any(j in vs and j != i for j in groups[GID[i]]))
print(f"train {len(tr_idx)}쌍 ({len(tr_groups)}묶음) | val {len(val_idx)}쌍 ({len(val_groups)}묶음), "
      f"쌍둥이가 갤러리에 있는 val 쿼리 {twin_in_val} ({twin_in_val / max(1, len(val_idx)):.0%})", flush=True)
json.dump({"val_idx": val_idx, "tr_idx": tr_idx}, open(f"{OUT}/split.json", "w"))

# %% 셀 5. 이미지 미리 읽기 (4000x3x384x384 uint8 ≈ 1.8GB RAM). 좌우반전·색 변형 없음 (left/right, 색 속성이 정답에 영향)
def load(i):
    im = Image.open(f"{root}/images/{i}.png").convert("RGB").resize((RES, RES), Image.BICUBIC)
    return torch.from_numpy(np.array(im)).permute(2, 0, 1)
t0 = time.time()
with ThreadPoolExecutor(8) as ex:
    IMGS = torch.stack(list(ex.map(load, df["id"])))
print(f"이미지 {tuple(IMGS.shape)} 로드 {time.time() - t0:.0f}s", flush=True)
def norm(x):   # SiglipImageProcessor 와 동일: [0,1] 로 바꾼 뒤 mean=std=0.5
    return (x.to(dev, non_blocking=True).float() / 255 - 0.5) / 0.5

tok = AutoTokenizer.from_pretrained(CKPT)
def tokenize(texts):
    return tok(texts, padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="pt")["input_ids"]
TOK = tokenize(P)

# %% 셀 6. 규칙 기반 오답 문장 (2번 팀원 코드): 같은 종류 속성 두 개를 맞바꾸거나 관계를 뒤집음
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

# %% 셀 7. 모델: 마지막 N층 + 헤드/정규화/logit scale·bias 만 학습, 단어 사전(embeddings)은 고정
model = AutoModel.from_pretrained(CKPT, attn_implementation="sdpa").to(dev)   # fp32 원본 + fp16 autocast
LV = model.config.vision_config.num_hidden_layers
LT = model.config.text_config.num_hidden_layers
for n, p in model.named_parameters():
    m = re.search(r"encoder\.layers\.(\d+)\.", n)
    if m:
        L = LV if n.startswith("vision_model") else LT
        p.requires_grad = int(m.group(1)) >= L - N_TRAIN_LAYERS
    else:
        p.requires_grad = "embeddings" not in n
train_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
print(f"층 수 vision {LV} / text {LT} | 학습 파라미터 {sum(p.numel() for _, p in train_params) / 1e6:.0f}M "
      f"/ 전체 {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M", flush=True)
INIT = {n: p.detach().cpu().clone() for n, p in train_params}    # WiSE-FT 용 학습 전 값

def enc_img(x): return F.normalize(model.vision_model(pixel_values=x).pooler_output.float(), dim=-1)
def enc_txt(ids): return F.normalize(model.text_model(input_ids=ids).pooler_output.float(), dim=-1)

# %% 셀 8. 평가: val 텍스트 → val 이미지 (쌍둥이가 같이 들어 있는 갤러리)
@torch.no_grad()
def evaluate(idx):
    model.eval(); I, T = [], []
    with torch.autocast("cuda", torch.float16):
        for s in range(0, len(idx), 64):
            b = torch.tensor(idx[s:s + 64])
            I.append(enc_img(norm(IMGS[b]))); T.append(enc_txt(TOK[b].to(dev)))
    I, T = torch.cat(I), torch.cat(T)
    sim = (T @ I.T).cpu()
    r = ((sim > sim.diag()[:, None]).sum(1) + 1).numpy().astype(float)
    r1 = float((r == 1).mean()); nd = float(np.where(r <= 5, 1 / np.log2(r + 1), 0).mean())
    pos = {i: k for k, i in enumerate(idx)}
    ok = n = 0                                   # 쌍둥이 묶음 안에서 자기 그림을 1등으로 골랐나
    for i in idx:
        mem = [pos[j] for j in groups[GID[i]] if j in pos]
        if len(mem) >= 2:
            q = pos[i]; n += 1
            ok += int(mem[int(sim[q, mem].argmax())] == q)
    return {"score": (r1 + nd) / 2, "R@1": r1, "NDCG@5": nd, "twin_acc": ok / max(1, n), "n_twin_q": n}

def fmt(m): return f"score {m['score']:.4f}  R@1 {m['R@1']:.4f}  NDCG@5 {m['NDCG@5']:.4f}  쌍둥이 내 정답률 {m['twin_acc']:.4f} (n={m['n_twin_q']})"

log = {"config": {k: globals()[k] for k in ["CKPT", "N_TRAIN_LAYERS", "EPOCHS", "PATIENCE", "BS", "LR", "WD",
                                              "WARMUP", "NEG_PER_SAMPLE", "VAL_FRAC", "TWIN_JACCARD", "TRAIN_ON_ALL", "SEED"]},
       "n_train": len(tr_idx), "n_val": len(val_idx), "epochs": []}
if val_idx:
    t0 = time.time(); zs = evaluate(val_idx); log["zero_shot"] = zs
    print(f"[zero-shot] {fmt(zs)}  ({time.time() - t0:.0f}s)", flush=True)

# %% 셀 9. 학습
def twin_batches(tr_groups):
    g = [x[:BS] for x in tr_groups]; random.shuffle(g)
    out, cur = [], []
    for x in g:
        if cur and len(cur) + len(x) > BS:
            out.append(cur); cur = []
        cur = cur + list(x)
    if cur: out.append(cur)
    random.shuffle(out)
    return out

n_epochs = FINAL_EPOCHS if TRAIN_ON_ALL else EPOCHS
steps_per_epoch = len(twin_batches(tr_groups))
total = EPOCHS * steps_per_epoch   # 2차도 1차와 같은 8-epoch 코사인 스케줄을 쓰고 FINAL_EPOCHS 에서 멈춤 (1차 최고점과 같은 학습률 경로)
opt = torch.optim.AdamW([p for _, p in train_params], lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(s / total, 1))))
scaler = torch.amp.GradScaler()
best, best_ep, best_state, bad = -1.0, 0, None, 0

for ep in range(n_epochs):
    t0 = time.time(); model.train(); tot = 0.0; nneg = 0
    batches = twin_batches(tr_groups)
    for b in batches:
        negs = []
        for k, i in enumerate(b):
            for _ in range(NEG_PER_SAMPLE):
                q = perturb(P[i])
                if q: negs.append(q)
        ids = TOK[torch.tensor(b)]
        if negs: ids = torch.cat([ids, tokenize(negs)])
        nneg += len(negs)
        with torch.autocast("cuda", torch.float16):
            I = enc_img(norm(IMGS[torch.tensor(b)]))
            T = enc_txt(ids.to(dev))
            logits = T @ I.T * model.logit_scale.exp() + model.logit_bias      # 행: 문장(정답들, 오답 문장들) / 열: 그림
        labels = -torch.ones_like(logits); labels[: len(b)].fill_diagonal_(1)
        loss = -F.logsigmoid(labels * logits.float()).sum() / len(b)             # SigLIP sigmoid 손실
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_([p for _, p in train_params], 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        tot += loss.item()
    rec = {"epoch": ep + 1, "loss": tot / len(batches), "sec": round(time.time() - t0), "neg_texts": nneg}
    msg = f"epoch {ep + 1}/{n_epochs}: loss {rec['loss']:.3f}  {rec['sec']}s  (오답 문장 {nneg}개)"
    if val_idx:
        m = evaluate(val_idx); rec.update(m); msg += f"  | val {fmt(m)}"
        if m["score"] > best:
            best, best_ep, bad = m["score"], ep + 1, 0
            best_state = {n: p.detach().cpu().clone() for n, p in train_params}
        else:
            bad += 1
    log["epochs"].append(rec); print(msg, flush=True)
    json.dump(log, open(f"{OUT}/log.json", "w"), indent=1, ensure_ascii=False)
    if val_idx and bad >= PATIENCE:
        print(f"조기 종료: {PATIENCE} epoch 연속 개선 없음", flush=True); break

# %% 셀 10. WiSE-FT: 학습 전(INIT)과 최고 epoch(best_state) 가중치를 α 비율로 섞어 비교
@torch.no_grad()
def set_alpha(alpha, state):
    for n, p in train_params:
        p.copy_(((1 - alpha) * INIT[n] + alpha * state[n]).to(p.device, p.dtype))

if val_idx:
    log["best_epoch"], log["best_score"] = best_ep, best
    log["wise"] = {}
    for a in ALPHAS:
        set_alpha(a, best_state); m = evaluate(val_idx); log["wise"][str(a)] = m
        print(f"[WiSE-FT α={a}] {fmt(m)}", flush=True)
    best_a = max(ALPHAS, key=lambda a: log["wise"][str(a)]["score"])
    log["best_alpha"] = best_a
    print("=" * 70)
    print(f"zero-shot : {fmt(log['zero_shot'])}")
    print(f"best      : epoch {best_ep}, α={best_a} → {fmt(log['wise'][str(best_a)])}")
    print(f"→ 2차 실행: TRAIN_ON_ALL=True, FINAL_EPOCHS={best_ep}, FINAL_ALPHA={best_a}")
else:
    final = {n: p.detach().cpu().clone() for n, p in train_params}
    set_alpha(FINAL_ALPHA, final)
    model.half().save_pretrained(f"{OUT}/model")
    tok.save_pretrained(f"{OUT}/model")
    from transformers import AutoProcessor
    AutoProcessor.from_pretrained(CKPT).save_pretrained(f"{OUT}/model")
    print("파일:", sorted(os.listdir(f"{OUT}/model")),
          f"{sum(os.path.getsize(f'{OUT}/model/' + f) for f in os.listdir(f'{OUT}/model')) / 1e9:.2f} GB", flush=True)
    print("저장:", f"{OUT}/model", flush=True)
json.dump(log, open(f"{OUT}/log.json", "w"), indent=1, ensure_ascii=False)
