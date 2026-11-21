# jina-clip-v2 zero-shot 평가

- 모델: [`jinaai/jina-clip-v2`](https://huggingface.co/jinaai/jina-clip-v2)
  - 텍스트: Jina-XLM-RoBERTa (최대 8192 토큰, 작업별 LoRA 어댑터)
  - 이미지: EVA02-L/14 @ 512px
  - 출력 1024차원 (Matryoshka: 앞부분만 잘라 512/256차원으로도 사용 가능)
- 실행 환경: Kaggle T4, torch 2.11, **transformers 5.20.2**
- 검증셋: 팀 공통 `RandomState(42).permutation` 앞 500개 (val 앞 10개 `[8, 12, 14, 17, 23, 26, 32, 33, 44, 45]`)

## 결과 (텍스트 500 → 이미지 500)

| text task | dim | R@1 | NDCG@5 | score | mean rank |
|---|---|---|---|---|---|
| **none** | **1024** | **0.8080** | **0.9239** | **0.8659** | 1.24 |
| none | 512 | 0.7960 | 0.9192 | 0.8576 | 1.25 |
| none | 256 | 0.7880 | 0.9158 | 0.8519 | 1.26 |
| retrieval.query | 1024 | 0.7860 | 0.9150 | 0.8505 | 1.26 |
| retrieval.query | 512 | 0.7860 | 0.9147 | 0.8504 | 1.27 |
| retrieval.query | 256 | 0.7940 | 0.9178 | 0.8559 | 1.26 |

- 같은 val 500 기준 SigLIP2 so400m (0.8353) 대비 **+0.031**
- `task="retrieval.query"`(검색 질문용 지시문을 붙이는 모드)가 오히려 더 나쁨 → **task 없이 쓰는 게 좋음**
- 차원을 256까지 줄여도 -0.01 수준
- 주의: 무작위 500개 갤러리에는 쌍둥이(색·위치만 바뀐 문장) 이미지가 적어서, 실제 테스트보다 점수가 높게 나옴. 모델 간 비교용으로만 볼 것

## ⚠️ transformers 5.x 호환 문제 (채점 서버도 5.22라서 그대로 해당)

jina-clip-v2는 HF에 올라간 원격 코드(`trust_remote_code=True`)로 동작하는데, 이 코드가 transformers 4.x 기준이라 5.x에서 두 가지가 깨짐.

1. **import 에러:** 원격 코드가 5.x에서 삭제된 `transformers.models.clip.modeling_clip.clip_loss`를 import함
   → 같은 이름의 함수를 넣어 주면 해결 (학습용 손실이라 추론에는 영향 없음)
2. **조용히 결과가 망가짐 (더 위험):** 5.x는 모델을 빈 껍데기(meta 장치)로 만든 뒤 가중치만 채움.
   가중치 파일에 없는 **비저장 버퍼**가 쓰레기 값으로 남음
   - 텍스트 RoPE `inv_freq` (정상 0~1 → -36 같은 값)
   - 이미지 RoPE `freqs_cos/freqs_sin` (전부 0)
   - LoRA `lora_dropout_mask` (원래 전부 1)

   → **에러 없이 돌지만 임베딩이 엉터리** (유사도 0.1~0.2대로 뒤죽박죽). 원래 공식대로 다시 계산해 덮어쓰면 정상 (정답 쌍 0.4~0.5)

`jina_compat.py`의 `load_jina()`가 두 문제를 모두 처리함. 노트북 5번 셀에도 같은 코드가 들어 있음.

```python
from jina_compat import load_jina
model = load_jina("jinaai/jina-clip-v2", device="cuda:0", half=True)
```

그 밖에: jina config는 `torch_dtype`을 문자열로만 받음 → `torch.float16`을 넘기면 TypeError. fp32로 불러온 뒤 `.half()` 할 것.

## 파일
- `01_zeroshot_jina_clip_v2.ipynb`: Kaggle 노트북 (GPU T4, Internet On). 셀 단위로 Run All
- `jina_compat.py`: transformers 5.x 호환 패치

## TODO
- 7번 셀의 이미지 인코딩 속도(16,000장 추정 시간) 기록 — 512px 입력이라 so400m보다 느릴 수 있음
- 제출용(오프라인) 내보내기 점검: 원격 코드 `.py` 파일까지 폴더 하나에 담기는지 (노트북 10번 셀)
