"""jina-clip-v2 를 transformers 5.x 에서 올바르게 불러오기 위한 호환 패치.

문제 1) 원격 코드가 transformers.models.clip.modeling_clip.clip_loss 를 import 하는데 5.x 에서 삭제됨
        → 같은 이름의 함수를 넣어 줌 (학습용 손실이라 추론에는 쓰이지 않음)
문제 2) 5.x 는 모델을 meta 장치에서 만든 뒤 가중치만 채워 넣음. 가중치 파일에 없는 '비저장 버퍼'가
        초기화되지 않은 쓰레기 값으로 남음 → 에러 없이 돌지만 임베딩이 엉터리가 됨
        - 텍스트 RoPE  inv_freq            → 원래 공식(_compute_inv_freq)으로 재계산
        - 이미지 RoPE  freqs_cos/freqs_sin → 같은 인자로 모듈을 새로 만들어 복사
        - LoRA         lora_dropout_mask   → 원래 값인 1 로 채움
"""
import torch


def patch_clip_loss():
    import transformers.models.clip.modeling_clip as mc
    if not hasattr(mc, "clip_loss"):
        def clip_loss(similarity):
            t = torch.arange(len(similarity), device=similarity.device)
            return (torch.nn.functional.cross_entropy(similarity, t)
                    + torch.nn.functional.cross_entropy(similarity.t(), t)) / 2.0
        mc.clip_loss = clip_loss


@torch.no_grad()
def fix_buffers(model):
    vc = model.config.vision_config
    n_text = n_vis = n_lora = 0
    for name, mod in model.named_modules():
        cls = type(mod).__name__
        if cls == "RotaryEmbedding" and hasattr(mod, "_compute_inv_freq"):
            mod.inv_freq.copy_(mod._compute_inv_freq(device=mod.inv_freq.device))
            assert getattr(mod, "scale_base", None) is None, "scale buffer 재계산 미구현"
            mod._seq_len_cached = 0  # cos/sin 캐시 무효화
            n_text += 1
        elif cls == "VisionRotaryEmbeddingFast":
            half_head_dim = vc.width // (vc.width // vc.head_width) // 2
            hw = vc.image_size // vc.patch_size
            fresh = type(mod)(dim=half_head_dim, pt_seq_len=vc.pt_hw_seq_len,
                              ft_seq_len=hw if vc.intp_freq else None, patch_dropout=mod.patch_dropout)
            assert fresh.freqs_cos.shape == mod.freqs_cos.shape, (fresh.freqs_cos.shape, mod.freqs_cos.shape)
            mod.freqs_cos.copy_(fresh.freqs_cos)
            mod.freqs_sin.copy_(fresh.freqs_sin)
            n_vis += 1
        if hasattr(mod, "lora_dropout_mask"):
            mod.lora_dropout_mask.fill_(1.0)
            n_lora += 1
    print(f"[jina_compat] 버퍼 복원: 텍스트 RoPE {n_text}, 이미지 RoPE {n_vis}, LoRA mask {n_lora}")


def load_jina(model_id_or_path, device="cpu", half=False):
    patch_clip_loss()
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id_or_path, trust_remote_code=True)
    fix_buffers(model)
    model = model.to(device)
    if half:
        model = model.half()
    return model.eval()
