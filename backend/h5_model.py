# backend/h5_model.py
import json, torch, h5py
from types import SimpleNamespace
from pathlib import Path
from backend.multitask.model import MultiTaskLearner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_CKPT_PATH = Path(__file__).parent / "models" / "checkpoint_best.pth"

# 메모리 캐시
_models: dict[str, MultiTaskLearner] = {}
_state_dict = None        # 처음 호출 때만 읽어오게

# ────────────────────────────────────────────────
def _load_state_dict():
    global _state_dict
    if _state_dict is None:
        # checkpoint가 **신뢰된 파일**임을 가정하고 Pickle 허용
        ckpt = torch.load(_CKPT_PATH, map_location="cpu", weights_only=False)
        _state_dict = ckpt.get("state_dict", ckpt)     # 키가 없으면 전체가 곧 state_dict
    return _state_dict
# ────────────────────────────────────────────────

def get_model(args_json: str) -> MultiTaskLearner:
    """args_json = 프론트에서 온 옵션(JSON 문자열)"""
    if args_json not in _models:
        args = SimpleNamespace(**json.loads(args_json))
        net  = MultiTaskLearner(args).to(device)
        net.load_state_dict(_load_state_dict(), strict=False)
        net.eval()
        _models[args_json] = net
    return _models[args_json]

def load_h5(fp) -> tuple[torch.Tensor, dict]:
    """fp = file-like (BytesIO)"""
    with h5py.File(fp, "r") as f:
        img  = torch.from_numpy(f["image"][...]).float()   # (24,128,224)
        mus  = f["muscle_mask"][...]
        vert = f["vertebrae_mask"][...]
    img = img.unsqueeze(0).unsqueeze(0)   # (1,1,24,128,224)
    return img, {"mus": mus, "vert": vert}
