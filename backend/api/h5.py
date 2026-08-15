# -*- coding: utf-8 -*-
import json, tempfile, random
from pathlib import Path
from uuid import uuid4
from typing import Dict

import numpy as np
from torch.serialization import add_safe_globals
import torch, h5py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query

from multitask.model import MultiTaskLearner
from utils import slice_uint8, to_b64, device

router = APIRouter(tags=["HDF5 (Multitask)"])

# ─────────── 세션/모델 캐시 ───────────
# sid → { 'items': {fid: {'name','image','muscle','vertebrae'} } }
h5_sessions: Dict[str, dict] = {}
h5_models  : Dict[str, MultiTaskLearner] = {}   # args_json → model
H5_CKPT = Path(__file__).resolve().parent.parent / "models" / "checkpoint_best.pth"

def get_h5_model(args_json: str) -> MultiTaskLearner:
    if args_json not in h5_models:
        # ── 안전 허용: numpy 스칼라/dtype
        try:
            add_safe_globals([np.dtype, np.core.multiarray.scalar])
        except Exception:
            pass

        # ── 신뢰 체크포인트라는 전제에서 weights_only=False 로드
        try:
            ckpt = torch.load(H5_CKPT, map_location=device, weights_only=False)
        except TypeError:
            # 구버전 torch는 weights_only 인자가 없음
            ckpt = torch.load(H5_CKPT, map_location=device)

        from types import SimpleNamespace
        args = SimpleNamespace(**json.loads(args_json))
        net  = MultiTaskLearner(args).to(device)

        state = ckpt.get("state_dict", ckpt)
        try:
            net.load_state_dict(state)
        except Exception:
            # 백본/헤드 키 일부만 느슨하게 로드 (체크포인트 구조 변화 대비)
            encoder_dict = {f"encoder.{k}": v for k, v in state.items() if k.startswith("backbone")}
            decoder_dict = {
                f"decoders.cls_decoder.{k.replace('classifier.', '')}": v
                for k, v in state.items() if k.startswith("classifier")
            }
            net.load_state_dict({**encoder_dict, **decoder_dict}, strict=False)

        net.eval()
        h5_models[args_json] = net
    return h5_models[args_json]


# ========= 전처리 유틸 =========
def normalize_like_script(img: np.ndarray) -> np.ndarray:
    """0~1000 clip → min-max [0,1] (script와 동일)"""
    img = np.clip(img, 0, 1000).astype(np.float32)
    vmax, vmin = img.max(), img.min()
    scale = max(vmax - vmin, 1e-6)
    return (img - vmin) / scale

def make_bone_only(image: np.ndarray, muscle_mask: np.ndarray, verte_mask: np.ndarray,
                   mode: str = "bone_only") -> np.ndarray:
    """
    mode: 'bone_only' | 'bone_muscle'
    반환: (D,H,W) float32 [0,1]
    """
    img = normalize_like_script(image)
    muscle_mask = muscle_mask.astype(np.uint8)
    verte_mask  = verte_mask.astype(np.uint8)
    muscle_mask[verte_mask == 1] = 0

    if mode == "bone_only":
        ver_img = img.copy()
        ver_img[verte_mask == 0] = 0
        return ver_img
    elif mode == "bone_muscle":
        ver_img = img.copy()
        mus_img = img.copy()
        ver_img[verte_mask == 0] = 0
        mus_img[muscle_mask == 0] = 0
        return ver_img + mus_img
    else:
        return img

def pick_stack_centers(vol: np.ndarray, k: int = 3) -> list[int]:
    """유효 슬라이스 후보 중에서 중심 인덱스 k개 선택 (양끝 제외)"""
    nonzero = np.where(vol.sum((1, 2)) > 1e-6)[0]
    D = vol.shape[0]
    if len(nonzero) < 5:
        # 내용 적으면 중앙 근처 보정
        c = D // 2
        candidates = [max(1, c-2), max(1, c), min(D-2, c+2)]
        return [int(np.clip(i, 1, D-2)) for i in candidates][:k]
    # 정상 케이스: 양끝 제외 후보 중에서 랜덤 샘플
    candidates = list(nonzero[1:-1])
    if len(candidates) >= k:
        picks = sorted(random.sample(candidates, k))
    else:
        # 부족하면 중앙 근처로 보충
        c = int(np.clip(nonzero[len(nonzero)//2], 1, D-2)) if len(nonzero) else D//2
        picks = (candidates + [c] * k)[:k]
        picks = [int(np.clip(i, 1, D-2)) for i in picks]
    return picks

def build_input_tensor_from_three_vertebra(item_list: list[dict],
                                           bone_only: str = "bone_only",
                                           channels_num_2d: int = 3) -> torch.Tensor:
    """
    item_list: [{'image','muscle','vertebrae','name'}, ...] (len=3)
    반환: (1, 9, 3, 128, 224) float32
    """
    stacks = []
    for it in item_list:
        vol = make_bone_only(it["image"], it["muscle"], it["vertebrae"], mode=bone_only)  # (D,H,W)
        centers = pick_stack_centers(vol, k=channels_num_2d)
        for i in centers:
            stacks.append(vol[i-1:i+2])  # (3,H,W)

    arr = np.array(stacks, dtype=np.float32)     # (9,3,128,224)
    ten = torch.from_numpy(arr).unsqueeze(0)     # (1,9,3,128,224)
    return ten

# ========= 엔드포인트 =========

@router.post("/load_h5_multi")
async def load_h5_multi(files: list[UploadFile] = File(...)):
    """
    HDF5 여러 개 업로드 → 세션 생성.
    응답: 각 파일의 (가운데 슬라이스) 썸네일과 파일명 리스트
    """
    if not files or len(files) == 0:
        raise HTTPException(400, "No files")
    sid = str(uuid4())
    items: Dict[str, dict] = {}

    for idx, f in enumerate(files):
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(await f.read())
            with h5py.File(tmp.name, "r") as h:
                img = h["image"][()]          # (D,H,W) float32
                mus = h["muscle_mask"][()]    # (D,H,W) uint8
                ver = h["vertebrae_mask"][()] # (D,H,W) uint8

        fid = f"{idx}"
        items[fid] = dict(
            name=f.filename,
            image=img, muscle=mus, vertebrae=ver,
        )

    h5_sessions[sid] = dict(items=items)

    # 썸네일(가운데 슬라이스)만 반환
    thumbs = []
    for fid, it in items.items():
        D = it["image"].shape[0]
        mid = D // 2
        thumbs.append(dict(
            fid=fid,
            name=it["name"],
            preview=dict(
                img      = to_b64(slice_uint8(it["image"][mid])),
                muscle   = to_b64(slice_uint8(it["muscle"][mid] * 255)),
                vertebra = to_b64(slice_uint8(it["vertebrae"][mid] * 255)),
            )
        ))
    return {"session": sid, "items": thumbs}

@router.get("/h5_volume")
async def h5_volume(session: str = Query(...), fid: str = Query(...)):
    """
    특정 업로드 항목(fid)의 전체 슬라이스(원본/마스크)를 Base64 목록으로 반환.
    미리보기 슬라이더용.
    """
    if session not in h5_sessions:
        raise HTTPException(400, "Bad session")
    items = h5_sessions[session]["items"]
    if fid not in items:
        raise HTTPException(400, "Bad fid")

    it = items[fid]
    img, mus, ver = it["image"], it["muscle"], it["vertebrae"]
    D = img.shape[0]
    volume = dict(
        img     = [to_b64(slice_uint8(img[i])) for i in range(D)],
        muscle  = [to_b64(slice_uint8(mus[i]*255)) for i in range(D)],
        vertebra= [to_b64(slice_uint8(ver[i]*255)) for i in range(D)],
        name    = it["name"],
    )
    return {"volume": volume}

@router.post("/predict_h5_multi")
async def predict_h5_multi(
    session: str      = Form(...),
    selected: str     = Form(...),   # "fid1,fid2,fid3"
    args_json: str    = Form(...),   # 프론트에서 보낸 모델 args
):
    if session not in h5_sessions:
        raise HTTPException(400, "Bad session")
    items = h5_sessions[session]["items"]

    fids = [s for s in selected.split(",") if s]
    if len(fids) != 3:
        raise HTTPException(400, "Select exactly 3 vertebra files")

    for fid in fids:
        if fid not in items:
            raise HTTPException(400, f"Bad fid: {fid}")

    args = json.loads(args_json)
    # 스크립트와 동일한 입력 구성: (1,9,3,128,224)
    bone_only        = args.get("bone_only", "bone_only")
    channels_num_2d  = int(args.get("channels_num_2d", 3))
    max_followup     = int(args.get("max_followup", 10))
    cls_enabled      = bool(args.get("cls_enabled", True))
    pred_enabled     = bool(args.get("pred_enabled", True))
    if not (cls_enabled or pred_enabled):
        raise HTTPException(400, "Choose at least one task (CLS or PRED).")

    item_list = [items[fid] for fid in fids]
    x = build_input_tensor_from_three_vertebra(item_list, bone_only, channels_num_2d).to(device)  # (1,9,3,128,224)

    # 모델
    net = get_h5_model(args_json)
    with torch.no_grad():
        sample = {
            "vers_img_tensor": x,
            "y": torch.tensor([0], dtype=torch.long, device=device),
            "time_at_event": torch.tensor([0], dtype=torch.long, device=device),
        }
        cls_out, pred_out, _ = net(sample, sample)

    result = {}
    if cls_enabled and ("prob" in cls_out and cls_out["prob"] is not None):
        # cls_out['prob']: softmax (B,2)
        p = float(cls_out["prob"][0,1].item())
        result["cls_probability"] = p

    if pred_enabled:
        # pred_out['prob']: sigmoid(logit) (B,T)
        if "prob" in pred_out and pred_out["prob"] is not None:
            curve = pred_out["prob"][0].detach().cpu().tolist()
        elif "logit" in pred_out and pred_out["logit"] is not None:
            curve = torch.sigmoid(pred_out["logit"])[0].detach().cpu().tolist()
        else:
            curve = []
        result["pred_curve"] = curve
        result["max_followup"] = max_followup

    result["selected_names"] = [items[f]["name"] for f in fids]
    return result
