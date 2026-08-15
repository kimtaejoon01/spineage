# -*- coding: utf-8 -*-
from pathlib import Path
from uuid import uuid4
from typing import Dict

import numpy as np
import torch, pydicom
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from torchvision.transforms import v2

from preprocess import n4_bias_correction, get_bbox, crop
from model import OVFNet
from utils import slice_uint8, to_b64, device

router = APIRouter(tags=["OVF (DICOM)"])

# ─────────── 상수 ───────────
AGE_MEAN, AGE_STD = 72.78, 9.38
BMD_MEAN, BMD_STD = -3.154, 1.077
ZERO_META = torch.zeros(1, 5, device=device)

# ─────────── 세션/모델 캐시 ───────────
dicom_sessions: Dict[str, np.ndarray] = {}   # sid → (F,H,W)
ovf_models   : Dict[bool, OVFNet]     = {}   # use_clinical → net

def get_ovf_model(flag: bool) -> OVFNet:
    if flag not in ovf_models:
        ck = Path(__file__).resolve().parent.parent / "models" / f"use_clinical_{int(flag)}.pth"
        net = OVFNet(flag).to(device)
        net.load_state_dict(torch.load(ck, map_location=device))
        net.eval()
        ovf_models[flag] = net
    return ovf_models[flag]

# ─────────── 워밍업 ───────────
async def warmup():
    dummy = torch.randn(1, 3, 224, 224, device=device)
    for flag in (False, True):
        with torch.no_grad():
            get_ovf_model(flag)(dummy, ZERO_META if flag else None)
    print("✅  OVF models pre-loaded & warm-up done")

# ─────────── 엔드포인트 (경로 동일 유지) ───────────

@router.post("/load")
async def load_dicom(files: list[UploadFile] = File(...)):
    """DICOM 업로드 → 썸네일과 세션ID 반환"""
    slices = []
    for f in files:
        f.file.seek(0)
        dcm = pydicom.dcmread(f.file, force=True)
        slices.append(dcm.pixel_array)
    arr = np.stack(slices)                       # (F,H,W)
    sid = str(uuid4())
    dicom_sessions[sid] = arr
    thumbs = [f"data:image/png;base64,{to_b64(slice_uint8(arr[i]))}" for i in range(arr.shape[0])]
    return {"session": sid, "thumbs": thumbs}

@router.post("/preview")
async def preview(session: str = Form(...), frames: str = Form(...), kps: str = Form(...)):
    if session not in dicom_sessions:
        raise HTTPException(400, "Bad session")
    arr = dicom_sessions[session]

    sel  = [int(x) for x in frames.split(",")]
    vals = list(map(int, kps.split(",")))
    kp   = np.array(list(zip(vals[::2], vals[1::2])), np.int32)

    bb  = get_bbox(kp, 1.5)
    roi = slice_uint8(crop(arr[sel[1]], bb))
    return {"png": to_b64(roi)}

@router.post("/predict")
async def predict(
    session: str = Form(...),
    frames : str = Form(...),
    kps    : str = Form(...),
    use_clinical: bool = Form(False),
    age: float|None = Form(None),
    sex: str|None   = Form(None),
    bmd: float|None = Form(None),
    pre: bool|None  = Form(False),
    post: bool|None = Form(False),
):
    if session not in dicom_sessions:
        raise HTTPException(400, "Bad session")
    arr = dicom_sessions[session]

    # N4 보정 1회
    if arr.dtype != np.float32:
        arr = n4_bias_correction(arr)
        dicom_sessions[session] = arr

    sel  = [int(x) for x in frames.split(",")]
    vals = list(map(int, kps.split(",")))
    kp   = np.array(list(zip(vals[::2], vals[1::2])), np.int32)
    bb   = get_bbox(kp, 1.5)

    tfm = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std =(0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    x = torch.stack([
        tfm(np.tile(crop(arr[i], bb)[..., None], (1, 1, 3)))
        for i in sel
    ]).to(device)

    meta = None
    if use_clinical:
        meta = torch.tensor([
            (age-AGE_MEAN)/AGE_STD,
            1 if sex and sex.lower() == "female" else 0,
            (bmd-BMD_MEAN)/BMD_STD,
            float(pre), float(post)
        ], dtype=torch.float32, device=device)

    net = get_ovf_model(use_clinical)
    with torch.no_grad():
        prob = torch.stack([
            torch.sigmoid(net(x[i,None], meta[None] if meta is not None else None)).squeeze()
            for i in range(3)
        ]).mean().item()

    return {"probability": prob}
