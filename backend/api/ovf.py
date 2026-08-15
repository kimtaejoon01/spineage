# -*- coding: utf-8 -*-
from pathlib import Path
from uuid import uuid4
from typing import Dict

import numpy as np
import torch
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from torchvision.transforms import v2

from preprocess import read_dicom_files, n4_bias_correction, get_bbox, crop
from model import OVFNet
from utils import slice_uint8, to_b64, device

router = APIRouter(tags=["OVF (DICOM)"])

AGE_MEAN, AGE_STD = 72.78, 9.38
BMD_MEAN, BMD_STD = -3.154, 1.077
ZERO_META = torch.zeros(1, 5, device=device)

dicom_sessions: Dict[str, dict] = {}
ovf_models: Dict[bool, OVFNet] = {}


def get_ovf_model(flag: bool) -> OVFNet:
    if flag not in ovf_models:
        ck = Path(__file__).resolve().parent.parent / "models" / f"use_clinical_{int(flag)}.pth"
        if not ck.exists():
            raise FileNotFoundError(f"OVF model checkpoint not found: {ck}")
        net = OVFNet(flag).to(device)
        net.load_state_dict(torch.load(ck, map_location=device))
        net.eval()
        ovf_models[flag] = net
    return ovf_models[flag]


async def warmup():
    dummy = torch.randn(1, 3, 224, 224, device=device)
    for flag in (False, True):
        with torch.no_grad():
            get_ovf_model(flag)(dummy, ZERO_META if flag else None)
    print("OVF models pre-loaded & warm-up done")


def _parse_selection(frames: str, kps: str, frame_count: int):
    try:
        sel = [int(x) for x in frames.split(",") if x != ""]
        vals = [int(x) for x in kps.split(",") if x != ""]
    except ValueError as exc:
        raise HTTPException(400, "Invalid frame or keypoint value") from exc

    if len(sel) != 3:
        raise HTTPException(400, "Select exactly 3 frames")
    if any(i < 0 or i >= frame_count for i in sel):
        raise HTTPException(400, "Frame index out of range")
    if len(vals) != 12:
        raise HTTPException(400, "Exactly 6 keypoints are required")

    kp = np.array(list(zip(vals[::2], vals[1::2])), dtype=np.int32)
    return sel, kp


def _get_session(session: str) -> dict:
    data = dicom_sessions.get(session)
    if data is None:
        raise HTTPException(400, "Bad session")
    return data


@router.post("/load")
async def load_dicom(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No DICOM files")
    try:
        arr = read_dicom_files(files)
    except Exception as exc:
        raise HTTPException(400, f"Failed to read DICOM series: {exc}") from exc

    sid = str(uuid4())
    dicom_sessions[sid] = {
        "volume": arr,
        "n4_corrected": False,
    }
    thumbs = [
        f"data:image/png;base64,{to_b64(slice_uint8(arr[i]))}"
        for i in range(arr.shape[0])
    ]
    return {"session": sid, "thumbs": thumbs}


@router.post("/preview")
async def preview(session: str = Form(...), frames: str = Form(...), kps: str = Form(...)):
    data = _get_session(session)
    arr = data["volume"]
    sel, kp = _parse_selection(frames, kps, arr.shape[0])

    bb = get_bbox(kp, 1.5)
    roi_raw = crop(arr[sel[1]], bb)
    if roi_raw.size == 0:
        raise HTTPException(400, "ROI is empty. Check keypoints.")
    return {"png": to_b64(slice_uint8(roi_raw))}


@router.post("/predict")
async def predict(
    session: str = Form(...),
    frames: str = Form(...),
    kps: str = Form(...),
    use_clinical: bool = Form(False),
    age: float | None = Form(None),
    sex: str | None = Form(None),
    bmd: float | None = Form(None),
    pre: bool | None = Form(False),
    post: bool | None = Form(False),
):
    data = _get_session(session)
    arr = data["volume"]
    sel, kp = _parse_selection(frames, kps, arr.shape[0])

    if use_clinical:
        if age is None or bmd is None or not sex:
            raise HTTPException(400, "age, sex and bmd are required when clinical data is enabled")
        if not np.isfinite(age) or not np.isfinite(bmd):
            raise HTTPException(400, "Invalid clinical value")

    if not data["n4_corrected"]:
        try:
            arr = n4_bias_correction(arr)
        except Exception as exc:
            raise HTTPException(400, f"N4 bias correction failed: {exc}") from exc
        data["volume"] = arr
        data["n4_corrected"] = True

    bb = get_bbox(kp, 1.5)
    crops = [crop(arr[i], bb) for i in sel]
    if any(c.size == 0 for c in crops):
        raise HTTPException(400, "ROI is empty. Check keypoints.")

    tfm = v2.Compose([
        v2.ToImage(),
        v2.Resize((224, 224), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])
    x = torch.stack([
        tfm(np.tile(c[..., None], (1, 1, 3)))
        for c in crops
    ]).to(device)

    meta = None
    if use_clinical:
        meta = torch.tensor([
            (age - AGE_MEAN) / AGE_STD,
            1 if sex.lower() == "female" else 0,
            (bmd - BMD_MEAN) / BMD_STD,
            float(pre),
            float(post),
        ], dtype=torch.float32, device=device)

    net = get_ovf_model(use_clinical)
    with torch.no_grad():
        prob = torch.stack([
            torch.sigmoid(net(x[i, None], meta[None] if meta is not None else None)).squeeze()
            for i in range(3)
        ]).mean().item()

    return {"probability": prob}
