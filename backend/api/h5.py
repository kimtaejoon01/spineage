# -*- coding: utf-8 -*-
import json
import tempfile
from pathlib import Path
from uuid import uuid4
from typing import Dict

import numpy as np
from torch.serialization import add_safe_globals
import torch
import h5py
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query

from multitask.model import MultiTaskLearner
from utils import slice_uint8, to_b64, device

router = APIRouter(tags=["HDF5 (Multitask)"])

h5_sessions: Dict[str, dict] = {}
h5_models: Dict[str, MultiTaskLearner] = {}
H5_CKPT = Path(__file__).resolve().parent.parent / "models" / "checkpoint_best.pth"


def get_h5_model(args_json: str) -> MultiTaskLearner:
    if args_json not in h5_models:
        if not H5_CKPT.exists():
            raise FileNotFoundError(f"HDF5 model checkpoint not found: {H5_CKPT}")

        try:
            add_safe_globals([np.dtype, np.core.multiarray.scalar])
        except Exception:
            pass

        try:
            ckpt = torch.load(H5_CKPT, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(H5_CKPT, map_location=device)

        from types import SimpleNamespace
        args = SimpleNamespace(**json.loads(args_json))
        net = MultiTaskLearner(args).to(device)

        state = ckpt.get("state_dict", ckpt)
        try:
            net.load_state_dict(state)
        except Exception:
            encoder_dict = {f"encoder.{k}": v for k, v in state.items() if k.startswith("backbone")}
            decoder_dict = {
                f"decoders.cls_decoder.{k.replace('classifier.', '')}": v
                for k, v in state.items() if k.startswith("classifier")
            }
            net.load_state_dict({**encoder_dict, **decoder_dict}, strict=False)

        net.eval()
        h5_models[args_json] = net
    return h5_models[args_json]


def normalize_like_script(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0, 1000).astype(np.float32)
    vmax, vmin = img.max(), img.min()
    scale = max(vmax - vmin, 1e-6)
    return (img - vmin) / scale


def make_bone_only(image: np.ndarray, muscle_mask: np.ndarray, verte_mask: np.ndarray,
                   mode: str = "bone_only") -> np.ndarray:
    img = normalize_like_script(image)
    muscle_mask = muscle_mask.astype(np.uint8)
    verte_mask = verte_mask.astype(np.uint8)
    muscle_mask[verte_mask == 1] = 0

    if mode == "bone_only":
        ver_img = img.copy()
        ver_img[verte_mask == 0] = 0
        return ver_img
    if mode == "bone_muscle":
        ver_img = img.copy()
        mus_img = img.copy()
        ver_img[verte_mask == 0] = 0
        mus_img[muscle_mask == 0] = 0
        return ver_img + mus_img
    return img


def pick_stack_centers(vol: np.ndarray, k: int = 3) -> list[int]:
    """Deterministically choose k valid stack centers from the foreground range."""
    D = vol.shape[0]
    if D < 3:
        raise ValueError("Volume must contain at least 3 slices")

    nonzero = np.where(vol.sum((1, 2)) > 1e-6)[0]
    valid = nonzero[(nonzero >= 1) & (nonzero <= D - 2)]

    if len(valid) == 0:
        valid = np.arange(1, D - 1)
    if len(valid) == 0:
        raise ValueError("No valid stack center")

    if len(valid) >= k:
        positions = np.linspace(0, len(valid) - 1, num=k)
        picks = [int(valid[int(round(p))]) for p in positions]
    else:
        picks = [int(v) for v in valid]
        while len(picks) < k:
            picks.append(picks[len(picks) % len(valid)])

    return sorted(picks[:k])


def build_input_tensor_from_three_vertebra(item_list: list[dict],
                                           bone_only: str = "bone_only",
                                           channels_num_2d: int = 3) -> torch.Tensor:
    if len(item_list) != 3:
        raise ValueError("Exactly 3 vertebra items are required")
    if channels_num_2d != 3:
        raise ValueError("This checkpoint expects channels_num_2d=3")

    stacks = []
    for it in item_list:
        vol = make_bone_only(it["image"], it["muscle"], it["vertebrae"], mode=bone_only)
        centers = pick_stack_centers(vol, k=channels_num_2d)
        for i in centers:
            stacks.append(vol[i - 1:i + 2])

    arr = np.asarray(stacks, dtype=np.float32)
    if arr.shape[0] != 9:
        raise ValueError(f"Expected 9 stacks, got {arr.shape[0]}")
    return torch.from_numpy(arr).unsqueeze(0)


@router.post("/load_h5_multi")
async def load_h5_multi(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, "No files")

    sid = str(uuid4())
    items: Dict[str, dict] = {}

    for idx, f in enumerate(files):
        try:
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                tmp.write(await f.read())
                tmp.flush()
                with h5py.File(tmp.name, "r") as h:
                    for key in ("image", "muscle_mask", "vertebrae_mask"):
                        if key not in h:
                            raise ValueError(f"Missing HDF5 dataset: {key}")
                    img = h["image"][()]
                    mus = h["muscle_mask"][()]
                    ver = h["vertebrae_mask"][()]
        except Exception as exc:
            raise HTTPException(400, f"Invalid HDF5 file {f.filename}: {exc}") from exc

        if img.ndim != 3 or img.shape != mus.shape or img.shape != ver.shape:
            raise HTTPException(400, f"Invalid volume shape in {f.filename}")

        fid = str(idx)
        items[fid] = dict(name=f.filename, image=img, muscle=mus, vertebrae=ver)

    h5_sessions[sid] = dict(items=items)

    thumbs = []
    for fid, it in items.items():
        mid = it["image"].shape[0] // 2
        thumbs.append(dict(
            fid=fid,
            name=it["name"],
            preview=dict(
                img=to_b64(slice_uint8(it["image"][mid])),
                muscle=to_b64(slice_uint8(it["muscle"][mid] * 255)),
                vertebra=to_b64(slice_uint8(it["vertebrae"][mid] * 255)),
            ),
        ))
    return {"session": sid, "items": thumbs}


@router.get("/h5_volume")
async def h5_volume(session: str = Query(...), fid: str = Query(...)):
    if session not in h5_sessions:
        raise HTTPException(400, "Bad session")
    items = h5_sessions[session]["items"]
    if fid not in items:
        raise HTTPException(400, "Bad fid")

    it = items[fid]
    img, mus, ver = it["image"], it["muscle"], it["vertebrae"]
    D = img.shape[0]
    volume = dict(
        img=[to_b64(slice_uint8(img[i])) for i in range(D)],
        muscle=[to_b64(slice_uint8(mus[i] * 255)) for i in range(D)],
        vertebra=[to_b64(slice_uint8(ver[i] * 255)) for i in range(D)],
        name=it["name"],
    )
    return {"volume": volume}


@router.post("/predict_h5_multi")
async def predict_h5_multi(
    session: str = Form(...),
    selected: str = Form(...),
    args_json: str = Form(...),
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

    try:
        args = json.loads(args_json)
    except json.JSONDecodeError as exc:
        raise HTTPException(400, "Invalid args_json") from exc

    bone_only = args.get("bone_only", "bone_only")
    channels_num_2d = int(args.get("channels_num_2d", 3))
    max_followup = int(args.get("max_followup", 10))
    cls_enabled = bool(args.get("cls_enabled", True))
    pred_enabled = bool(args.get("pred_enabled", True))

    if channels_num_2d != 3:
        raise HTTPException(400, "This model currently supports Channels=3 only")
    if not (cls_enabled or pred_enabled):
        raise HTTPException(400, "Choose at least one task (CLS or PRED).")

    try:
        item_list = [items[fid] for fid in fids]
        x = build_input_tensor_from_three_vertebra(item_list, bone_only, channels_num_2d).to(device)
        net = get_h5_model(args_json)
        with torch.no_grad():
            sample = {
                "vers_img_tensor": x,
                "y": torch.tensor([0], dtype=torch.long, device=device),
                "time_at_event": torch.tensor([0], dtype=torch.long, device=device),
            }
            cls_out, pred_out, _ = net(sample, sample)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(400, str(exc)) from exc

    result = {}
    if cls_enabled and ("prob" in cls_out and cls_out["prob"] is not None):
        result["cls_probability"] = float(cls_out["prob"][0, 1].item())

    if pred_enabled:
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
