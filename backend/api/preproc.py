# backend/api/preproc.py
# -*- coding: utf-8 -*-
import os, shutil, tempfile, time
from glob import glob
from pathlib import Path
from uuid import uuid4
from typing import Dict, List

import h5py
import SimpleITK as sitk
import numpy as np
import torch, json
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query

from utils import slice_uint8, to_b64, device
from pipeline.make_mil_input import make_MIL_input_2
from .h5 import get_h5_model, build_input_tensor_from_three_vertebra

# TotalSegmentator Python API
from totalsegmentator.python_api import totalsegmentator

router = APIRouter(tags=["Preprocess (DICOM/NIfTI → HDF5)"])

WORKDIR = Path(__file__).resolve().parent.parent / "workspace"
WORKDIR.mkdir(exist_ok=True, parents=True)

VER_NAMES = ["vertebrae_T12","vertebrae_L1","vertebrae_L2","vertebrae_L3","vertebrae_L4"]

def _case_id_from_name(fname: str) -> str:
    s = fname.lower()
    if s.endswith(".nii.gz"): return fname[:-7]
    if s.endswith(".nii"):    return fname[:-4]
    return Path(fname).stem

def _dicom_series_to_nifti(dicom_files: List[UploadFile], out_path: Path):
    tmpdir = Path(tempfile.mkdtemp())
    try:
        for i, f in enumerate(dicom_files):
            with open(tmpdir / f"{i:05d}.dcm", "wb") as g:
                g.write(f.file.read())
        reader = sitk.ImageSeriesReader()
        series_IDs = reader.GetGDCMSeriesIDs(str(tmpdir))
        if not series_IDs:
            raise HTTPException(400, "No DICOM series found")
        file_names = reader.GetGDCMSeriesFileNames(str(tmpdir), series_IDs[0])
        reader.SetFileNames(file_names)
        image = reader.Execute()
        sitk.WriteImage(image, str(out_path))
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

def _run_totalseg(nifti_path: Path, out_dir: Path):
    """
    TotalSegmentator (필수, Python API)
    - roi_subset: 뼈(T12~L4) + 근육(autochthon/iliopsoas 좌/우)만 빠르게 분할
    - fast=True
    """
    TARGET_ROIS = [
        # Vertebrae bodies
        "vertebrae_T12","vertebrae_L1","vertebrae_L2","vertebrae_L3","vertebrae_L4",
        # Muscles
        "autochthon_left","autochthon_right","iliopsoas_left","iliopsoas_right",
    ]
    try:
        start = time.time()
        totalsegmentator(
            input=str(nifti_path),
            output=str(out_dir),
            roi_subset=TARGET_ROIS,
            fast=True,
            statistics=False,
            verbose=False
        )
        elapsed = time.time() - start
        print(f"✅ TotalSegmentator 완료: {elapsed:.2f}s (roi_subset {len(TARGET_ROIS)})")
    except Exception as e:
        raise HTTPException(400, f"TotalSegmentator 실행 실패: {type(e).__name__}: {e}")

def _check_vertebra_masks_exist(root: Path, case_id: str, file_type: str) -> bool:
    out_seg_case = root / "tot_seg_mask" / case_id
    for vn in VER_NAMES:
        if (out_seg_case / f"{vn}{file_type}").exists():
            return True
    return False

def _collect_h5_items(data_root: Path):
    items = []
    for h5 in glob(str(data_root / "MIL_input_2/*/*.hdf5")):
        with h5py.File(h5, "r") as f:
            img = f["image"][()]; mus = f["muscle_mask"][()]; ver = f["vertebrae_mask"][()]
        mid = img.shape[0] // 2
        items.append(dict(
            path=h5,
            name=os.path.basename(h5),
            preview=dict(
                img      = to_b64(slice_uint8(img[mid])),
                muscle   = to_b64(slice_uint8(mus[mid]*255)),
                vertebra = to_b64(slice_uint8(ver[mid]*255)),
            )
        ))
    return items

@router.post("/preproc/nifti")
async def ingest_nifti(
    file: UploadFile = File(...),
    file_type: str = Form(".nii.gz"),
):
    session = str(uuid4())
    root = WORKDIR / session
    (root / "nifti").mkdir(parents=True, exist_ok=True)

    nifti_path = root / "nifti" / file.filename
    with open(nifti_path, "wb") as g:
        g.write(await file.read())

    case_id = _case_id_from_name(nifti_path.name)

    # 1) TotalSegmentator (필수)
    out_seg = root / "tot_seg_mask" / case_id
    out_seg.mkdir(parents=True, exist_ok=True)
    _run_totalseg(nifti_path, out_seg)
    if not _check_vertebra_masks_exist(root, case_id, file_type):
        raise HTTPException(400, "TotalSegmentator produced no vertebra masks. Check input.")

    # 2) MIL_input_2 (근육은 TotalSeg 결과로 결합)
    made = make_MIL_input_2(str(root), file_type)
    if made == 0:
        raise HTTPException(400, "No HDF5 generated after pipeline.")

    items = _collect_h5_items(root)
    if not items:
        raise HTTPException(400, "No HDF5 generated after pipeline (collect stage).")

    PREPROC_SESSIONS[session] = dict(items={str(i): it for i, it in enumerate(items)},
                                     muscle_available=True)
    thumbs = [{"fid": str(i), "name": it["name"], "preview": it["preview"]} for i, it in enumerate(items)]
    return {"session": session, "items": thumbs, "muscle_available": True}

@router.post("/preproc/dicom")
async def ingest_dicom(
    files: list[UploadFile] = File(...),
    file_type: str = Form(".nii.gz"),
):
    if not files:
        raise HTTPException(400, "No DICOMs")
    session = str(uuid4())
    root = WORKDIR / session
    (root / "nifti").mkdir(parents=True, exist_ok=True)

    nifti_path = root / "nifti" / "case.nii.gz"
    _dicom_series_to_nifti(files, nifti_path)
    case_id = _case_id_from_name(nifti_path.name)

    out_seg = root / "tot_seg_mask" / case_id
    out_seg.mkdir(parents=True, exist_ok=True)
    _run_totalseg(nifti_path, out_seg)
    if not _check_vertebra_masks_exist(root, case_id, file_type):
        raise HTTPException(400, "TotalSegmentator produced no vertebra masks. Check input.")

    made = make_MIL_input_2(str(root), file_type)
    if made == 0:
        raise HTTPException(400, "No HDF5 generated after pipeline.")

    items = _collect_h5_items(root)
    if not items:
        raise HTTPException(400, "No HDF5 generated after pipeline (collect stage).")

    PREPROC_SESSIONS[session] = dict(items={str(i): it for i, it in enumerate(items)},
                                     muscle_available=True)
    thumbs = [{"fid": str(i), "name": it["name"], "preview": it["preview"]} for i, it in enumerate(items)]
    return {"session": session, "items": thumbs, "muscle_available": True}

# ===== h5 페이지와 호환되는 미리보기/예측 브리지 =====

PREPROC_SESSIONS: Dict[str, dict] = {}

@router.get("/preproc/volume")
async def preproc_volume(session: str = Query(...), fid: str = Query(...)):
    if session not in PREPROC_SESSIONS:
        raise HTTPException(400, "Bad session")
    items = PREPROC_SESSIONS[session]["items"]
    if fid not in items:
        raise HTTPException(400, "Bad fid")
    it = items[fid]
    with h5py.File(it["path"], "r") as f:
        img = f["image"][()]; mus = f["muscle_mask"][()]; ver = f["vertebrae_mask"][()]
    D = img.shape[0]
    volume = dict(
        img     = [to_b64(slice_uint8(img[i])) for i in range(D)],
        muscle  = [to_b64(slice_uint8(mus[i]*255)) for i in range(D)],
        vertebra= [to_b64(slice_uint8(ver[i]*255)) for i in range(D)],
        name    = it["name"],
    )
    return {"volume": volume}

@router.post("/preproc/predict")
async def preproc_predict(
    session: str   = Form(...),
    selected: str  = Form(...),   # "fid1,fid2,fid3"
    args_json: str = Form(...),
):
    if session not in PREPROC_SESSIONS:
        raise HTTPException(400, "Bad session")
    items = PREPROC_SESSIONS[session]["items"]

    fids = [s for s in selected.split(",") if s]
    if len(fids) != 3:
        raise HTTPException(400, "Select exactly 3 vertebra files")
    for fid in fids:
        if fid not in items:
            raise HTTPException(400, f"Bad fid: {fid}")

    args = json.loads(args_json)
    # muscle_available=True 전제이므로 bone_muscle도 허용
    item_list = []
    for fid in fids:
        p = items[fid]["path"]
        with h5py.File(p, "r") as f:
            item_list.append(dict(
                image=f["image"][()],
                muscle=f["muscle_mask"][()],
                vertebrae=f["vertebrae_mask"][()],
                name=os.path.basename(p),
            ))
    bone_only        = args.get("bone_only", "bone_only")
    channels_num_2d  = int(args.get("channels_num_2d", 3))
    cls_enabled      = bool(args.get("cls_enabled", True))
    pred_enabled     = bool(args.get("pred_enabled", True))
    if not (cls_enabled or pred_enabled):
        raise HTTPException(400, "Choose at least one task (CLS or PRED).")

    x = build_input_tensor_from_three_vertebra(item_list, bone_only, channels_num_2d).to(device)
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
        result["cls_probability"] = float(cls_out["prob"][0,1].item())
    if pred_enabled:
        if "prob" in pred_out and pred_out["prob"] is not None:
            curve = pred_out["prob"][0].detach().cpu().tolist()
        elif "logit" in pred_out and pred_out["logit"] is not None:
            curve = torch.sigmoid(pred_out["logit"])[0].detach().cpu().tolist()
        else:
            curve = []
        result["pred_curve"] = curve
    result["selected_names"] = [items[f]["name"] for f in fids]
    result["muscle_available"] = True
    return result
