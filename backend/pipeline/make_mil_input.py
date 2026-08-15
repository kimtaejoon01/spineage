# backend/pipeline/make_mil_input.py
import os
from glob import glob
from pathlib import Path
from tqdm import tqdm

import h5py
import numpy as np
import SimpleITK as sitk

def _resample_img(sitk_volume, new_spacing):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkLinear)
    resample.SetOutputDirection(sitk_volume.GetDirection())
    new_spacing = np.array(new_spacing)
    resample.SetOutputSpacing(new_spacing)
    old_size = np.array(sitk_volume.GetSize())
    old_spacing = np.array(sitk_volume.GetSpacing())
    new_size = np.int16(np.ceil(old_size*old_spacing/new_spacing))
    new_origin = np.array(sitk_volume.GetOrigin())
    new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetOutputOrigin(new_origin)
    return resample.Execute(sitk_volume)

def _resample_mask(sitk_volume, new_spacing, new_size=None):
    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(sitk.sitkNearestNeighbor)
    resample.SetOutputDirection(sitk_volume.GetDirection())
    new_spacing = np.array(new_spacing)
    resample.SetOutputSpacing(new_spacing)
    old_size = np.array(sitk_volume.GetSize())
    old_spacing = np.array(sitk_volume.GetSpacing())
    new_origin = np.array(sitk_volume.GetOrigin())
    if new_size is None:
        new_size = np.int16(np.ceil(old_size*old_spacing/new_spacing))
        new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetOutputOrigin(new_origin)
    return resample.Execute(sitk_volume)

def _get_roi_bbox(patch_mask, patch_size):
    fg = np.where(patch_mask)
    x_min, x_max = fg[0].min(), fg[0].max()
    y_min, y_max = fg[1].min(), fg[1].max()
    z_min, z_max = fg[2].min(), fg[2].max()
    center = [(x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2]
    x_start = int(center[0]-patch_size[0]/2); x_fin = int(center[0]+patch_size[0]/2)
    y_start = int(center[1]-patch_size[1]/2); y_fin = int(center[1]+patch_size[1]/2)
    z_start = int(center[2]-patch_size[2]/2); z_fin = int(center[2]+patch_size[2]/2)
    if center[0] < patch_size[0]/2: x_start=0; x_fin=patch_size[0]
    elif x_fin > patch_mask.shape[0]: x_start=patch_mask.shape[0]-patch_size[0]; x_fin=patch_mask.shape[0]
    if center[1] < patch_size[1]/2: y_start=0; y_fin=patch_size[1]
    elif y_fin > patch_mask.shape[1]: y_start=patch_mask.shape[1]-patch_size[1]; y_fin=patch_mask.shape[1]
    if center[2] < patch_size[2]/2: z_start=0; z_fin=patch_size[2]
    elif z_fin > patch_mask.shape[2]: z_start=patch_mask.shape[2]-patch_size[2]; z_fin=patch_size[2]
    return (x_start,y_start,z_start), (x_fin,y_fin,z_fin)

def _crop_roi(img, bbox):
    return img[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], bbox[0][2]:bbox[1][2]].copy()

def make_MIL_input_2(data_root: str, file_type: str = ".nii.gz"):
    """
    입력:  data_root/nifti/*.nii(.gz),
           data_root/tot_seg_mask/<case_id>/{vertebrae_T12..L4, autochthon_*, iliopsoas_*}.nii(.gz)
    출력:  data_root/MIL_input_2/<case_id>/<vertebrae_*.hdf5>

    규칙:
      • ROI 패치의 z-범위는 '척추체(vertebra) 마스크'만으로 결정 (근육은 z-게이트에만 참조)
      • 근육 마스크는 autochthon/iliopsoas 좌/우 4개를 OR 결합
      • 좌표계 반전 없음 (TotalSegmentator 산출물끼리 동일 기준 가정)
    """
    patch_size = (24, 128, 224)
    made = 0

    def _case_id_from_name(fname: str) -> str:
        s = fname.lower()
        if s.endswith(".nii.gz"): return fname[:-7]
        if s.endswith(".nii"):    return fname[:-4]
        return Path(fname).stem

    for img_path in tqdm(glob(os.path.join(data_root, f"nifti/*{file_type}"))):
        # 기준 이미지 1×1×3 mm 리샘플
        templete = sitk.ReadImage(img_path)
        templete = _resample_img(templete, (1., 1., 3.))
        img = sitk.GetArrayFromImage(templete)             # (Z,Y,X)

        case_id = _case_id_from_name(os.path.basename(img_path))
        ver_list = ["vertebrae_T12","vertebrae_L1","vertebrae_L2","vertebrae_L3","vertebrae_L4"]

        totseg_case_dir = os.path.join(data_root, "tot_seg_mask", case_id)

        # ── 근육(4 ROI) 결합 (OR). z-gating은 나중에 척추체별로 적용
        muscle_rois = [
            os.path.join(totseg_case_dir, f"autochthon_left{file_type}"),
            os.path.join(totseg_case_dir, f"autochthon_right{file_type}"),
            os.path.join(totseg_case_dir, f"iliopsoas_left{file_type}"),
            os.path.join(totseg_case_dir, f"iliopsoas_right{file_type}"),
        ]
        muscle_mask_all = np.zeros(img.shape, dtype=np.uint8)
        for p in muscle_rois:
            if os.path.isfile(p):
                m = _resample_mask(sitk.ReadImage(p), (1.,1.,3.), templete.GetSize())
                m = sitk.GetArrayFromImage(m).astype(np.uint8)
                muscle_mask_all |= (m > 0).astype(np.uint8)

        # ── 출력 디렉토리
        mil_case_dir    = os.path.join(data_root, "MIL_input_2",   case_id)
        refine_case_dir = os.path.join(data_root, "refine_mask_v2", case_id)  # (있으면 우선)
        os.makedirs(mil_case_dir, exist_ok=True, mode=0o777)

        for ver in ver_list:
            # 뼈 마스크: refined 우선, 없으면 tot_seg 사용
            refined_path = os.path.join(refine_case_dir, f"{ver}{file_type}")
            rawseg_path  = os.path.join(totseg_case_dir,  f"{ver}{file_type}")
            if os.path.isfile(refined_path):
                load_path = refined_path
            elif os.path.isfile(rawseg_path):
                load_path = rawseg_path
            else:
                print(f"[WARN] {case_id}: {ver} mask not found (refined/raw). skip.")
                continue

            save_path = os.path.join(mil_case_dir, f"{ver}.hdf5")
            if os.path.isfile(save_path):
                made += 1
                continue

            ver_mask = _resample_mask(sitk.ReadImage(load_path), (1.,1.,3.), templete.GetSize())
            ver_mask = sitk.GetArrayFromImage(ver_mask).astype(np.uint8)
            if ver_mask.sum() == 0:
                print(f"[WARN] {case_id}: {ver} empty mask. skip.")
                continue

            # ── z-gating: 척추체 마스크의 z 범위만 사용
            ver_z = np.where(ver_mask > 0)[0]
            z_min, z_max = int(ver_z.min()), int(ver_z.max())

            # 패치 마스크 = (ver_mask) ∪ (muscle_mask_all 의 z∈[z_min,z_max] 구간)
            patch_mask = (ver_mask > 0).astype(np.uint8)
            if muscle_mask_all.any():
                patch_mask[z_min:z_max+1] |= muscle_mask_all[z_min:z_max+1]

            # bbox & crop
            bbox = _get_roi_bbox(patch_mask, patch_size)
            img_roi    = _crop_roi(img,              bbox)
            muscle_roi = _crop_roi(muscle_mask_all,  bbox)
            ver_roi    = _crop_roi(ver_mask,         bbox)

            if img_roi.shape != patch_size:
                print(f"[WARN] {case_id}: unexpected ROI {img_roi.shape}, skip.")
                continue

            with h5py.File(save_path, "w") as f:
                f.create_dataset("image",          data=img_roi,    dtype=np.float32)
                f.create_dataset("muscle_mask",    data=muscle_roi, dtype=np.uint8)
                f.create_dataset("vertebrae_mask", data=ver_roi,    dtype=np.uint8)
            made += 1

    return made
