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
    new_size = np.int16(np.ceil(old_size * old_spacing / new_spacing))
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
        new_size = np.int16(np.ceil(old_size * old_spacing / new_spacing))
        new_size = [int(s) for s in new_size]
    resample.SetSize(new_size)
    resample.SetOutputOrigin(new_origin)
    return resample.Execute(sitk_volume)


def _get_roi_bbox(patch_mask, patch_size):
    fg = np.where(patch_mask)
    if not fg[0].size:
        raise ValueError("ROI mask is empty")

    x_min, x_max = fg[0].min(), fg[0].max()
    y_min, y_max = fg[1].min(), fg[1].max()
    z_min, z_max = fg[2].min(), fg[2].max()
    center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]

    starts = []
    ends = []
    for axis, (c, size) in enumerate(zip(center, patch_size)):
        dim = patch_mask.shape[axis]
        if dim < size:
            raise ValueError(f"Volume axis {axis} ({dim}) is smaller than requested patch size ({size})")
        start = int(round(c - size / 2))
        start = max(0, min(start, dim - size))
        starts.append(start)
        ends.append(start + size)

    return tuple(starts), tuple(ends)


def _crop_roi(img, bbox):
    return img[
        bbox[0][0]:bbox[1][0],
        bbox[0][1]:bbox[1][1],
        bbox[0][2]:bbox[1][2],
    ].copy()


def make_MIL_input_2(data_root: str, file_type: str = ".nii.gz"):
    """
    입력:  data_root/nifti/*.nii(.gz),
           data_root/tot_seg_mask/<case_id>/{vertebrae_T12..L4, autochthon_*, iliopsoas_*}.nii(.gz)
    출력:  data_root/MIL_input_2/<case_id>/<vertebrae_*.hdf5>
    """
    patch_size = (24, 128, 224)
    made = 0

    def _case_id_from_name(fname: str) -> str:
        s = fname.lower()
        if s.endswith(".nii.gz"):
            return fname[:-7]
        if s.endswith(".nii"):
            return fname[:-4]
        return Path(fname).stem

    for img_path in tqdm(glob(os.path.join(data_root, f"nifti/*{file_type}"))):
        template = sitk.ReadImage(img_path)
        template = _resample_img(template, (1.0, 1.0, 3.0))
        img = sitk.GetArrayFromImage(template)

        case_id = _case_id_from_name(os.path.basename(img_path))
        ver_list = ["vertebrae_T12", "vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4"]
        totseg_case_dir = os.path.join(data_root, "tot_seg_mask", case_id)

        muscle_rois = [
            os.path.join(totseg_case_dir, f"autochthon_left{file_type}"),
            os.path.join(totseg_case_dir, f"autochthon_right{file_type}"),
            os.path.join(totseg_case_dir, f"iliopsoas_left{file_type}"),
            os.path.join(totseg_case_dir, f"iliopsoas_right{file_type}"),
        ]
        muscle_mask_all = np.zeros(img.shape, dtype=np.uint8)
        for p in muscle_rois:
            if os.path.isfile(p):
                m = _resample_mask(sitk.ReadImage(p), (1.0, 1.0, 3.0), template.GetSize())
                m = sitk.GetArrayFromImage(m).astype(np.uint8)
                muscle_mask_all |= (m > 0).astype(np.uint8)

        mil_case_dir = os.path.join(data_root, "MIL_input_2", case_id)
        refine_case_dir = os.path.join(data_root, "refine_mask_v2", case_id)
        os.makedirs(mil_case_dir, exist_ok=True, mode=0o777)

        for ver in ver_list:
            refined_path = os.path.join(refine_case_dir, f"{ver}{file_type}")
            rawseg_path = os.path.join(totseg_case_dir, f"{ver}{file_type}")
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

            ver_mask = _resample_mask(sitk.ReadImage(load_path), (1.0, 1.0, 3.0), template.GetSize())
            ver_mask = sitk.GetArrayFromImage(ver_mask).astype(np.uint8)
            if ver_mask.sum() == 0:
                print(f"[WARN] {case_id}: {ver} empty mask. skip.")
                continue

            ver_z = np.where(ver_mask > 0)[0]
            z_min, z_max = int(ver_z.min()), int(ver_z.max())

            patch_mask = (ver_mask > 0).astype(np.uint8)
            if muscle_mask_all.any():
                patch_mask[z_min:z_max + 1] |= muscle_mask_all[z_min:z_max + 1]

            try:
                bbox = _get_roi_bbox(patch_mask, patch_size)
            except ValueError as exc:
                print(f"[WARN] {case_id}: {ver} ROI error: {exc}. skip.")
                continue

            img_roi = _crop_roi(img, bbox)
            muscle_roi = _crop_roi(muscle_mask_all, bbox)
            ver_roi = _crop_roi(ver_mask, bbox)

            if img_roi.shape != patch_size:
                print(f"[WARN] {case_id}: unexpected ROI {img_roi.shape}, skip.")
                continue

            with h5py.File(save_path, "w") as f:
                f.create_dataset("image", data=img_roi, dtype=np.float32)
                f.create_dataset("muscle_mask", data=muscle_roi, dtype=np.uint8)
                f.create_dataset("vertebrae_mask", data=ver_roi, dtype=np.uint8)
            made += 1

    return made
