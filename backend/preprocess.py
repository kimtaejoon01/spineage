# backend/preprocess.py
import numpy as np
import pydicom
import SimpleITK as sitk

# ───── DICOM 시리즈 읽기 (UploadFile 리스트 → numpy) ─────
def read_dicom_files(files):
    dicoms = []
    for f in files:
        f.file.seek(0)
        d = pydicom.dcmread(f.file, force=True)
        dicoms.append(d)

    # 인스턴스 번호 순 정렬
    dicoms.sort(key=lambda d: int(d.InstanceNumber))

    slices = [d.pixel_array * d.RescaleSlope + d.RescaleIntercept for d in dicoms]
    return np.stack(slices).astype(np.float32)      # (F,H,W)

# ───── N4 bias correction ─────
def n4_bias_correction(arr: np.ndarray) -> np.ndarray:
    img = sitk.GetImageFromArray(arr)
    img = sitk.Cast(img, sitk.sitkFloat32)

    mask = sitk.OtsuThreshold(img, 0, 1)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    out = corrector.Execute(img, mask)

    return sitk.GetArrayFromImage(out)              # (F,H,W) float32

# ───── ROI 도우미 ─────
def get_bbox(kp: np.ndarray, scale: float = 1.5):
    y = kp[:,0]; x = kp[:,1]
    cy, cx = y.mean(), x.mean()
    h = (y.max()-y.min()) * scale
    w = (x.max()-x.min()) * scale
    y0 = int(max(cy - h/2, 0))
    x0 = int(max(cx - w/2, 0))
    return (y0, x0, int(h), int(w))         # (top,left,height,width)

def crop(arr_sl: np.ndarray, bb):
    y0,x0,h,w = bb
    return arr_sl[y0:y0+h, x0:x0+w]