# backend/preprocess.py
import numpy as np
import pydicom
import SimpleITK as sitk


def _all_floatable(values):
    try:
        return [float(v) for v in values]
    except (TypeError, ValueError):
        return None


def _sort_dicom_entries(entries):
    """Sort DICOM slices using spatial metadata when available.

    Priority:
      1) ImagePositionPatient (projected onto the slice normal when possible)
      2) SliceLocation
      3) InstanceNumber
      4) original upload order
    """
    if len(entries) < 2:
        return entries

    datasets = [ds for _, ds in entries]

    # Best case: use patient-space position projected onto the slice normal.
    try:
        if all(hasattr(ds, "ImagePositionPatient") for ds in datasets):
            positions = np.asarray(
                [[float(x) for x in ds.ImagePositionPatient[:3]] for ds in datasets],
                dtype=np.float64,
            )

            if all(hasattr(ds, "ImageOrientationPatient") for ds in datasets):
                orient = np.asarray(
                    [float(x) for x in datasets[0].ImageOrientationPatient[:6]],
                    dtype=np.float64,
                )
                row = orient[:3]
                col = orient[3:]
                normal = np.cross(row, col)
                norm = np.linalg.norm(normal)
                if norm > 1e-8:
                    normal /= norm
                    projected = positions @ normal
                    if np.ptp(projected) > 1e-8:
                        order = np.argsort(projected, kind="stable")
                        return [entries[int(i)] for i in order]

            # If orientation is unavailable, sort along the patient-space axis
            # with the largest positional spread.
            spread = np.ptp(positions, axis=0)
            axis = int(np.argmax(spread))
            if spread[axis] > 1e-8:
                order = np.argsort(positions[:, axis], kind="stable")
                return [entries[int(i)] for i in order]
    except (TypeError, ValueError, IndexError):
        pass

    slice_locations = _all_floatable(
        [getattr(ds, "SliceLocation", None) for ds in datasets]
    )
    if slice_locations is not None and all(v is not None for v in slice_locations):
        return [
            item for _, item in sorted(
                zip(slice_locations, entries), key=lambda pair: pair[0]
            )
        ]

    instance_numbers = _all_floatable(
        [getattr(ds, "InstanceNumber", None) for ds in datasets]
    )
    if instance_numbers is not None and all(v is not None for v in instance_numbers):
        return [
            item for _, item in sorted(
                zip(instance_numbers, entries), key=lambda pair: pair[0]
            )
        ]

    return entries


def read_dicom_files(files):
    """Read a single-frame DICOM series into a float32 array (F, H, W)."""
    entries = []
    for index, f in enumerate(files):
        f.file.seek(0)
        ds = pydicom.dcmread(f.file, force=True)
        entries.append((index, ds))

    if not entries:
        raise ValueError("No DICOM files")

    entries = _sort_dicom_entries(entries)
    slices = []
    expected_shape = None

    for _, ds in entries:
        pixel = np.asarray(ds.pixel_array)
        if pixel.ndim != 2:
            raise ValueError(
                f"Only single-frame 2D DICOM slices are supported; got shape {pixel.shape}"
            )

        if expected_shape is None:
            expected_shape = pixel.shape
        elif pixel.shape != expected_shape:
            raise ValueError(
                f"DICOM slice shape mismatch: expected {expected_shape}, got {pixel.shape}"
            )

        slope_raw = getattr(ds, "RescaleSlope", None)
        intercept_raw = getattr(ds, "RescaleIntercept", None)
        slope = 1.0 if slope_raw is None else float(slope_raw)
        intercept = 0.0 if intercept_raw is None else float(intercept_raw)

        scaled = pixel.astype(np.float32) * slope + intercept
        slices.append(scaled)

    return np.stack(slices, axis=0).astype(np.float32, copy=False)


def n4_bias_correction(arr: np.ndarray) -> np.ndarray:
    img = sitk.GetImageFromArray(arr)
    img = sitk.Cast(img, sitk.sitkFloat32)

    mask = sitk.OtsuThreshold(img, 0, 1)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrector.SetMaximumNumberOfIterations([50, 50, 30, 20])
    out = corrector.Execute(img, mask)

    return sitk.GetArrayFromImage(out)


def get_bbox(kp: np.ndarray, scale: float = 1.5):
    y = kp[:, 0]
    x = kp[:, 1]
    cy, cx = y.mean(), x.mean()
    h = (y.max() - y.min()) * scale
    w = (x.max() - x.min()) * scale
    y0 = int(max(cy - h / 2, 0))
    x0 = int(max(cx - w / 2, 0))
    return (y0, x0, int(h), int(w))


def crop(arr_sl: np.ndarray, bb):
    y0, x0, h, w = bb
    return arr_sl[y0:y0 + h, x0:x0 + w]
