# -*- coding: utf-8 -*-
import io, base64
import numpy as np
from PIL import Image
import torch

def slice_uint8(a: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(a, (.05, .95))
    a = np.clip(a, lo, hi)
    a = (a - a.min()) / (a.max() - a.min() + 1e-7)
    return (a * 255).astype(np.uint8)

def to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
