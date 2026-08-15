# backend/pipeline/refine_mask.py
# 현재 파이프라인에선 TotalSegmentator 로 근육/뼈를 모두 얻으므로
# 정제 단계는 사용하지 않음. (호환을 위해 빈 함수 유지)
def refine_masks(data_root: str):
    # no-op (kept for compatibility)
    return
