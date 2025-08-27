# features.py  (원본 세트 고정 버전)
# -*- coding: utf-8 -*-
from config import NUM_KEPT, CAT_COLS, TARGET, GROUP_COL

# 학습에서 제외할 명시적 열들 (ID/로그성 등)
EXCLUDE_COLS = ["cell_id"]

def get_features(df):
    """
    원본(Original) 비교용: 오직 NUM_KEPT + CAT_COLS 만 사용.
    이전과 동일한 '원본' 기준을 보장하기 위해 자동 탐지를 쓰지 않는다.
    """
    # 안전 차단: TARGET, GROUP, EXCLUDE 제거
    use_num = [c for c in NUM_KEPT if c in df.columns]
    use_cat = [c for c in CAT_COLS if c in df.columns]
    X = df[use_num + use_cat].copy()
    y = df[TARGET].astype(float)
    groups = df[GROUP_COL]
    return X, y, groups, use_num, use_cat
