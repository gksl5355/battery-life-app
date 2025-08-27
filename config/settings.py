# config/settings.py
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# LLM 설정 (필요에 맞게 교체)
# OpenAI 예시: OPENAI_API_KEY 환경변수 필요
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")  # 가벼운 기본치
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# 보고서 기본 옵션
TOPK_FEATS = int(os.getenv("TOPK_FEATS", "8"))
