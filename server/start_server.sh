#!/bin/bash

# FastAPI 서버 실행 스크립트

# 가상환경 활성화 (필요한 경우)
# conda activate unitalker5090

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

# 로컬 추론 활성화 (외부 inference 서버 프록시 대신 로컬 GPU 사용)
export USE_LOCAL_INFERENCE=1

# 서버 실행
# host="0.0.0.0"으로 설정하여 다른 PC에서 접근 가능하도록
python -m uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
