#!/bin/bash
# UniTalker 학습 스크립트

# Conda 환경 활성화 (trainframework / unitalker5090 / unitalker)
source ~/miniconda3/etc/profile.d/conda.sh

# 사용 가능한 환경 확인 후 활성화
if conda env list | grep -q "trainframework"; then
    conda activate trainframework
elif conda env list | grep -q "unitalker5090"; then
    conda activate unitalker5090
elif conda env list | grep -q "unitalker"; then
    conda activate unitalker
else
    echo "Error: trainframework, unitalker, or unitalker5090 environment not found"
    exit 1
fi

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

# 학습 실행
python3 models/unitalker/train.py --config "$@"
