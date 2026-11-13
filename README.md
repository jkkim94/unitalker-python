# unitalker-python

SSH 학습/추론 서버 코드를 로컬 FaceTool과 맞추기 위한 미러입니다.  
가중치·데이터셋·DNA는 포함하지 않습니다.

## FaceTool 연동

- 계약: [`docs/FACETOOL_SERVICE_CONTRACT.md`](docs/FACETOOL_SERVICE_CONTRACT.md)
- API: [`docs/FRONTEND_INFERENCE_API.md`](docs/FRONTEND_INFERENCE_API.md)
- 서버: `server/main.py` (`USE_LOCAL_INFERENCE=1`, 기본 포트 8000)

```bash
export USE_LOCAL_INFERENCE=1
bash server/start_server.sh
# 또는: python -m uvicorn server.main:app --host 0.0.0.0 --port 8000
```

## 구조

```
├── server/           # FastAPI 추론 API
├── models/unitalker/ # 학습·추론 코드 (weights 제외)
├── configs/
├── core/
├── docs/             # FaceTool 계약·API
└── scripts/
```
