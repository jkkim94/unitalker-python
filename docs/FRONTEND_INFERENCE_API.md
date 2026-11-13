# 프론트엔드 — 오디오 추론 API 연동 가이드

UniTalker 학습 서버(`server/main.py`)와 프론트 연동 규약입니다.

**FaceTool 준수 계약(필수):** [`FACETOOL_SERVICE_CONTRACT.md`](FACETOOL_SERVICE_CONTRACT.md)  
GT 정책: [`FACETOOL_EVAL_BACKEND_SCOPE.md`](FACETOOL_EVAL_BACKEND_SCOPE.md)

---

## 1. 요약

| 구분 | `output_kind` | run 예 | 출력 채널 | 프론트 사용 |
|------|---------------|--------|-----------|-------------|
| ARKit | `arkit_51` | `pretrained` | **51** | GLB morph (`blendshapes_glb`) |
| MetaHuman 260ch | `metahuman_260` | `GET /api/runs`의 MH run | **260** | RigLogic (이름 바인딩) |

- 모델 타입은 **별도 토글 없이** run 메타/`output_kind`로 결정합니다.
- FaceTool은 run_id 패턴으로 목록을 필터하지 않습니다. 계약 문서 MUST 참고.

---

## 2. 환경

| 변수 | 설명 |
|------|------|
| `USE_LOCAL_INFERENCE=1` | 로컬 UniTalker 추론 (권장). `0`이면 `NEW_API_URL` 프록시 |
| `NEW_API_URL` | 외부 추론 서버 (로컬 비활성 시) |

로컬 추론 시 **`run` 필수**.

---

## 3. API 목록

### 3.1 학습 run 목록

```http
GET /api/runs
```

**응답 예**

```json
{
  "success": true,
  "baseDir": "/path/to/models/unitalker",
  "runs": [
    {
      "id": "pretrained",
      "path": "...",
      "blendshapePath": "...",
      "epochCount": 0,
      "hasBlendshapeSamples": false,
      "label": "UniTalker official (D0-D7)",
      "strategy": "PRETRAINED",
      "output_kind": "arkit_51",
      "channels": 51,
      "dataset": "D3_3DETF",
      "engines": { "pytorch": true, "onnx": true },
      "dnaId": null,
      "hasCheckpoint": true,
      "isDefault": false
    },
    {
      "id": "results_D8_control_pretrained",
      "label": "D8 MetaHuman FT control-MSE",
      "strategy": "PRETRAINED",
      "output_kind": "metahuman_260",
      "channels": 260,
      "dataset": "D8",
      "engines": { "pytorch": true, "onnx": true },
      "dnaId": "0421_V33",
      "hasCheckpoint": true,
      "isDefault": false
    }
  ]
}
```

**프론트 UI**

- `label`을 드롭다운에 표시 (없으면 `id`).
- `output_kind`로만 그룹 분리 (ARKit / MetaHuman).
- `hasCheckpoint === false` 는 비활성.
- 기본 선택: `isDefault === true`, 없으면 원하는 `output_kind`의 첫 항목.

### 3.2 체크포인트 목록

```http
GET /api/checkpoints?run={run_id}
```

일반 학습 run: `best.pt` 등. `pretrained` run: `UniTalker-B-D0-D7.pt` 기본.

### 3.3 화자(identity) 수

```http
GET /api/speakers?run={run_id}
```

MetaHuman 단일 화자 데이터셋은 보통 `count: 1` → `speaker: 0`.

---

## 4. 추론 (핵심)

```http
POST /api/inference
Content-Type: application/json
```

### 4.1 요청 body

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `audio_base64` | string | **예** | 16kHz mono WAV 등 |
| `run` | string | 로컬 시 **예** | `GET /api/runs`의 `id` |
| `checkpoint` | string | 아니오 | 기본 `best.pt` |
| `speaker` | int | 아니오 | 기본 `0` |
| `engine` | string | 아니오 | `pytorch`(기본) \| `onnx` |
| `enable_eye_blink` | bool | 아니오 | 프록시 경로 |
| `source_audio_filename` | string | 아니오 | 디버그용. **GT 비교 미사용** |
| `compare_vanilla` | bool | 아니오 | Vanilla 비교 추론 |

**ARKit 요청 예**

```json
{
  "audio_base64": "<base64>",
  "run": "pretrained",
  "checkpoint": "UniTalker-B-D0-D7.pt",
  "speaker": 0
}
```

**MetaHuman 요청 예**

```json
{
  "audio_base64": "<base64>",
  "run": "results_D8_control_pretrained",
  "checkpoint": "best.pt",
  "speaker": 0,
  "engine": "pytorch"
}
```

### 4.2 성공 응답

| 필드 | 설명 |
|------|------|
| `output_kind` | `"arkit_51"` \| `"metahuman_260"` |
| `channels` | `data[0].length` (51 또는 260) |
| `blendshapes` | 채널 이름 |
| `blendshapes_glb` | GLB용 축약 이름. **ARKit만** |
| `data` | `float[][]` `[frames][channels]` |
| `fps` | 30 |
| `frames` | `data.length` |
| `engine` | 실제 사용한 `pytorch` \| `onnx` |

**GT는 응답에 포함하지 않습니다.**

### 4.3 ARKit 51

- `run: "pretrained"` → 52ch head 추론 후 tongueOut 제거 → 51ch.
- GLB: `blendshapes_glb` 순서와 `data` 열 1:1.

### 4.4 MetaHuman 260

- `blendshapes`: MetaHuman rig control 컬럼명 (`feature_names.txt`와 동일).
- `data`: `[frames][260]` — RigLogic에 260열 전체, **이름 바인딩**.
- `blendshapes_glb` 사용하지 않음.

### 4.5 프론트 분기 (계약)

```typescript
function detectOutputKind(res: InferenceResponse): "arkit_51" | "metahuman_260" {
  return res.output_kind;
}

if (kind === "arkit_51") {
  applyToGlbMorph(res.blendshapes_glb, res.data);
} else if (kind === "metahuman_260") {
  sendToRigLogicByName(res.blendshapes, res.data);
}
```

run_id 문자열로 kind를 추정하지 마십시오.

### 4.6 에러

| HTTP | 상황 |
|------|------|
| 400 | `audio_base64` 없음, 로컬 추론 시 `run` 없음, 잘못된 `engine` |
| 404 | 체크포인트/ONNX 없음 |
| 500 | 추론 실패 |

---

## 5. 레거시 뷰어 API (FaceTool GT 평가 비대상)

상세: [`FACETOOL_EVAL_BACKEND_SCOPE.md`](FACETOOL_EVAL_BACKEND_SCOPE.md)

| API | FaceTool 평가 |
|-----|---------------|
| `GET /api/epochs`, `POST /api/epoch`, `GET /api/gt` | **미사용** |

---

## 6. 문의

- FaceTool 계약: `docs/FACETOOL_SERVICE_CONTRACT.md`
- 백엔드: `server/main.py`, `server/run_config.py`, `models/unitalker/run_config.csv`
- 학습/데이터: `docs/METAHUMAN_UNITALKER_INTEGRATION.md`
