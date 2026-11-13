# FaceTool 서비스 계약 (MUST)

FaceTool(`Viewer/FaceTool`)과 trainFramework 추론 서버(`server/main.py`) 사이의 **준수 계약**입니다.  
모델(run)을 추가할 때 **FaceTool 코드를 수정하지 않아도** 되게 하는 것이 목표입니다.

관련 문서:

- API 필드 상세: [`FRONTEND_INFERENCE_API.md`](FRONTEND_INFERENCE_API.md)
- GT는 서버에 맡기지 않음: [`FACETOOL_EVAL_BACKEND_SCOPE.md`](FACETOOL_EVAL_BACKEND_SCOPE.md)
- 서버 SoT: 각 run 폴더의 `service_manifest.json` (+ 체크포인트)

---

## 1. 원칙

| 원칙 | 설명 |
|------|------|
| 목록 SoT | 모델 드롭다운은 **오직** `GET /api/runs` |
| 분기 SoT | 렌더는 **오직** `output_kind` (`arkit_51` \| `metahuman_260`) |
| 바인딩 | MetaHuman은 **컬럼 이름**으로 RigLogic 연결 (인덱스 단독 금지) |
| GT | 로컬 NPY. 서버 `/api/gt` · epoch `gtCsvContent` **사용 금지** |
| 모델 추가 | 서버에 `best.pt`(+ `service_manifest.json`, 선택 ONNX)만 두면 FaceTool **코드 변경 0** |

---

## 2. MUST

1. **모델 목록**은 `GET /api/runs`의 `runs[]`만 사용한다.  
   - run_id 화이트리스트, `*control260*` 같은 **이름 패턴 필터 금지**.
2. **렌더 분기**는 추론 응답(또는 run 메타)의 `output_kind`만 본다.  
   - `arkit_51` → GLB morph, `blendshapes_glb`와 `data` 열 1:1  
   - `metahuman_260` → RigLogic, `blendshapes[]` **이름**으로 컨트롤 매핑
3. `channels` · `data[0].length` · `blendshapes.length`가 하나라도 다르면 **적용하지 않고** 에러 UI.
4. GT 비교는 로컬 `{basePath}/{split}/{clip}.npy` + 동일 stem `{clip}.wav`.
5. MetaHuman GT/Pred 정렬은 **컬럼명 교집합** 후 비교한다. Pred의 HeadYaw 등이 0이어도 이름 매칭을 유지한다.
6. 알 수 없는 `output_kind`는 “지원하지 않음” UI. GLB/RigLogic을 임의 추측 적용하지 않는다.
7. `hasCheckpoint === false` 인 run은 드롭다운에서 비활성(또는 숨김). 추론 요청을 보내지 않는다.
8. `dnaId`(있을 때)에 맞는 로컬 DNA를 선택한다. 서버는 메쉬를 보내지 않는다.

---

## 3. MUST NOT

- FaceTool 배포·릴리스마다 서버 run 폴더명을 하드코딩하지 않는다.
- MetaHuman 경로에서 `blendshapes_glb`를 쓰지 않는다.
- 서버 응답 채널을 51/260 외로 잘라 DNA/GLB에 넣지 않는다.
- 서버 GT API로 학습 검증 GT를 가져오지 않는다.

---

## 4. SHOULD (검증 UX)

- GT \| Pred 동기 재생, 프레임 스크러버
- 채널별 `|GT − Pred|` 타임라인 (클라이언트 계산)
- 동일 오디오로 run A/B 두 번 `POST /api/inference`

---

## 5. API 계약 요약

### 5.1 `GET /api/runs`

FaceTool이 UI를 그리는 데 필요한 필드:

| 필드 | 의미 |
|------|------|
| `id` | run 폴더명 (`POST /api/inference`의 `run`) |
| `label` | 드롭다운 표시명 |
| `output_kind` | `arkit_51` \| `metahuman_260` |
| `channels` | 51 또는 260 |
| `dataset` | 학습/템플릿 데이터셋 키 (로컬 GT basePath 매핑용) |
| `engines` | `{ pytorch: bool, onnx: bool }` 사용 가능 엔진 |
| `dnaId` | 로컬 DNA 식별자 (예: `0421_V33`, `D9_Nuridam`) |
| `hasCheckpoint` | 추론 가능 여부 |
| `isDefault` | 기본 선택 후보 |
| `strategy` | 표시용 (PRETRAINED 등) |

그룹/필터는 `output_kind`로만 한다. `id` 문자열 패턴으로 하지 않는다.

### 5.2 `POST /api/inference`

**요청**

| 필드 | 필수 | 설명 |
|------|------|------|
| `audio_base64` | 예 | 16kHz mono WAV 등 |
| `run` | 로컬 추론 시 예 | `GET /api/runs`의 `id` |
| `checkpoint` | 아니오 | 기본 `best.pt` |
| `speaker` | 아니오 | 기본 `0` |
| `engine` | 아니오 | `pytorch`(기본) \| `onnx` |
| `source_audio_filename` | 아니오 | 디버그 전용. GT 조회에 쓰지 않음 |

**응답 (필수)**

| 필드 | 설명 |
|------|------|
| `success` | true |
| `output_kind` | `arkit_51` \| `metahuman_260` |
| `channels` | `data[0].length` |
| `blendshapes` | 채널 이름 배열 |
| `data` | `float[frames][channels]` |
| `fps` | 30 |
| `frames` | `data.length` |
| `blendshapes_glb` | **ARKit만** |
| `engine` | 실제 사용한 엔진 |

```typescript
function applyInference(res: InferenceResponse) {
  if (res.channels !== res.blendshapes.length) throw new Error("contract mismatch");
  if (res.data.length && res.data[0].length !== res.channels) throw new Error("contract mismatch");

  switch (res.output_kind) {
    case "arkit_51":
      applyToGlbMorph(res.blendshapes_glb!, res.data);
      break;
    case "metahuman_260":
      applyToRigLogicByName(res.blendshapes, res.data);
      break;
    default:
      showUnsupported(res.output_kind);
  }
}
```

---

## 6. 로컬 GT / DNA

| `output_kind` | GT basePath 예 | 형식 |
|---------------|----------------|------|
| `metahuman_260` | `E:\DatasetGen\D8_metahuman_control` 등 | `{split}/{clip}.npy` + `feature_names.txt` |
| `arkit_51` | LipReading 등 | `{split}/{clip}.npy` |

`GET /api/runs`의 `dataset` → FaceTool 설정의 basePath 매핑 테이블만 유지한다.  
run마다 코드 분기를 추가하지 않는다.

| `dnaId` | 의미 |
|---------|------|
| `0421_V33` | D8 마루샘 DNA |
| `D9_Nuridam` | D9 누리담 DNA |

---

## 7. 모델 추가 시 FaceTool 체크리스트

서버에 새 `models/unitalker/results_*/best.pt` (+ `service_manifest.json`, 선택 `.onnx`)가 생기면:

- [ ] FaceTool **코드 변경 없음**
- [ ] `GET /api/runs`에 새 항목이 보이는지 확인
- [ ] 로컬 GT `basePath`가 해당 `dataset`과 맞는지 설정만 확인
- [ ] `dnaId`에 맞는 로컬 DNA가 FaceTool에 있는지 확인
- [ ] `hasCheckpoint === true` 인 항목으로 추론 → GT\|Pred 재생

---

## 8. 서버 `service_manifest.json`

경로: `models/unitalker/{run_id}/service_manifest.json`  
FaceTool은 이 파일을 직접 읽지 않는다. `GET /api/runs` / 추론 응답으로만 소비한다.

스키마 요약: `schema_version`, `run_id`, `output_kind`, `channels`, `dataset_for_blendshape`, `template_idx`, `feature_names_file`, `fps`, `sample_rate`, `engines`, `dna_id`, `label`.

생성: `scripts/write_service_manifest.py`, `models/unitalker/export_onnx.py` (export 시 merge).
