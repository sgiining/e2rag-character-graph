# e2rag-character-graph: Entity·Event 추출 코드와 데이터 구조

이 문서는 `src/extraction`의 코드 4개와 `data` 폴더의 결과 파일을 설명한다.
코드는 Python 3.12 기준이며 LLM은 OpenAI `gpt-4o-mini`를 사용한다.
API 키는 프로젝트 루트의 `.env`(`OPENAI_API_KEY`)에서 읽는다. `.env`는 저장소에 올리지 않는다.

---

## 1. 프로젝트 데이터 처리 구조

### 1.1 전체 흐름

```text
data/raw/<작품>.txt
   │  src/preprocessing/preprocess_text.py   (제목·역자 정보 등 제거)
   ▼
data/cleaned/<작품>.txt ───────────────────────────────┐
   │  src/chunking/split_jsonl_by_chunk.py              │ (원문 좌표 기준)
   ▼                                                    │
data/chunk/<작품>.jsonl  (1200토큰, 겹침 100토큰)        │
   │                                                    │
   ├─▶ extract_entity_mentions_v5.py ──▶ data/entity_mentions_new/<작품>/<실행시각>/
   │                                         │ entities.jsonl, summary.json
   │                                         ▼
   │       convert_mentions_to_observations.py ──▶ data/entity_observation/<작품>.entities.jsonl
   │                                                         │
   └─▶ extract_events_v1.py ──▶ data/events_new/<작품>/<실행시각>/      │
                                   │ events.jsonl, summary.json       │
                                   ▼                                  ▼
                          link_entity_event_v1.py ──▶ data/entity_event_new/<작품>.entity_event_edges.jsonl
                                                                    <작품>.entity_event_edges.summary.json
```

- Entity 추출과 Event 추출은 서로 독립적이다. 둘 다 `cleaned`와 `chunk`만 입력으로 받는다.
- `convert`는 Entity 결과만 사용하고 `link`는 `convert` 결과와 Event 결과를 함께 사용한다.
- 모든 코드는 실행 전에 `cleaned`와 `chunk`의 SHA-256 해시를 검사한다. 앞 단계와 다른 원문이 들어오면 실행을 멈춘다.

### 1.2 `src/extraction` 폴더의 역할

원문 chunk에서 Entity와 Event를 추출하고 둘을 연결하는 코드가 들어 있다.
전처리(`src/preprocessing`)와 chunk 분할(`src/chunking`)은 이 폴더 밖에 있다.

| 파일 | 한 줄 요약 | LLM 호출 |
|---|---|---|
| `extract_entity_mentions_v5.py` | 원문 위치 단위 Entity mention 추출 | 예 |
| `convert_mentions_to_observations.py` | mention을 (chunk, 이름) 단위 observation으로 변환 | 아니요 |
| `extract_events_v1.py` | chunk 단위 Event와 Event-Event 관계 추출 | 예 |
| `link_entity_event_v1.py` | Entity observation과 Event 연결 | 아니요 |

### 1.3 헷갈리기 쉬운 용어 (실제 구현 기준)

| 용어 | 실제 구현에서의 의미 | 저장 위치 |
|---|---|---|
| chunk | `split_jsonl_by_chunk.py`가 만든 1200토큰 구간. `chunk_id`는 1부터 시작한다 | `data/chunk` |
| window (창) | Entity 추출 때 chunk 안을 다시 약 700자로 나눈 검토 구간. `window_id`는 `"<chunk_id>:<번호>"` | `coverage.jsonl` |
| mention | 원문의 특정 위치 하나에 나온 Entity 표기. 같은 원문 위치(`original_span`)면 하나의 mention이다 | `entity_mentions_new/.../entities.jsonl` |
| accepted observation | 한 window에서 통과 판정을 받은 mention 후보 한 건. 같은 위치가 여러 window·단계에서 나오면 여러 건이다 | `accepted_observations.jsonl` |
| Entity observation | 한 chunk 안에서 같은 이름을 가진 mention들을 묶은 한 줄. 다른 chunk끼리는 합치지 않는다 | `entity_observation/*.entities.jsonl` |
| `name_key` | 같은 이름을 찾아보기 위한 참고 키. 같은 인물이라는 뜻이 아니다 | observation |
| Event | chunk에서 추출한 사건 노드. 같은 원문 위치를 다시 추출한 경우(주로 chunk 겹침 구간)만 하나로 합친다 | `events.jsonl` |
| occurrence | 한 Event가 추출된 각 기록. `occurrence_chunk_ids`에 등장 chunk가 모인다 | `events.jsonl` |
| Event-Event edge | 두 Event 사이의 방향이 있는 관계(`PRECEDES` 등) | `event_edges.jsonl` |
| Entity-Event edge | observation과 Event의 연결(bipartite edge) | `entity_event_new/*.jsonl` |
| τ | 서사 위치. 코드에서는 `chunk_id`(첫 등장 chunk)와 `available_at_chunk`(공개 가능 chunk)로 저장된다 | 모든 노드·edge |
| `source_id` | Entity mention 결과에서 chunk id를 가리키는 필드 이름(E²RAG 표기) | `entities.jsonl` |

**주의:** 파일 이름에 `entities`가 두 곳에서 쓰인다.
`entity_mentions_new/.../entities.jsonl`은 **mention 단위**이고 `entity_observation/<작품>.entities.jsonl`은 **observation 단위**이다.

### 1.4 좌표 표기 규칙

- `original_span: [a, b]` : `data/cleaned/<작품>.txt` 전체 문자열 기준 문자 위치. `a` 포함 `b` 제외(0부터).
- `chunk_span: [a, b]` : 해당 chunk의 `text` 기준 문자 위치. 같은 규칙이다.
- `token_start`, `token_end` : 원문 전체 토큰 기준 위치. 1부터 시작하고 양 끝을 포함한다.

---

## 2. `src/extraction` 코드 파일별 역할

### 2.1 `extract_entity_mentions_v5.py`

- **주요 역할**
  - 각 chunk를 약 700자 window로 나눠 Entity 언급을 원문 위치 단위로 추출한다.
  - LLM이 낸 이름과 근거 구절을 코드가 원문에서 직접 찾아 좌표로 검증하고 통과한 것만 저장한다.
  - 파일 이름은 `v5`이지만 결과의 `extractor_version` 값은 `"v3"`이다.
- **입력 데이터**
  - `--cleaned data/cleaned/<작품>.txt`
  - `--chunks data/chunk/<작품>.jsonl`
  - `--project-root` 아래의 `.env`
- **주요 처리 과정**
  1. 입력 검증: tiktoken으로 chunk 토큰 범위가 원문과 맞는지 확인하고 chunk를 원문 문자 좌표에 매핑한다.
  2. window 생성: 어절 경계에 맞춰 겹치는 window를 만든다(기본 700자, 겹침 150자).
  3. 생성(A·B): window마다 서로 다른 지시로 두 번 추출한다.
     A는 순서대로 모든 언급을 찾고 B는 문장별 발화자·주체·대명사를 찾는다.
     근거가 빠진 항목이 절반을 넘으면 다시 요청한다.
  4. 좌표 해석: 이름과 근거 구절을 원문에서 찾는다. 찾지 못하면 `literal_fail`로 기록하고 버린다.
  5. 표면형 확장: 같은 window에서 이미 찾은 이름이 다른 위치에도 있으면 판정 후보로 추가한다.
  6. 판정: 같은 원문 위치의 후보를 묶어 LLM이 지시 대상 범주를 판정하고 코드 규칙으로 `supported/reject/uncertain`을 정한다. `supported`만 남긴다.
  7. 병합: 같은 원문 위치(`original_span`)만 하나의 mention으로 합친다. 유형은 다수결로 정하고 동률이면 `null`이다.
- **출력 데이터** `data/entity_mentions_new/<작품>/<UTC 실행시각>/`
  - `summary.json`, `entities.jsonl`, `accepted_observations.jsonl`, `coverage.jsonl`, `review.jsonl`, `audit.jsonl`, `calls.jsonl`
  - 실행할 때마다 새 폴더를 만들고 기존 결과를 덮어쓰지 않는다.
  - 실패한 window가 있으면 `status`가 `incomplete`가 되고 종료 코드 2로 끝난다.
- **실행 관계**
  - 후속 코드 `convert_mentions_to_observations.py`가 `summary.json`과 `entities.jsonl`을 읽는다.
  - `extract_events_v1.py`와 `convert_mentions_to_observations.py`가 이 파일의 입력 검증 함수(`load_inputs`, `digest` 등)를 import해서 재사용한다.

### 2.2 `convert_mentions_to_observations.py`

- **주요 역할**
  - mention 단위 결과를 Event 연결과 이후 단계에서 쓰는 (chunk, 이름) 단위 Entity observation으로 바꾼다.
  - 같은 chunk 안의 같은 이름 mention을 한 줄로 묶고 각 mention의 위치는 `mentions`에 보존한다. LLM은 호출하지 않는다.
- **입력 데이터**
  - `--run-dir data/entity_mentions_new/<작품>/<실행시각>` 안의 `summary.json`, `entities.jsonl`
  - `--cleaned data/cleaned/<작품>.txt`
  - `--chunks data/chunk/<작품>.jsonl`
- **주요 처리 과정**
  1. `summary.json`의 `status`가 `completed_with_review`인지와 원문·chunk 해시가 같은지 확인한다.
  2. 각 mention을 그 위치를 온전히 포함하는 모든 chunk에 배정한다. 겹침 구간의 mention은 두 chunk에 모두 들어간다.
  3. (chunk, 정규화한 이름)으로 묶어 observation을 만든다. 유형은 mention 유형의 다수결이다.
  4. `description`과 `evidence`는 묶음에서 설명이 있는 첫 mention의 값을 그대로 쓴다.
  5. 모든 mention이 빠짐없이 배정됐는지와 ID 중복이 없는지 검사한다.
- **출력 데이터**
  - 기본 `data/entity_observation/<작품>.entities.jsonl`(`--output`으로 변경 가능)
  - 이미 있으면 `--overwrite` 없이는 저장하지 않는다.
- **실행 관계**
  - `extract_entity_mentions_v5.py`의 결과를 입력으로 받는다.
  - `link_entity_event_v1.py`가 이 결과를 읽는다.

### 2.3 `extract_events_v1.py`

- **주요 역할**
  - chunk마다 Entity 추출과 별개로 사건(Event)과 같은 chunk 안의 사건 간 관계를 추출한다.
  - 사건 근거 구절과 trigger의 원문 좌표를 코드로 검증한다.
  - 시간 순서가 모순되는 관계를 결정적 규칙으로 표시한다. 결과의 `extractor_version` 값은 `"event.v6"`이다.
- **입력 데이터**
  - `--cleaned data/cleaned/<작품>.txt`
  - `--chunks data/chunk/<작품>.jsonl`
  - `.env`
- **주요 처리 과정** (chunk마다 반복)
  1. extract: 사건과 관계를 추출한다(사건 ID `E1, E2…`).
  2. glean: 빠진 사건을 한 번 더 요청한다(ID `G1, G2…`). `--no-gleaning`으로 끌 수 있다.
  3. 좌표 검증: evidence를 정확 일치 → 공백 차이 → 앞뒤 문장부호 제거 → 줄 단위 부분 일치 → 유사 일치 순서로 원문에서 찾는다. 찾지 못한 사건은 거부한다.
  4. repair: 거부된 사건만 원문을 다시 복사해 달라고 한 번 요청한다. `--no-repair`로 끌 수 있다.
  5. verify: 원문과 대조해 `supported/revise/not_event`를 판정한다. `revise`이면 고친 값으로 바꾸고 `not_event`는 제거한다. 고친 값이 원래와 같으면 한 번 더 요청하고 그래도 같으면 `revise_unfixed`로 남긴다. `--no-verify`로 끌 수 있다.
  6. 저장소 병합: 이미 있는 사건과 근거 위치가 겹치고 라벨·trigger·설명 중 하나가 같거나 비슷하면 같은 사건으로 보고 등장 기록(`occurrences`)만 추가한다.
  7. 관계 검증: 존재하는 두 사건을 잇는지 확인하고 라벨을 대문자로 정규화한다. `FOLLOWS`는 제외한다.

  모든 chunk를 처리한 뒤에는 다음 두 가지를 실행한다.
  - 같은 쌍의 반대 방향·다른 라벨 관계에 `conflicting_pair`를 표시한다.
  - 시간 모순(쌍 충돌, 순환)이 있는 관계는 `temporal_valid=false`로 표시한다. 관계를 삭제하지는 않는다.

  `--cross-chunk`를 주면 인접 chunk 사이 관계도 추출한다. 현재 결과는 모두 이 옵션 없이 실행했다(`summary.json`의 `"cross_chunk": false`).
- **출력 데이터** `data/events_new/<작품>/<UTC 실행시각>/`
  - `summary.json`, `events.jsonl`, `event_edges.jsonl`, `review.jsonl`, `audit.jsonl`, `calls.jsonl`
  - 실행할 때마다 새 폴더를 만든다.
- **실행 관계**
  - Entity 결과와 무관하게 단독 실행할 수 있다.
  - `link_entity_event_v1.py`가 `summary.json`과 `events.jsonl`을 읽는다.
  - `event_edges.jsonl`은 현재 `src/extraction`의 어떤 코드도 읽지 않는다. 이후 그래프 구성 단계에서 쓰기 위한 결과다.

### 2.4 `link_entity_event_v1.py`

- **주요 역할**
  - 같은 chunk에 있는 Entity observation과 Event를 연결하는 edge를 만든다.
  - 연결 근거를 `methods`에 모두 기록한다. E²RAG 원논문 방식과 본 연구 확장 방식을 구분할 수 있다. LLM은 호출하지 않는다.
- **입력 데이터**
  - `--entities data/entity_observation/<작품>.entities.jsonl`
  - `--event-run data/events_new/<작품>/<실행시각>` 안의 `summary.json`, `events.jsonl`
- **주요 처리 과정**
  1. Event 실행이 `completed_with_review`인지와 작품 ID·원문 해시·chunk 해시가 양쪽에서 같은지 확인한다.
  2. Event가 등장한 모든 chunk(`occurrence_chunk_ids`)마다 같은 chunk의 observation과 비교한다.
  3. 아래 방법 중 하나라도 맞으면 연결하고 맞은 방법을 모두 기록한다.

     | method | 조건 | 구분 |
     |---|---|---|
     | `name_in_description` | 정규화한 Event 설명에 이름이 포함됨 | 원논문 식 (1) |
     | `name_in_label` | Event 이름(label)에 이름이 포함됨 | E²RAG 공식 구현 |
     | `name_in_description_bounded` | 위 포함이 단어 경계(뒤에 조사 허용)에 맞음 | 본 연구 확장 |
     | `name_in_label_bounded` | label 포함이 단어 경계에 맞음 | 본 연구 확장 |
     | `mention_in_evidence` | mention 원문 위치가 Event 근거 구절 위치 안에 있음 | 본 연구 확장 |
     | `participant` | Event 참여자 표기 위치와 mention 위치가 같음 | 본 연구 확장 |

  4. `recommended`를 계산한다. 원문 좌표 근거(`mention_in_evidence` 또는 `participant`)가 있거나 `name_in_description_bounded`이면 `true`이다. label에만 이름이 있는 연결은 권장하지 않는다.
- **출력 데이터**
  - 기본 `data/entity_event_new/<작품>.entity_event_edges.jsonl`과 `<작품>.entity_event_edges.summary.json`
  - 이미 있으면 `--overwrite` 없이는 저장하지 않는다.
- **실행 관계**
  - `convert_mentions_to_observations.py`와 `extract_events_v1.py`의 결과를 모두 필요로 하는 마지막 단계다.

---

## 3. `data` 폴더별 역할

| 폴더 | 저장 파일 | 의미와 용도 | 생성 코드 | 연결 관계 |
|---|---|---|---|---|
| `raw/` | `<작품>.txt` 5개 | 공유마당 원본 번역문. 제목·역자 정보·역자 약력 포함 | 없음(원본) | `preprocess_text.py`의 입력 |
| `cleaned/` | `<작품>.txt` 5개 | 불필요한 정보를 제거한 본문. 모든 `original_span`의 기준 문자열 | `src/preprocessing/preprocess_text.py` | 모든 추출 코드의 입력 |
| `chunk/` | `<작품>.jsonl` 5개 | 1200토큰(겹침 100) chunk. `chunk_id`가 τ의 단위 | `src/chunking/split_jsonl_by_chunk.py` | 모든 추출 코드의 입력 |
| `entity_mentions_new/` | `<작품>/<실행시각>/` 폴더 | mention 추출 결과와 실행 기록 | `extract_entity_mentions_v5.py` | `entities.jsonl`과 `summary.json`이 `convert`의 입력 |
| `entity_observation/` | `<작품>.entities.jsonl` 5개 | (chunk, 이름) 단위 Entity observation | `convert_mentions_to_observations.py` | `link`의 입력 |
| `events_new/` | `<작품>/<실행시각>/` 폴더 | Event, Event-Event edge, 실행 기록 | `extract_events_v1.py` | `events.jsonl`과 `summary.json`이 `link`의 입력 |
| `entity_event_new/` | `<작품>.entity_event_edges.jsonl`과 `.summary.json` 5쌍 | Entity-Event 연결 결과와 통계 | `link_entity_event_v1.py` | 이후 그래프 구성 단계의 입력 |

**실행 폴더 이름** `<UTC 실행시각>`은 `YYYYMMDDTHHMMSS_ffffffZ` 형식이다(예: `20261014T072303_007795Z`).

**현재 사용 중인 실행 폴더**

`눈의 여왕`의 `entity_mentions_new`에는 실행 폴더가 두 개 있다. observation이 실제로 사용한 폴더는 각 레코드의 `extractor.source_run`에 기록되어 있다.

| 작품 | chunk 수 | Entity 실행(`source_run`) | mention | observation | Event 실행 | Event | Event-Event | Entity-Event |
|---|---|---|---|---|---|---|---|---|
| 1_눈의 여왕 | 20 | `20261014T061924_435741Z` | 1,206 | 482 | `20261014T072303_007795Z` | 310 | 310 | 829 |
| 2_엄지 공주 | 7 | `20261012T061317_990756Z` | 430 | 176 | `20261014T074011_885877Z` | 106 | 106 | 336 |
| 3_인어 공주 | 14 | `20261012T063447_690772Z` | 815 | 394 | `20261014T074603_163016Z` | 187 | 174 | 626 |
| 4_성냥팔이 소녀 | 2 | `20261012T071428_675477Z` | 75 | 24 | `20261014T075623_593432Z` | 29 | 27 | 72 |
| 5_미운 아기 오리 | 6 | `20261012T072333_649726Z` | 320 | 131 | `20261014T075833_904172Z` | 93 | 94 | 299 |

`눈의 여왕`의 이전 Entity 실행 `20261012T052707_451382Z`는 현재 observation에 쓰이지 않는다.

---

## 4. 결과 파일의 구조 및 필드 설명

같은 구조의 파일은 작품별로 반복하지 않고 한 번만 설명한다. 표의 "필수"는 해당 파일의 모든 레코드에 키가 있다는 뜻이다. 값이 `null`일 수 있으면 따로 적었다.

### 4.1 `data/cleaned/<작품>.txt`

- 형식: UTF-8 텍스트
- 내용: 전처리된 본문. `original_span`은 모두 이 파일 전체 문자열 기준이다.

### 4.2 `data/chunk/<작품>.jsonl`

- 형식: JSONL(한 줄이 chunk 하나)

| 필드명 | 의미 | 값 예시 |
|---|---|---|
| `chunk_id` | chunk 번호. 1부터 연속 | `1` |
| `text` | chunk 원문 | `"일곱 가지 이야기\n\n첫 번째 이야기…"` |
| `token_start` | 원문 전체 토큰 기준 시작 위치(1부터, 포함) | `1` |
| `token_end` | 끝 위치(포함) | `1200` |
| `token_count` | 토큰 수 | `1200` |
| `chunk_size_setting` | chunk 크기 설정 | `1200` |
| `overlap_setting` | 겹침 크기 설정 | `100` |
| `tokenizer_model` | 토큰화 기준 모델 | `"gpt-4o-mini"` |

### 4.3 `entity_mentions_new/<작품>/<실행시각>/entities.jsonl` (mention 단위)

- 형식: JSONL(한 줄이 원문 위치 하나의 mention)

| 필드명 | 의미 | 값 예시 |
|---|---|---|
| `mention_id` | mention ID(원문 해시와 위치로 생성) | `"m_39ca3ef64b813dad47dd8e6b"` |
| `entity_name` | 원문 표기 그대로의 이름 | `"거울"` |
| `entity_type` | 유형: `person`, `creature`, `object`, `location`, `organization`, `other`. 유형 투표가 동률이면 `null` | `"object"` |
| `description` | 판정 단계에서 LLM이 쓴 설명 | `"이 이야기에 등장하는 거울이다."` |
| `mention_kind` | 언급 형태: `named`, `nominal`, `pronominal` | `"named"` |
| `source_id` | 대표 관측이 나온 chunk id | `1` |
| `original_span` | 원문 위치 | `[20, 22]` |
| `document_sha256` | 원문 해시 | `"bc978c…"` |
| `type_votes` | 관측별 유형 투표 수 | `{"object": 1}` |
| `type_resolution` | 유형 결정 방식: `unanimous`, `majority`, `unresolved_tie` | `"unanimous"` |
| `observations` | 이 위치로 통과한 관측 목록. 각 항목은 4.4의 accepted observation과 같은 구조 | `[{…}]` |
| `needs_review` | 유형·형태 충돌이나 위치 겹침이 있으면 `true` | `false` |

### 4.4 `entity_mentions_new/.../accepted_observations.jsonl`

- 형식: JSONL(한 줄이 window 하나에서 통과한 후보 하나)
- 내용: 병합 전 관측 기록이다. `entities.jsonl`의 `observations`에 같은 내용이 들어간다.

| 필드명 | 의미 | 값 예시 |
|---|---|---|
| `entity_name` | 이름 | `"거울"` |
| `evidence` | 근거 구절(원문 부분 문자열) | `"거울과 거울 조각"` |
| `source_id` | chunk id | `1` |
| `window_id` | 판정한 window | `"1:1"` |
| `window_original_span` | window의 원문 위치 | `[0, 700]` |
| `original_span` / `chunk_span` | 이름의 원문 위치 / chunk 내 위치 | `[20, 22]` |
| `evidence_original_span` / `evidence_chunk_span` | 근거 구절의 원문 위치 / chunk 내 위치 | `[20, 29]` |
| `token_start`, `token_end` | 소속 chunk의 토큰 범위 | `1`, `1200` |
| `entity_type` | 판정 범주로 코드가 정한 유형 | `"object"` |
| `mention_kind` | 판정 단계의 언급 형태 | `"nominal"` |
| `proposed_type_raw` | 생성 단계 LLM이 처음 낸 유형. 확장 후보면 `null` | `"creature"` |
| `description` | 판정 단계 설명 | `"…"` |
| `inclusion_reason` | 생성 단계의 포함 이유. 확장 후보면 빈 문자열 | `"…"` |
| `evidence_occurrence`, `mention_occurrence` | LLM이 낸 출현 순서 값(기록용). 현재 결과는 모두 `null` | `null` |
| `evidence_match` | 근거 찾기 방식: `exact`, `whitespace_normalized`, `code_sentence`(확장 후보는 코드가 문장 단위로 근거를 잘라냄) | `"exact"` |
| `resolution` | 위치 결정 방식: `unique`(1곳), `expanded`(여러 곳 각각), `surface_expansion`(표면형 확장) | `"unique"` |
| `phase` | 대표 관측의 단계: `A`, `B`, `surface_expansion` | `"A"` |
| `phases` | 이 위치를 낸 모든 단계 | `["A", "B"]` |
| `seed_priority` | **선택 필드.** 표면형 확장으로만 나온 후보에만 있다. 0=인물·생물·조직 씨앗, 1=그 외 | `0` |
| `proposals` | 생성 단계(A·B)의 제안 목록(`source`, `proposed_project_type`, `mention_kind`, `description`) | `[{…}]` |
| `proposed_inclusion_reasons` | 생성 단계 포함 이유 목록 | `["…"]` |
| `semantic_review` | 판정 결과(아래 표) | `{…}` |

`semantic_review` 내부 필드

| 필드명 | 의미 |
|---|---|
| `verdict` | 코드 최종 판정. 이 파일에는 `supported`만 있다 |
| `rule` | 판정 규칙: `all_checks_passed`(범주로 유형 결정), `all_checks_passed_type_name`(직접 지정 유형 사용) |
| `reason` | LLM 판정 이유 |
| `included`, `name_complete` | LLM의 포함 여부와 이름 완전성 판단 |
| `referent_category` | 지시 대상 범주(`human`, `living_nonhuman`, `physical_object` 등) |
| `direct_type` | 범주 대신 직접 지정한 유형. 대부분 `null` |
| `mention_kind` | 판정 단계 언급 형태 |
| `type_evidence`, `agency_evidence` | 유형·행위성 근거 구절 |
| `type_evidence_grounded`, `agency_evidence_grounded` | 각 근거가 window 원문에 실제로 있는지 |
| `description_ok` | 설명이 비어 있지 않은지 |
| `invalid_fields` | 형식이 잘못된 응답 필드 목록 |

### 4.5 `entity_mentions_new/.../coverage.jsonl`

- 형식: JSONL(한 줄이 window 하나)
- 내용: window별 처리 결과. 원문 전체를 빠짐없이 검토했는지 확인하는 용도다.

| 필드명 | 의미 |
|---|---|
| `window_id`, `chunk_id`, `text` | window 식별자, 소속 chunk, window 원문 |
| `original_start`, `original_end`, `chunk_start` | window의 원문 위치와 chunk 내 시작 위치 |
| `status` | `reviewed`(완료) 또는 `incomplete`(요청 실패 등) |
| `generated_positions` | A·B 생성으로 위치가 확인된 후보 수 |
| `seed_names` | 표면형 확장에 쓴 이름 목록 |
| `expansion_candidates`, `expansion_skipped_cap` | 확장 후보 수, 상한 초과로 판정하지 못한 수 |
| `expansion_supported`, `expansion_rejected`, `expansion_uncertain`, `expansion_unreviewed` | 확장 후보의 판정 결과별 수 |
| `generated_supported`, `generated_rejected`, `generated_uncertain`, `generated_unreviewed` | 생성 후보의 판정 결과별 수 |
| `accepted_spans` | 통과한 원문 위치 목록 |
| `errors` | 오류 메시지 목록 |

### 4.6 `entity_mentions_new/.../summary.json`

- 형식: JSON(실행 하나의 요약)
- 주요 필드
  - 실행 정보: `status`(`completed_with_review` / `incomplete` / `failed` / `input_checked_only`), `extractor_version`, `model`, `cleaned`, `chunks`
  - 설정: `window_chars`, `window_overlap`, `expansion_scope`, `max_expansion_per_window`
  - 입력 검증: `input_chunks`, `document_characters`, `document_sha256`, `chunks_sha256`, `boundary_warnings`, `nonwhitespace_coverage`, `total_windows`, `window_midword_edges`
  - 결과: `unique_mentions`(= `entities.jsonl` 줄 수), `accepted_observations`, `flagged_mentions`, `inclusion_reason_placeholders`, `mention_kind_counts`, `entity_type_counts`
  - `stats`: 단계별 세부 카운터(`raw_candidates`, `literal_pass`, `literal_fail`, `semantic_supported`, `semantic_reject`, `api_calls`, `input_tokens`, `windows_completed`, `windows_failed` 등)

### 4.7 실행 기록 파일 `review.jsonl`, `audit.jsonl`, `calls.jsonl`

`entity_mentions_new`와 `events_new`의 실행 폴더에 같은 이름으로 있다. 후속 코드는 읽지 않는 **검토용 기록**이다.

| 파일 | 내용 | 공통 키 |
|---|---|---|
| `calls.jsonl` | LLM 요청·응답 원문 | `label`(예: `"1:1:A"`, `"1:extract"`), `attempt`, `payload_sha256`, `finish_reason`, `response`. Entity 쪽만 `request_items`가 있다 |
| `audit.jsonl` | 모든 처리 결정의 기록 | `kind` + 종류별 필드 |
| `review.jsonl` | 사람이 확인할 항목만 모은 기록 | `kind` + 종류별 필드 |

현재 결과에 나타난 `kind` 값

| 위치 | 파일 | `kind` 값 |
|---|---|---|
| Entity | `review.jsonl` | `overlapping_spans`, `classification_conflict`, `literal_fail`, `semantic_reject`, `semantic_uncertain`, `window_prefix_skipped`, `window_suffix_skipped`, `utf8_boundary`, `window_needs_review` |
| Entity | `audit.jsonl` | `semantic_review`, `literal_pass`, `classification_corrected`, `literal_fail`, `occurrence_boundary_skipped`, `request_failure` |
| Event | `review.jsonl` | `event_flags`, `relation_rejected`, `edge_flags`, `event_rejected`, `temporal_pair_conflict`, `utf8_boundary`, `edge_conflict`, `event_removed_by_verify`, `temporal_cycle_broken` |
| Event | `audit.jsonl` | `event_added`, `event_merged`, `relation_rejected`, `event_rejected`, `event_removed_by_verify` |

### 4.8 `data/entity_observation/<작품>.entities.jsonl` (observation 단위)

- 형식: JSONL(한 줄이 (chunk, 이름) 하나)
- 모든 필드가 필수다.

| 필드명 | 의미 | 값 예시 |
|---|---|---|
| `observation_id` | observation ID(작품·chunk·이름 해시) | `"eobs_f3c7e5c0c1e198c7"` |
| `name_key` | `<작품>:name:<이름>` 형태의 참고 키. 동일 인물을 보장하지 않는다 | `"1_눈의 여왕:name:거울"` |
| `work_id` | 작품 ID(chunk 파일 이름) | `"1_눈의 여왕"` |
| `canonical_name` | 이름(공백 정리) | `"거울"` |
| `aliases` | 별칭 목록. 현재 결과는 모두 빈 목록 | `[]` |
| `entity_type` | mention 유형의 다수결. 동률이면 `null` | `"object"` |
| `description` | 묶음 중 첫 mention의 LLM 설명을 그대로 옮긴 값 | `"이 문맥에서 거울은 …"` |
| `evidence` | 같은 mention의 첫 근거 구절 | `"거울과 거울 조각"` |
| `chunk_id` | 소속 chunk(τ) | `1` |
| `available_at_chunk` | 공개 가능 chunk. `chunk_id`와 같다 | `1` |
| `token_start`, `token_end` | 소속 chunk의 토큰 범위 | `1`, `1200` |
| `is_overlap_repeat` | 이 이름의 mention 위치가 모두 앞선 chunk의 같은 이름 observation에도 있었으면 `true`(겹침 구간 반복) | `false` |
| `needs_review` | 유형 미확정이거나 mention 중 하나라도 검토 대상이면 `true` | `true` |
| `type_votes` | mention 유형 투표 수 | `{"object": 14}` |
| `mention_count` | 묶인 mention 수 | `14` |
| `mentions` | 묶인 mention 목록(아래 표) | `[{…}]` |
| `extractor` | 생성 정보: `model`, `schema_version`(`entity_observation.v2`), `converted_from`(`entity_mention.v3`), `source_run`, `document_sha256`, `chunks_sha256` | `{…}` |

`mentions` 항목 필드: `mention_id`, `original_span`, `chunk_span`(이 chunk 기준), `entity_type`, `mention_kind`, `type_resolution`, `needs_review`, `description`, `evidence`, `evidence_original_span`

### 4.9 `events_new/<작품>/<실행시각>/events.jsonl`

- 형식: JSONL(한 줄이 Event 하나). `chunk_id`, 원문 위치 순서로 정렬되어 있다.

| 필드명 | 의미 | 값 예시 |
|---|---|---|
| `event_id` | Event ID(원문 해시·근거 위치·trigger 위치로 생성) | `"evt_cd9291a6485c1212"` |
| `work_id` | 작품 ID | `"1_눈의 여왕"` |
| `label` | 사건의 짧은 이름 | `"악마가 거울을 완성하다"` |
| `trigger` | 사건을 직접 나타내는 원문 표현. evidence 안에서 찾지 못하면 `null` | `"거울을 마침내 완성해"` |
| `trigger_span` | trigger 원문 위치. `trigger`가 `null`이면 `null` | `[169, 180]` |
| `event_type` | `action`, `interaction`, `conflict`, `support`, `occurrence`, `other` | `"occurrence"` |
| `description` | 한두 문장 설명(E²RAG의 desc) | `"이 악마가 … 완성했다."` |
| `evidence` | 근거 구절(원문 부분 문자열) | `"이 악마가 … 무척 좋았다"` |
| `summary` | `description` 사본. 팀 기존 코드 호환용이며 현재 `src/extraction`에는 이 필드를 읽는 코드가 없다 | `description`과 동일 |
| `participant_names` | `participants`의 이름만 모은 목록 | `["악마"]` |
| `original_span` / `chunk_span` | evidence의 원문 위치 / 첫 chunk 내 위치 | `[153, 190]` |
| `participants` | 참여자 목록(아래 표) | `[{…}]` |
| `narrative_level` | `main`(본 줄거리) 또는 `embedded`(액자 속 이야기·꿈·상상) | `"main"` |
| `verdict` | verify 판정: `supported`, `revise`(고침), `revise_unfixed`(고치지 못함). verify를 끄면 `null` | `"supported"` |
| `verify_reason` | verify 판정 이유 | `"원문과 일치하는 사건이다."` |
| `chunk_id` | 처음 추출된 chunk(τ) | `1` |
| `available_at_chunk` | 공개 가능 chunk. `chunk_id`와 같다 | `1` |
| `occurrence_chunk_ids` | 이 사건이 추출된 모든 chunk | `[1]` 또는 `[3, 4]` |
| `occurrences` | 추출 기록 목록. 각 항목은 `chunk_id`, `phase`(`extract`/`glean`/`repair`), `label`, `description`, `original_span` | `[{…}]` |
| `flags` | 검토용 표시(아래 표) | `["evidence_fuzzy"]` |
| `needs_review` | `flags`가 하나라도 있으면 `true` | `true` |
| `extractor` | `model`, `schema_version`(`event.v6`), `document_sha256`, `chunks_sha256` | `{…}` |
| `label_original`, `description_original`, `participants_original` | **선택 필드.** verify가 값을 고친 사건(`verdict = "revise"`)에만 있다. 고치기 전 값 | `"아이들이 창문을 내다봄"` |

`participants` 항목

| 필드명 | 의미 | 값 예시 |
|---|---|---|
| `name` | 참여자의 원문 표기(chunk 원문에 있는 표기만 허용) | `"악마"` |
| `evidence_spans` | evidence 안에서 이 표기가 나온 원문 위치 목록 | `[[155, 157]]` |
| `resolved_from_context` | evidence 안에는 없고 chunk 다른 곳에만 있으면 `true`. 이때 `evidence_spans`는 빈 목록 | `false` |

`flags` 값(`trigger_missing`은 코드에만 있고 현재 결과에는 없다)

| flag | 의미 |
|---|---|
| `evidence_exact_ambiguous` | 정확히 일치하는 위치가 여러 곳이라 첫 위치를 사용 |
| `evidence_whitespace_normalized` | 공백·줄바꿈만 달라서 맞춤 |
| `evidence_edge_trimmed` | 앞뒤 따옴표·문장부호를 떼고 맞춤 |
| `evidence_partial` | 줄 단위 조각만 원문과 맞음 |
| `evidence_fuzzy` | 한두 글자 차이를 허용해 맞춤 |
| `evidence_repaired` | repair 단계에서 다시 받은 근거 |
| `trigger_not_in_evidence` / `trigger_missing` | trigger가 evidence에 없음 / trigger를 내지 않음 |
| `trigger_ambiguous` | trigger가 evidence 안에 여러 번 나옴 |
| `trigger_is_evidence` | trigger가 긴 evidence 전체와 같음 |
| `event_type_invalid` | 허용되지 않은 유형이라 `other`로 바꿈 |
| `participant_from_context` | 참여자 중 evidence 밖에서만 확인된 표기가 있음 |
| `participant_not_in_description` | 참여자 이름이 description에 없음 |
| `participant_not_in_chunk` | LLM이 낸 참여자 중 chunk 원문에 없는 표기를 버림 |
| `narrative_level_invalid` | verify가 허용되지 않은 `narrative_level`을 냄(값은 `main` 유지) |
| `verified_revised` | verify가 label·description·participants를 고침 |
| `verify_unfixed` | 오류로 판정됐지만 두 번 요청해도 고치지 못함 |

### 4.10 `events_new/<작품>/<실행시각>/event_edges.jsonl`

- 형식: JSONL(한 줄이 Event-Event edge 하나). `available_at_chunk`, `edge_id` 순서로 정렬되어 있다.

| 필드명 | 의미 | 값 예시 |
|---|---|---|
| `edge_id` | edge ID(출발·도착·라벨로 생성) | `"ee_06d82f308be96e11"` |
| `work_id` | 작품 ID | `"1_눈의 여왕"` |
| `source_event_id`, `target_event_id` | 출발·도착 Event | `"evt_e569…"` |
| `relation_type` | 대문자로 정규화한 라벨. 현재 결과에는 `PRECEDES`, `DURING`, `REVEALS`, `CAUSES`, `CASES`가 있다 | `"PRECEDES"` |
| `relation_type_raw` | LLM이 낸 원래 라벨 | `"PRECEDES"` |
| `core_label` | E²RAG 기본 라벨(`PRECEDES, FOLLOWS, DURING, CAUSES, COUNTERACTS, REVEALS, PARALLELS`)인지 | `true` |
| `label_suggestion` | 기본 라벨과 한 글자 차이면 추정 라벨(예: `CASES` → `CAUSES`). 라벨 자체는 바꾸지 않는다. 없으면 `null` | `null` |
| `description` | 두 사건의 연결 설명 | `"악마가 … 알렸다."` |
| `cue` | 관계를 드러내는 원문 구절. 원문에서 찾았을 때만 `{chunk_id, text, match, original_span}`이고 그 외는 `null` | `null` |
| `evidence_kind` | `explicit`(원문에 명시) 또는 `implicit`(문맥상). 형식이 잘못되면 `null` | `"implicit"` |
| `confidence` | LLM 신뢰도(0~1) | `0.9` |
| `endpoint_scores` | 관계 설명이 출발·도착 사건 설명과 공유하는 문자 bigram 비율 | `[0.44, 0.828]` |
| `scope` | `intra_chunk`(같은 chunk) 또는 `cross_chunk`(인접 chunk). 현재 결과는 모두 `intra_chunk` | `"intra_chunk"` |
| `chunk_ids` | 이 관계를 추출한 chunk 목록 | `[1]` |
| `available_at_chunk` | 공개 가능 chunk = max(출발 τ, 도착 τ, 추출 chunk 중 가장 앞) | `1` |
| `flags` | 검토용 표시(아래 표) | `[]` |
| `needs_review` | `flags`가 하나라도 있으면 `true` | `false` |
| `temporal_valid` | 시간 모순 해소 결과. `false`이면 쌍 충돌이나 순환 때문에 시간 순서 판단에서 제외된 edge다. edge 자체는 남아 있다 | `true` |

`flags` 값(코드에 정의된 전체 목록이며 `confidence_invalid`는 현재 결과에 없다): `conflicting_pair`(같은 쌍에 반대 방향·다른 라벨 존재), `temporal_pair_conflict`, `temporal_cycle_broken`(시간 모순으로 제외), `description_endpoint_mismatch`(설명이 양 끝 사건과 맞지 않음), `noncore_label`, `label_typo_suspected`, `evidence_kind_invalid`, `cue_not_in_text`, `confidence_invalid`

### 4.11 `events_new/<작품>/<실행시각>/summary.json`

- 형식: JSON
- 주요 필드
  - 실행 정보: `status`, `extractor_version`(`event.v6`), `model`, `cleaned`, `chunks`, `work_id`
  - 옵션: `gleaning`, `repair`, `verify`, `cross_chunk`
  - 입력: `input_chunks`, `document_sha256`, `chunks_sha256`
  - 결과: `events`, `event_edges`, `failed_units`, `event_type_counts`, `narrative_level_counts`, `verdict_counts`, `relation_type_counts`, `edge_scope_counts`, `event_flag_counts`, `edge_flag_counts`, `temporal_invalid_edges`, `flagged_events`, `flagged_edges`
  - `stats`: `raw_events`, `events_rejected`, `events_repaired`, `events_removed_by_verify`, `events_revised_by_verify`, `events_overlap_merged`, `raw_relations`, `relations_rejected`, `api_calls`, `input_tokens`, `output_tokens` 등

### 4.12 `data/entity_event_new/<작품>.entity_event_edges.jsonl`

- 형식: JSONL(한 줄이 Entity observation–Event 연결 하나). 모든 필드가 필수다.

| 필드명 | 의미 | 값 예시 |
|---|---|---|
| `edge_id` | 연결 ID(observation ID와 Event ID로 생성) | `"evlink_07988b749de13c2e"` |
| `work_id` | 작품 ID | `"1_눈의 여왕"` |
| `source_observation_id` | 연결된 Entity observation | `"eobs_f3c7e5c0c1e198c7"` |
| `target_event_id` | 연결된 Event | `"evt_4a7bc562409bf94a"` |
| `edge_type` | 항상 `"entity_event_relation"` | `"entity_event_relation"` |
| `entity_name` | observation 이름(조회 편의용 사본) | `"거울"` |
| `entity_type` | observation 유형(사본, `null` 가능) | `"object"` |
| `event_label` | Event 이름(사본) | `"거울 조각이 사람들에게 영향을 미치다"` |
| `chunk_id` | 연결이 성립한 chunk(observation의 chunk) | `1` |
| `available_at_chunk` | max(observation τ, Event τ) | `1` |
| `methods` | 성립한 연결 방법 목록(2.4 표 참고) | `["name_in_description", …]` |
| `paper_method` | `name_in_description` 포함 여부(원논문 식 (1)) | `true` |
| `official_impl_method` | `name_in_description` 또는 `name_in_label` 포함 여부(공식 구현) | `true` |
| `matched_mention_ids` | `mention_in_evidence`·`participant`로 맞은 mention ID | `["m_7055…"]` |
| `name_match_bounded` | 설명 포함이 단어 경계에 맞는지. 설명 포함(`name_in_description`)이 없으면 `null` | `true` |
| `recommended` | 그래프 구성에 권장하는 연결인지(2.4 참고) | `true` |

한 Event가 두 chunk에 걸쳐 있으면(`occurrence_chunk_ids`가 2개) 각 chunk의 observation과 따로 연결된다.

### 4.13 `data/entity_event_new/<작품>.entity_event_edges.summary.json`

- 형식: JSON

| 필드명 | 의미 |
|---|---|
| `work_id`, `entities`, `event_run` | 작품 ID와 실행에 쓴 입력 경로 |
| `edges` | 전체 연결 수 |
| `recommended`, `not_recommended` | 권장·비권장 연결 수 |
| `only_paper_method` | 원문 좌표 근거 없이 이름 포함으로만 성립한 연결 수 |
| `only_paper_method_unbounded` | 그중 단어 경계도 맞지 않는 연결 수 |
| `only_extension_methods` | `name_in_description` 없이 성립한 연결 수 |
| `label_only_official` | `name_in_label`은 있고 `name_in_description`은 없는 연결 수 |
| `method:<방법>` | 방법별 연결 수 |
| `events`, `events_without_entity` | Event 수, 연결이 하나도 없는 Event 수 |
| `entity_observations`, `entity_observations_without_event` | observation 수, 연결이 없는 observation 수 |
| `events_without_recommended_entity` | 권장 연결이 없는 Event 수 |

---

## 5. 코드별 입력·출력 연결 관계 및 실행 순서

### 5.1 연결 표

**직접 생성**은 코드가 만드는 파일이다. **입력**은 실제로 읽는 파일이다.
**참고만**은 경로만 받아 해시 검증에만 쓰거나 다른 파일 안의 경로 값만 기록하는 경우다.

| 실행 코드 | 입력 파일 | 출력 파일(직접 생성) | 결과 의미 |
|---|---|---|---|
| `src/preprocessing/preprocess_text.py` | `data/raw/<작품>.txt` | `data/cleaned/<작품>.txt` | 본문만 남긴 원문 |
| `src/chunking/split_jsonl_by_chunk.py` | `data/cleaned/*.txt` | `data/chunk/<작품>.jsonl` | 1200/100토큰 chunk |
| `extract_entity_mentions_v5.py` | `data/cleaned/<작품>.txt`, `data/chunk/<작품>.jsonl`, `.env` | `data/entity_mentions_new/<작품>/<실행시각>/` 안의 `summary.json`, `entities.jsonl`, `accepted_observations.jsonl`, `coverage.jsonl`, `review.jsonl`, `audit.jsonl`, `calls.jsonl` | 원문 위치 단위 mention과 실행 기록 |
| `convert_mentions_to_observations.py` | `.../<실행시각>/summary.json`, `.../<실행시각>/entities.jsonl`, `data/cleaned/<작품>.txt`, `data/chunk/<작품>.jsonl` | `data/entity_observation/<작품>.entities.jsonl` | (chunk, 이름) 단위 Entity observation |
| `extract_events_v1.py` | `data/cleaned/<작품>.txt`, `data/chunk/<작품>.jsonl`, `.env` | `data/events_new/<작품>/<실행시각>/` 안의 `summary.json`, `events.jsonl`, `event_edges.jsonl`, `review.jsonl`, `audit.jsonl`, `calls.jsonl` | Event, Event-Event edge, 실행 기록 |
| `link_entity_event_v1.py` | `data/entity_observation/<작품>.entities.jsonl`, `.../<실행시각>/summary.json`, `.../<실행시각>/events.jsonl` | `data/entity_event_new/<작품>.entity_event_edges.jsonl`, `<작품>.entity_event_edges.summary.json` | Entity-Event 연결과 통계 |

**후속 코드가 읽지 않는 파일**

다음 파일은 현재 `src/extraction`의 어떤 코드도 입력으로 읽지 않는다.

- 검토용 기록: `accepted_observations.jsonl`, `coverage.jsonl`, `review.jsonl`, `audit.jsonl`, `calls.jsonl`
- 다음 단계용 결과: `event_edges.jsonl`

`link_entity_event_v1.py`는 `cleaned`와 `chunk`를 직접 열지 않는다. 대신 observation과 `events/summary.json`에 기록된 해시끼리 비교해서 두 결과가 같은 원문에서 나왔는지 확인한다.

**코드 간 import**

`convert_mentions_to_observations.py`와 `extract_events_v1.py`는 실행 시 같은 폴더의 `extract_entity_mentions_v5.py`를 import한다. 입력 검증(`load_inputs`), 해시(`digest`), 파일 기록(`append`, `dump`), API 클라이언트(`make_client`)를 재사용하기 때문이다. 이 파일을 옮기거나 이름을 바꾸면 두 코드가 실행되지 않는다.

### 5.2 실행 순서

```text
1. preprocess_text.py            raw → cleaned
2. split_jsonl_by_chunk.py       cleaned → chunk
3. extract_entity_mentions_v5.py ┐ 서로 독립적이며 순서는 상관없다
4. extract_events_v1.py          ┘
5. convert_mentions_to_observations.py   (3번 결과 필요)
6. link_entity_event_v1.py               (4번과 5번 결과 필요)
```

### 5.3 실행 명령 형식 (Windows, 프로젝트 루트에서)

`<작품>`은 `1_눈의 여왕`처럼 확장자 없는 파일 이름이다.
`<실행시각>`은 3·4번 실행이 끝날 때 출력되는 `결과 폴더:` 값의 마지막 폴더 이름이다.

```bat
py src\extraction\extract_entity_mentions_v5.py --project-root . --cleaned "data\cleaned\<작품>.txt" --chunks "data\chunk\<작품>.jsonl"
py src\extraction\extract_events_v1.py --project-root . --cleaned "data\cleaned\<작품>.txt" --chunks "data\chunk\<작품>.jsonl"
py src\extraction\convert_mentions_to_observations.py --run-dir "data\entity_mentions_new\<작품>\<실행시각>" --cleaned "data\cleaned\<작품>.txt" --chunks "data\chunk\<작품>.jsonl"
py src\extraction\link_entity_event_v1.py --entities "data\entity_observation\<작품>.entities.jsonl" --event-run "data\events_new\<작품>\<실행시각>"
```

- `convert`와 `link`는 출력 파일이 이미 있으면 멈춘다. 다시 만들려면 `--overwrite`를 붙인다.
- 두 추출기 모두 `--check-only`를 주면 API를 호출하지 않고 입력 정합성만 검사한다.
