# 8단계 Embedding 및 Vector Index

이 모듈은 기존 추출 결과를 읽어 Entity Observation, Event, Relation Observation, Chunk를 각각 NumPy 인덱스로 만든다. 기존 추출 파일과 ID 및 τ는 변경하지 않는다. 그래프 구성과 one-hop 확장 및 답변 생성은 구현하지 않는다.

## 설치와 파일 배치

제공 ZIP을 `D:\26_capstone\e2rag-character-graph`에 풀어 `src\embedding`을 추가한다. 기존 파일을 교체할 필요는 없다. `requirements-embedding.txt`와 `data\vector_index\.gitignore`도 프로젝트에 추가한다.

```bat
cd /d D:\26_capstone\e2rag-character-graph
py -m pip install -r requirements-embedding.txt
```

루트 `.env`는 다음 키를 포함해야 한다. API 키는 코드나 결과에 저장하지 않는다. 이미 설정된 환경변수가 있으면 환경변수가 우선한다.

```dotenv
OPENAI_API_KEY=사용자_API_키
```

모델은 `text-embedding-3-small`이고 차원은 1536이다. OpenAI Embeddings API로 생성한 벡터를 float32로 변환한 뒤 L2 정규화한다. 동일한 벡터 공간을 유지하기 위해 검색 질문에도 같은 모델과 차원을 사용한다. 현재 데이터의 구축은 임베딩 API만 호출하며 GPT 답변 생성 API는 호출하지 않는다.

## 5편 구축 명령

아래 명령은 프로젝트 루트에서 순서대로 실행한다. 같은 캐시에 두 프로세스를 동시에 쓰지 않는다.

```bat
py src\embedding\build_vector_index.py --work-id "1_눈의 여왕"
py src\embedding\build_vector_index.py --work-id "2_엄지 공주"
py src\embedding\build_vector_index.py --work-id "3_인어 공주"
py src\embedding\build_vector_index.py --work-id "4_성냥팔이 소녀"
py src\embedding\build_vector_index.py --work-id "5_미운 아기 오리"
```

Event와 Relation 실행 폴더가 각각 하나일 때 자동으로 선택한다. 둘 이상이면 임의로 최신 폴더를 선택하지 않고 중단한다. 그때는 정확한 JSONL 파일을 `--events`, `--relations`로 지정한다. 최신 ZIP에서 눈의 여왕의 명령은 다음과 같다.

```bat
py src\embedding\build_vector_index.py --work-id "1_눈의 여왕" --events "data\events_new\1_눈의 여왕\20261014T132234_966945Z\events.jsonl" --relations "data\relation_observation\1_눈의 여왕\20261016T134524_574877Z\relation_observations.jsonl"
```

## API 없이 입력 검증

`--prepare-only`는 4종 검색 레코드를 메모리에서 만들고 원본 ID와 해시 및 좌표와 경계를 확인한다. 벡터는 만들지 않는다. 이 옵션에는 API 키가 필요 없다.

```bat
py src\embedding\build_vector_index.py --work-id "1_눈의 여왕" --prepare-only
py src\embedding\build_vector_index.py --work-id "2_엄지 공주" --prepare-only
py src\embedding\build_vector_index.py --work-id "3_인어 공주" --prepare-only
py src\embedding\build_vector_index.py --work-id "4_성냥팔이 소녀" --prepare-only
py src\embedding\build_vector_index.py --work-id "5_미운 아기 오리" --prepare-only
```

## 완성 인덱스 검증

`--check-only`는 원본 입력에서 metadata를 재생성한 뒤 저장된 내용과 대조한다. 벡터나 질문을 새로 임베딩하지 않는다. 모든 독서 진행도에서 모든 저장 문서 벡터를 검색 질의로 사용한다. 각 검색의 전체 반환 ID 집합이 해당 p에서 공개된 원본 ID 집합과 같은지 검사한다. 자기 문장과 같은 벡터가 여러 개이면 동점을 허용한다. 의미 검색 품질 평가는 별개다.

```bat
py src\embedding\build_vector_index.py --work-id "1_눈의 여왕" --check-only
py src\embedding\build_vector_index.py --work-id "2_엄지 공주" --check-only
py src\embedding\build_vector_index.py --work-id "3_인어 공주" --check-only
py src\embedding\build_vector_index.py --work-id "4_성냥팔이 소녀" --check-only
py src\embedding\build_vector_index.py --work-id "5_미운 아기 오리" --check-only
```

## 검색

자연어 질문 검색은 질문을 임베딩하므로 새 질문에는 API 호출이 발생한다. 재질의는 캐시를 재사용한다. `e2rag`는 Entity, Event, Relation 세 저장소를 각각 검색한다. cue 추출과 one-hop 확장은 후속 모듈의 역할이다. 이 명령은 seed 검색까지만 수행한다.

```bat
py src\embedding\vector_index.py --work-id "1_눈의 여왕" --store e2rag --p 5 --query "게르다는 카이를 어떻게 생각하나요?" --top-k 5
py src\embedding\vector_index.py --work-id "1_눈의 여왕" --store relation --p 5 --query "두 인물의 관계는 어떤가요?" --top-k 5
py src\embedding\vector_index.py --work-id "1_눈의 여왕" --store chunk --p 5 --query "게르다는 카이를 어떻게 생각하나요?" --top-k 5
```

Chunk는 일반 RAG 기준선 전용이며 `search_e2rag`에는 포함되지 않는다. Entity/Event/Relation의 top-k는 각 저장소별 개수다. 결과가 없더라도 미래 범위로 검색을 넓히지 않는다. 검색 점수는 의미적 유사도이며 관계의 진실성 확률이 아니다.

기존 문서 벡터를 사용하는 구조 점검에는 API가 필요 없다. 다음 ID는 최신 ZIP의 실제 ID다. 코드 자체에는 인물이나 작품별 ID가 하드코딩되지 않는다.

```bat
py src\embedding\vector_index.py --work-id "1_눈의 여왕" --store event --p 1 --self-id evt_310f7704f6a33037 --top-k 5
py src\embedding\vector_index.py --work-id "1_눈의 여왕" --store chunk --p 1 --self-id 1 --top-k 5
```

## 실제 입력과 좌표

| 대상 | 입력 | 원본 ID |
|---|---|---|
| Entity | `data/entity_observation/<작품>.entities.jsonl` | `observation_id` |
| Event | `data/events_new/<작품>/<실행 폴더>/events.jsonl` | `event_id` |
| Relation | `data/relation_observation/<작품>/<실행 폴더>/relation_observations.jsonl` | `relation_observation_id` |
| Chunk | `data/chunk/<작품>.jsonl` | 정수 `chunk_id` |

각 원본 ID는 같은 타입으로 `node_id`에 저장된다. Chunk ID는 다른 저장소와 구분되는 정수이며 전역 이름을 새로 붙이지 않는다. Entity와 Event 및 Relation의 연결 엣지는 임베딩하지 않는다. 임베딩 구축에 그래프 결과가 필요하지 않다.

원문은 `data/cleaned/<작품>.txt`를 사용한다. 기존 `extract_entity_mentions_v5.load_inputs`의 토큰→UTF-8→문자 좌표 매핑을 그대로 호출한다. 불완전한 UTF-8 경계 문자를 복사하지 않고 `view`에 대응하는 cleaned 원문을 사용한다. 별도의 근사 문자열 검색으로 chunk 좌표를 추정하지 않는다.

- `t`: 독자가 현재 읽는 chunk의 위치. 인덱스의 저장 위치로 재정의하지 않는다.
- `τ`: 해당 노드나 관계가 나타난 위치. `tau`와 `available_at_chunk`에 원본 값을 보존한다.
- `p`: 검색 요청의 읽기 진행 위치. 이번 함수에서는 공개 완료 chunk 번호로 받는다.
- `read_end_char(p)`: chunk 1..p가 공개한 cleaned 원문의 끝 문자 위치.
- `[a,b)`: cleaned를 Python 문자열로 읽은 뒤 a부터 b 직전까지의 문자 구간. byte offset이 아니다.

페이지나 chunk 중간 위치를 쓰는 UI는 후속 단계에서 원문 위치로 변환해야 한다. 현재 `--p`를 현재 chunk를 읽기 시작했다는 의미로 전달하면 안 된다.

## 근거와 description 처리

Entity의 상위 description/evidence와 대응하는 대표 mention을 찾고 해당 근거 좌표를 검증한다. 대표 근거가 경계를 넘으면 공개 범위 안에서 끝나며 해당 mention 이름 구간을 포함하는 다른 근거를 찾는다. 대체 근거를 쓰면 그 mention의 description을 사용한다. 기존 대표 description은 가져오지 않는다.

대체 근거가 없으면 대표 근거를 `read_end_char(τ)`에서 자른다. description은 빈 문자열로 만들고 `evidence_partial=true`, `description_used=false`로 기록한다. 이 처리는 작품 이름이나 오류 목록을 사용하지 않는다. 원본 파일을 수정하지도 않는다.

Event/Relation은 대표 원문 근거와 해당 설명만 사용한다. 현재 데이터는 두 종류 모두 근거 전체가 소속 chunk 안에 있다. 이후 입력이 이 조건을 위반하면 추측해서 수정하지 않고 중단한다. `occurrences`, `occurrence_chunk_ids`, 연결 LLM의 verification_reason과 이후 이력은 임베딩 및 검색용 metadata에 복사하지 않는다.

**자동 검사 범위:** description에 대응하는 원본 근거의 공개 범위와 출처를 검사한다. 기존 LLM 설명의 모든 의미가 근거에서 함의되는지 새로 판정하지 않는다. 임의의 환각 또는 간접적 스포일러를 결정적 좌표 검사만으로 완전히 검증했다고 주장하지 않는다. 근거가 잘린 경우 기존 description이 남지 않는 것은 자동 검사한다. 최종 QA 문맥과 답변의 공개 범위 검사는 QA 단계에서 다시 수행해야 한다.

## 출력

```text
data/vector_index/
  .gitignore
  _cache/embeddings.json
  _checks/<작품>.prepare.json
  <작품>/
    entity.npy
    entity.meta.jsonl
    event.npy
    event.meta.jsonl
    relation.npy
    relation.meta.jsonl
    chunk.npy
    chunk.meta.jsonl
    manifest.json
    verify_report.json
```

- `.npy`: `(레코드 수, 1536)`인 float32 정규화 행렬이다.
- `.meta.jsonl`: i번째 객체가 `.npy` i번째 행에 대응한다.
- `manifest.json`: 입력 상대 경로/파일 해시와 코드 해시, 모델, 차원, 저장소별 건수, 읽기 경계, 결과 해시를 저장한다. 같은 입력과 코드이면 build_id가 같다.
- `verify_report.json`: 실제 완성 벡터와 원본 데이터를 대조한 검증 결과다.
- `_checks`: API 없는 입력 검사 결과다. 임베딩 완료 보고서가 아니다.
- `_cache`: 모델명/차원/검색 문장 해시에 대응하는 원본 API 벡터를 보관한다. 임베딩 비용 재사용용이며 검색 인덱스의 채택 레코드를 대체하지 않는다.

공통 metadata에는 `node_id`, `node_type`, `work_id`, `tau`, `available_at_chunk`, `chunk_id`, `evidence`, `evidence_original_span`, `evidence_partial`, `alternative_evidence`, `description`, `description_used`, `description_basis_span`, `description_basis_complete`, `read_end_char_tau`, `embed_text`, `text_sha256`, `needs_review`, `flags`가 있다. description 의미 판정을 수행하지 않았다는 점을 `description_check`에도 기록한다.

Entity에는 이름/name_key/유형/선택한 mention ID/overlap 표시가 추가된다. Event에는 label/event_type/narrative_level이 추가된다. Relation에는 원본 source/target Observation ID와 이름 및 name_key, 관계 유형, 방향, polarity, modality, narrative_level이 추가된다. null 관계 유형은 삭제하거나 임의로 채우지 않고 표시한다. Chunk에는 `read_end_char`가 추가된다.

## 주요 함수

| 파일 | 함수·클래스 | 역할 |
|---|---|---|
| build_vector_index.py | require, norm | 검증 예외와 비교용 공백 정리 |
| build_vector_index.py | file_info, one_file | 입력 경로/버전 기록과 모호한 실행 폴더 차단 |
| build_vector_index.py | prepare | 입력 해시/좌표/ID 검사와 4종 검색 레코드 생성 |
| build_vector_index.py | validate | 원본 재생성 metadata 비교와 모든 p에서 검색 검증 |
| build_vector_index.py | build | 입력 검사/캐시 호출/저장/검증과 재실행 재사용 |
| build_vector_index.py | main | 구축·입력 검사·완성 검증 CLI |
| vector_index.py | digest, read_jsonl, write_json | 해시, JSONL 읽기, JSON 원자적 저장 |
| vector_index.py | normalize | 벡터 형식 검사와 L2 정규화 |
| vector_index.py | Embedder.encode | 배치/토큰 제한/재시도/응답 순서 검증/캐시 |
| vector_index.py | VectorIndex | 한 작품의 파일 무결성 검사와 인덱스 로딩 |
| vector_index.py | check_progress, read_end | p의 단위/범위 검사와 원문 공개 끝점 조회 |
| vector_index.py | search | 단일 저장소에서 τ≤p 사전 필터 후 정확 검색 |
| vector_index.py | search_e2rag | Chunk를 제외한 세 저장소의 seed 후보 검색 |
| vector_index.py | main | 질문 또는 기존 벡터로 검색하는 CLI |

## 9단계와 QA 연동

프로젝트 루트에서 다음처럼 import할 수 있다.

```python
from src.embedding.vector_index import VectorIndex
index = VectorIndex(r"D:\26_capstone\e2rag-character-graph", "1_눈의 여왕")
seeds = index.search_e2rag(p=5, query_text="현재 인물들의 관계는 어떤가요?", top_k=5)
```

원논문식 cue를 사용하려면 후속 모듈에서 생성한 Entity cue를 entity 저장소로, Event cue를 event 저장소로 각각 전달한다. `search_e2rag`의 단일 질문 편의 함수가 원논문의 cue 추출을 수행하는 것은 아니다.

검색 결과는 원본 ID와 score 및 tau와 안전한 근거를 포함한다. 요청의 p와 `read_end_char_p`, 인덱스 build_id도 포함한다. 그래프 담당자는 `node_type`과 `work_id` 및 `node_id`로 원본 노드를 찾으면 된다. Relation 끝점 ID도 그대로 사용할 수 있다. name_key가 같다는 이유로 동일 인물이라고 확정하거나 Observation을 병합하지 않는다.

그래프 확장은 후속 코드에서 수행한다. 확장된 노드뿐 아니라 **엣지 자체의 available_at_chunk**도 p 이하여야 한다. 양 끝점의 τ 최댓값만으로 엣지 공개 위치를 낮추면 안 된다. Relation–Event는 최종 `relation_event_links.jsonl`을 사용한다. 임베딩 유사도만으로 새 관계나 인과 연결을 만들지 않는다.

QA는 안전한 검색 결과를 받더라도 원본 노드의 전체 occurrences를 그대로 추가하면 안 된다. 최종 passage/노드 설명/엣지를 다시 공개 범위로 제한한다. Chunk 기준선은 `search("chunk", ...)`를 별도 호출한다. QA 평가셋의 정답이나 future_evidence는 인덱스 구축과 검색에 입력하지 않는다.

## 재실행과 실패 복구

동일한 입력/코드의 완성 인덱스가 있으면 원본과 저장 결과를 다시 검증한 뒤 API 없이 재사용한다. 인덱스 폴더를 지우더라도 `_cache`를 유지하면 같은 문장의 벡터는 재호출하지 않는다. 입력 또는 텍스트 규칙이 바뀌면 해당 텍스트의 미캐시 벡터만 요청한다. 파일 개수와 행을 append하지 않으므로 레코드가 중복되지 않는다.

재시도 대상은 연결/timeout과 HTTP 408/409/429 및 5xx다. 최대 5회 시도하고 1/2/4/8초 간격으로 대기한다. 인증 실패와 잘못된 입력은 반복하지 않는다. 배치 크기는 기본 64이며 배치 합계 200,000토큰을 넘지 않도록 나눈다. 개별 입력이 8,191토큰을 넘으면 자르지 않고 중단한다.

임시 `.building`에 저장한 뒤 교체하고 검증 실패 시 기존 인덱스를 복원한다. 강제 종료 후 `.building`이나 `.backup`이 남으면 동시 실행 여부를 확인한다. `.backup`은 이전 정상 결과일 수 있으므로 무조건 삭제하지 않는다. 완성 결과를 확인한 뒤 복원하거나 임시 폴더만 정리한다. 완성 manifest가 있는데 `.npy`가 손상되면 검증이 실패한다. 그 작품 인덱스 폴더만 다른 위치로 옮긴 뒤 다시 구축하면 캐시를 재사용한다.

## Git 공유

포함: 두 Python 파일, README, requirements-embedding.txt, vector_index/.gitignore, 작품별 4종 .npy와 metadata, manifest.json, verify_report.json.

제외: API 키/.env, __pycache__, `_cache`, `_checks`, `.building`, `.backup`, `.tmp`.

제공 ZIP의 .gitignore는 `data/vector_index` 내부에만 적용된다. 기존 루트 .gitignore의 `.env`, `__pycache__/` 규칙은 유지한다. 벡터/metadata/manifest를 부분적으로 섞어서 공유하지 않는다. 빌드가 끝나면 한 커밋에 함께 포함한다.

```bat
git add src/embedding requirements-embedding.txt data/vector_index
git status
```

## 이번 제공 과정의 검증 상태

별도 `validation_results.json`과 검증 요약을 참고한다. 실제 원문과 최신 ZIP의 5편 입력을 사용해 준비 검사를 수행했다. 환경에 OPENAI_API_KEY가 없어 실제 모델 임베딩은 생성하지 않았다. 테스트에서 OpenAI 응답 형식을 흉내 낸 결정적 난수 벡터로 캐시와 배치, 저장, 검색, 재실행, 훼손 차단을 검사했다. 그 벡터는 실제 임베딩이 아니며 배포 ZIP에 포함하지 않는다.

사용자 환경에서 5편 구축 명령을 실행한 뒤 verify_report.json의 passed 여부와 자연어 검색 품질을 확인해야 한다. 유사도나 QA 성능의 실측 결과는 이번에 제시하지 않는다.
