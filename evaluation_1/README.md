# evaluation_1 실행 방법

기존 `evaluation_1` 폴더를 결과와 함께 삭제한 뒤 이 ZIP을 프로젝트 루트에 압축 해제한다. 이전 evaluation 코드나 결과를 가져오지 않는다. `src`와 `data`는 그대로 둔다.

## 순서대로 실행

프로젝트 루트 `.env`의 `OPENAI_API_KEY`를 사용한다.

```bat
cd /d D:\26_capstone\e2rag-character-graph
py -m pip install -r evaluation_1\requirements.txt
py evaluation_1\run.py check
py evaluation_1\run.py prepare --assets method_v1
py evaluation_1\run.py generate --assets method_v1 --run method_v1
py evaluation_1\run.py judge --assets method_v1 --run method_v1
py evaluation_1\run.py report --assets method_v1 --run method_v1
```

- check: QA 134건과 원문·좌표·기존 그래프/인덱스 해시 확인. API 호출 없음.
- prepare: 기존 Entity와 chunk로 기본 Entity–Entity 관계만 추가 추출. S2·S4 공통 그래프와 인덱스 생성.
- generate: 세 시스템의 답변 402개 새로 생성.
- judge: 세 평가자가 각 문항의 세 답변을 익명으로 판정.
- report: 문항별·유형별·작품별·시스템별 결과 자동 집계.

같은 코드 실행 중 네트워크 오류 등으로 미완료 항목이 남으면 같은 명령으로 다시 실행할 수 있다. 이는 이전 버전 결과를 가져오는 기능이 아니다. 오류는 화면과 `failures.jsonl` 및 `calls.jsonl`에 기록한다. 인증·모델 지원 오류는 즉시 중단한다. 모델은 자동 대체하지 않는다.

## 이번 수정

1. S2 보완: 기본 Entity–Entity 관계 추가. paper_method=True 연결 사용. Entity/Event 이름+설명 임베딩. Event 복수 등장 chunk 반영. 관계 설명과 부분 그래프 보존. S4도 공통 처리를 사용하며 Relation Observation과 p 제한은 유지.
2. 미래 채점: 모델은 abstained·spoiler·reason만 반환. A/B/C 판정과 자료형을 검증한 뒤 코드에서 score·correct를 null로 저장.

관계 추출의 근거는 모델이 원문을 복사하지 않는다. 코드가 원문에 S1, S2 등의 문장 번호를 부여하고 모델이 번호를 선택한다. 코드가 선택한 문장 구간에서 evidence와 절대 좌표를 직접 만든다. 존재하지 않는 번호와 잘못된 끝점 및 자료형은 거절한다. 여러 문장을 선택하면 첫 문장 시작부터 마지막 문장 끝까지의 연속 원문을 저장한다. 번호 선택의 의미적 타당성까지 규칙 검사로 보장하는 것은 아니다.

S1·S2는 작품 전체를 검색한다. S4만 p까지 검색한다. 세 시스템 모두 생성 단계에서 p를 받는다. 생성 모델과 평가자 및 일반·관계 채점 기준은 유지했다. 원문 전처리와 기존 Entity·Event 결과는 수정하지 않는다.

## 파일 역할

| 파일 | 역할 |
|---|---|
| run.py | 단계 실행과 완료 여부 확인 |
| common.py | 입력 검사·API·저장·임베딩 공통 처리 |
| prepare.py | 문장 번호 기반 기본 관계 추출과 평가 자산 준비 |
| retrieval.py | 공통 검색·문맥 구성·답변 생성 |
| judging.py | 일반/미래 채점 요청과 응답 검증 |
| reporting.py | 이전 evaluation의 집계 코드 유지 |
| systems/vanilla.py | S1 검색 진입점 |
| systems/e2rag.py | S2 검색 진입점 |
| systems/proposed.py | S4 검색 진입점 |
| systems/__init__.py | 시스템 패키지 |
| config.json | 모델과 토큰·검색 설정 |
| requirements.txt | 필요한 패키지 |

## 결과 확인

- `evaluation_1/assets/method_v1/`: 추가 관계·공통 그래프·평가 인덱스.
- `evaluation_1/runs/method_v1/answers/`: 시스템별 실제 답변.
- `evaluation_1/runs/method_v1/judges/`: 평가자별 판정.
- `evaluation_1/runs/method_v1/reports/report.html`: 답변과 평가 결과 보기.
- `evaluation_1/runs/method_v1/reports/questions_wide.csv`: 문항별 정답·세 답변·평가자별 점수.
- `evaluation_1/runs/method_v1/reports/summary.csv`: 전체·유형별·작품별 집계.

평균 점수는 일반 75 + 관계 변화 26 = 101문항에 대해 세 평가자의 평균을 계산한다. 미래 33문항은 유보 성공과 스포일러 비율로 별도 집계한다. 관계 변화 13쌍의 쌍 정확도와 정답률도 집계한다.

검증: 실제 프로젝트 데이터 구조와 전체 49 chunk의 문장 좌표를 확인했다. 잘못된 문장 번호 거절과 오류 재시도 및 전체 생성·채점·집계를 모의 API로 확인했다. 실제 API 추론은 제작 환경에서 실행하지 않았다. 테스트 코드와 결과 파일은 배포 ZIP에 넣지 않았다.
