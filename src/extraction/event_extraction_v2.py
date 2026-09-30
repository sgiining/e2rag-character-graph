"""chunk별 사건과 사건 간 관계를 추출합니다.

입력: data/chunk/<work>.jsonl
출력: data/event/<work>.events.jsonl
      data/event/<work>.event_edges.jsonl
      data/event/<work>.event_rejections.jsonl

완료된 작업은 체크포인트에 저장합니다.
중단 후 같은 명령으로 실행하면 남은 작업을 이어서 처리합니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import (
    OpenAI,
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)


EVENT_TYPES = [
    "action",
    "interaction",
    "conflict",
    "support",
    "occurrence",
    "other",
]

# 허용 목록이 아니라 원본 GitHub 프롬프트에 제시된 관계 유형의 예시입니다.
RELATION_TYPE_EXAMPLES = [
    "causes",
    "precedes",
    "triggers",
    "influences",
    "parallels",
]

SCHEMA_VERSION = "event.v3"


SYSTEM = """한국어 문학 원문에서 사건과 방향이 있는 사건 간 관계를 추출합니다.

유효한 JSON만 반환합니다.
JSON 바깥에 설명이나 코드블록을 붙이지 않습니다.

사건명과 사건 설명 및 관계 설명은 한국어로 작성합니다.
JSON 키와 event_type 및 relation_type 값은 영어로 작성합니다.

제공된 원문에 없는 사건이나 참여자 및 시간 순서와 인과관계를 만들지 않습니다.
원문에 명시되어 있거나 문맥에서 분명하게 뒷받침되는 관계를 추출합니다.
같은 참여자가 등장하거나 사건이 연속해서 서술된다는 이유만으로 인과관계를 만들지 않습니다.
원문에서 언급된 순서와 이야기 속에서 실제로 발생한 시간 순서를 구분합니다.
관계의 존재나 방향이 불확실하면 해당 관계를 제외합니다.

relation_type의 예시는 고정된 허용 목록이 아닙니다.
원문에서 확인되는 연결에 적합한 다른 세부 유형도 사용할 수 있습니다.
같은 관계를 반대 방향의 동의 표현으로 중복 출력하지 않습니다.

근거 구절은 원문에서 그대로 인용합니다.
"""


CHUNK_PROMPT = """현재 chunk에서 개별 사건을 추출합니다.
서로 다른 시점에 실제로 발생한 별개의 사건을 하나로 합치지 않습니다.

chunk ID: {chunk_id}
토큰 구간: {token_start}-{token_end}

각 사건에 다음 정보를 포함합니다.

- local_id: 응답 안에서 중복되지 않는 사건 ID. E1, E2와 같이 작성합니다.
- label: 사건의 짧은 이름
- event_type: {event_types} 중 하나
- summary: 현재 chunk에서 확인되는 사건의 설명
- evidence: 사건을 뒷받침하는 원문 인용문
- participant_names: 원문에 명시된 참여자의 이름이나 호칭을 담은 배열

추출한 사건 중 서로 분명하게 관련된 사건 쌍을 연결합니다.
현재 응답에 있는 사건의 local_id만 사용합니다.

각 관계에 다음 정보를 포함합니다.

- source_local_id: 관계가 시작되는 사건 ID
- target_local_id: 관계가 향하는 사건 ID
- relation_type: 연결의 성격을 나타내는 영어 라벨
  예시: {relation_type_examples}. 이 목록에 제한되지 않습니다.
- description: 두 사건이 어떻게 연결되는지 설명한 한국어 문장
- evidence_kind: 명시된 관계는 "explicit". 문맥상 분명한 관계는 "implicit".
- confidence: 판단의 확신 정도를 나타내는 0부터 1 사이의 숫자
- evidence_spans: 근거 인용문과 출처 chunk_id를 담은 배열

떨어진 원문 구절을 하나로 이어 붙이지 말고 별도 근거 항목으로 작성합니다.
사건이나 관계가 없으면 해당 배열을 비워 둡니다.

다음 JSON 구조로만 반환합니다.

{{
    "events": [
        {{
            "local_id": "E1",
            "label": "...",
            "event_type": "...",
            "summary": "...",
            "evidence": "...",
            "participant_names": ["..."]
        }},
        {{
            "local_id": "E2",
            "label": "...",
            "event_type": "...",
            "summary": "...",
            "evidence": "...",
            "participant_names": ["..."]
        }}
    ],
    "relations": [
        {{
            "source_local_id": "E1",
            "target_local_id": "E2",
            "relation_type": "causes",
            "description": "...",
            "evidence_kind": "explicit",
            "confidence": 0.9,
            "evidence_spans": [
                {{
                    "chunk_id": {chunk_id},
                    "quote": "..."
                }}
            ]
        }}
    ]
}}

원문:
{chunk_text}
"""


CROSS_PROMPT = """인접한 두 chunk에서 추출한 사건 사이의 관계를 판단합니다.

각 관계는 왼쪽 사건 하나와 오른쪽 사건 하나를 연결해야 합니다.
같은 chunk의 사건끼리는 연결하지 않습니다.
원문 근거에 따라 양쪽 방향 모두 허용합니다.

제공된 사건 ID만 사용하며 새로운 사건이나 ID를 만들지 않습니다.
요약뿐 아니라 아래 원문을 확인해 판단합니다.
근거가 부족하면 해당 관계를 제외합니다.

각 관계에 다음 정보를 포함합니다.

- source_event_id: 관계가 시작되는 사건 ID
- target_event_id: 관계가 향하는 사건 ID
- relation_type: 연결의 성격을 나타내는 영어 라벨
  예시: {relation_type_examples}. 이 목록에 제한되지 않습니다.
- description: 두 사건이 어떻게 연결되는지 설명한 한국어 문장
- evidence_kind: 명시된 관계는 "explicit". 문맥상 분명한 관계는 "implicit".
- confidence: 판단의 확신 정도를 나타내는 0부터 1 사이의 숫자
- evidence_spans: 근거 인용문과 실제 출처 chunk_id를 담은 배열

떨어진 원문 구절은 별도 근거 항목으로 작성합니다.
확인되는 관계가 없으면 relations를 빈 배열로 반환합니다.

다음 JSON 구조로만 반환합니다.

{{
    "relations": [
        {{
            "source_event_id": "...",
            "target_event_id": "...",
            "relation_type": "causes",
            "description": "...",
            "evidence_kind": "explicit",
            "confidence": 0.9,
            "evidence_spans": [
                {{
                    "chunk_id": {left_id},
                    "quote": "..."
                }}
            ]
        }}
    ]
}}

왼쪽 사건:
{left}

왼쪽 원문 — chunk_id={left_id}:
{left_text}

오른쪽 사건:
{right}

오른쪽 원문 — chunk_id={right_id}:
{right_text}
"""


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def sid(*parts: str, prefix: str) -> str:
    digest = hashlib.sha256(
        "\x1f".join(parts).encode()
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def write_atomic(path: Path, text: str) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(text, encoding="utf-8")
    temp.replace(path)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    write_atomic(
        path,
        json.dumps(state, ensure_ascii=False, indent=2),
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(
        json.dumps(row, ensure_ascii=False) + "\n"
        for row in rows
    )
    write_atomic(path, text)


def required_text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"{key}는 비어 있지 않은 문자열이어야 합니다."
        )

    return value


def checked_spans(
    raw: Any,
    texts: dict[int, str],
) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(
            "근거 목록이 없거나 형식이 잘못되었습니다."
        )

    result = []

    for span in raw:
        if not isinstance(span, dict):
            raise ValueError(
                "근거 항목은 JSON 객체여야 합니다."
            )

        cid = span.get("chunk_id")

        if type(cid) is not int or cid not in texts:
            raise ValueError(
                "근거의 chunk_id가 현재 검사 범위에 없습니다."
            )

        quote = required_text(span, "quote")

        if norm(quote) not in norm(texts[cid]):
            raise ValueError(
                "근거 인용문이 해당 chunk 원문에 없습니다."
            )

        item = {
            "chunk_id": cid,
            "quote": quote,
        }

        if item not in result:
            result.append(item)

    return result


def rejection(
    scope: str,
    location: Any,
    raw: Any,
    reason: str,
) -> dict[str, Any]:
    print(
        f"  제외: {scope} / {location} / {reason}",
        flush=True,
    )

    return {
        "scope": scope,
        "location": location,
        "reason": reason,
        "record": raw,
    }


def call(
    client: OpenAI,
    model: str,
    prompt: str,
    retries: int,
    required_keys: tuple[str, ...],
) -> dict[str, Any]:
    last = None

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": SYSTEM,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            text = response.choices[0].message.content or ""
            text = re.sub(
                r"^```(?:json)?\s*|\s*```$",
                "",
                text.strip(),
                flags=re.I,
            )

            data = json.loads(text)

            if not isinstance(data, dict):
                raise ValueError(
                    "응답은 JSON 객체여야 합니다."
                )

            for key in required_keys:
                if not isinstance(data.get(key), list):
                    raise ValueError(
                        f"{key}는 배열이어야 합니다."
                    )

            return data

        except (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            InternalServerError,
            ValueError,
        ) as exc:
            last = exc

            if attempt + 1 < retries:
                wait = min(30, 2 ** attempt * 2)

                print(
                    f"  재시도 {attempt + 1}/{retries - 1}: "
                    f"{type(exc).__name__}",
                    flush=True,
                )

                time.sleep(wait)

    raise RuntimeError(
        f"API 호출 또는 응답 형식 검사 실패: {last}"
    ) from last


def extract_events(
    rows: list[Any],
    chunk: dict[str, Any],
    work: str,
    model: str,
) -> tuple[dict[str, str], list[dict], list[dict]]:
    cid = chunk["chunk_id"]
    text = chunk["text"]

    counts = Counter(
        row["local_id"]
        for row in rows
        if isinstance(row, dict)
        and isinstance(row.get("local_id"), str)
    )

    local = {}
    events = []
    rejected = []

    for order, raw in enumerate(rows, 1):
        try:
            if not isinstance(raw, dict):
                raise ValueError(
                    "사건 항목은 JSON 객체여야 합니다."
                )

            lid = required_text(raw, "local_id")

            if counts[lid] > 1:
                raise ValueError(
                    "응답 안에서 local_id가 중복되었습니다."
                )

            label = norm(required_text(raw, "label"))
            summary = norm(required_text(raw, "summary"))

            if raw.get("event_type") not in EVENT_TYPES:
                raise ValueError(
                    "허용되지 않은 사건 유형입니다."
                )

            evidence = required_text(raw, "evidence")

            if norm(evidence) not in norm(text):
                raise ValueError(
                    "사건 근거가 원문에 없습니다."
                )

            names = raw.get("participant_names")

            if not isinstance(names, list) or any(
                not isinstance(name, str) or not name.strip()
                for name in names
            ):
                raise ValueError(
                    "participant_names는 문자열 배열이어야 합니다."
                )

            eid = sid(
                work,
                str(cid),
                str(order),
                label,
                norm(evidence),
                prefix="evt",
            )

            events.append({
                "event_id": eid,
                "work_id": work,
                "chunk_id": cid,
                "available_at_chunk": cid,
                "label": label,
                "event_type": raw["event_type"],
                "summary": summary,
                "evidence": evidence,
                "participant_names": list(
                    dict.fromkeys(norm(name) for name in names)
                ),
                "token_start": chunk["token_start"],
                "token_end": chunk["token_end"],
                "extractor": {
                    "model": model,
                    "schema_version": SCHEMA_VERSION,
                },
            })

            local[lid] = eid

        except ValueError as exc:
            rejected.append(
                rejection("event", cid, raw, str(exc))
            )

    return local, events, rejected


def extract_relations(
    rows: list[Any],
    refs: dict[str, str],
    texts: dict[int, str],
    model: str,
    scope: str,
    left_ids: set[str] | None = None,
    right_ids: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    cross = scope == "adjacent_chunks"

    source_key = (
        "source_event_id" if cross else "source_local_id"
    )
    target_key = (
        "target_event_id" if cross else "target_local_id"
    )

    edges = []
    rejected = []

    for raw in rows:
        try:
            if not isinstance(raw, dict):
                raise ValueError(
                    "관계 항목은 JSON 객체여야 합니다."
                )

            source = refs.get(required_text(raw, source_key))
            target = refs.get(required_text(raw, target_key))

            if not source or not target or source == target:
                raise ValueError(
                    "연결 대상이 없거나 자기 자신을 연결합니다."
                )

            if cross and not (
                (
                    source in (left_ids or set())
                    and target in (right_ids or set())
                )
                or (
                    source in (right_ids or set())
                    and target in (left_ids or set())
                )
            ):
                raise ValueError(
                    "양쪽 chunk의 사건을 하나씩 연결해야 합니다."
                )

            # 관계 라벨은 목록과 대조하거나 다른 표기로 변환하지 않습니다.
            relation_type = required_text(
                raw,
                "relation_type",
            )
            description = norm(
                required_text(raw, "description")
            )

            if raw.get("evidence_kind") not in (
                "explicit",
                "implicit",
            ):
                raise ValueError(
                    "evidence_kind는 explicit 또는 implicit이어야 합니다."
                )

            confidence = raw.get("confidence")

            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
            ):
                raise ValueError(
                    "confidence는 0부터 1 사이의 숫자여야 합니다."
                )

            spans = checked_spans(
                raw.get("evidence_spans"),
                texts,
            )

            evidence_key = json.dumps(
                spans,
                ensure_ascii=False,
                sort_keys=True,
            )

            edges.append({
                "edge_id": sid(
                    source,
                    target,
                    relation_type,
                    evidence_key,
                    prefix="ee",
                ),
                "source_event_id": source,
                "target_event_id": target,
                "relation_type": relation_type,
                "description": description,
                "evidence_kind": raw["evidence_kind"],
                "confidence": confidence,
                "evidence": " / ".join(
                    span["quote"] for span in spans
                ),
                "evidence_spans": spans,
                # 판단에 사용한 문맥을 모두 읽은 시점부터 관계를 공개합니다.
                "available_at_chunk": max(texts),
                "scope": scope,
                "extractor": {
                    "model": model,
                    "schema_version": "event_edge.v3",
                },
            })

        except ValueError as exc:
            rejected.append(
                rejection(
                    scope,
                    list(texts),
                    raw,
                    str(exc),
                )
            )

    return edges, rejected


def main(
    input_path: Path,
    out_dir: Path,
    model: str | None,
    timeout: float,
    retries: int,
    reset: bool,
) -> None:
    load_dotenv()

    model = (
        model
        or os.getenv("EXTRACTION_MODEL")
        or "gpt-4o-mini"
    )

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            ".env 또는 환경 변수에서 OPENAI_API_KEY를 찾지 못했습니다."
        )

    if retries < 1 or timeout <= 0:
        raise ValueError(
            "retries는 1 이상이어야 하고 timeout은 양수여야 합니다."
        )

    input_text = input_path.read_text(
        encoding="utf-8-sig"
    )

    chunks = [
        json.loads(line)
        for line in input_text.splitlines()
        if line.strip()
    ]

    if not chunks:
        raise ValueError(
            "입력 JSONL에 chunk가 없습니다."
        )

    for chunk in chunks:
        if not isinstance(chunk, dict):
            raise ValueError(
                "입력의 각 행은 JSON 객체여야 합니다."
            )

        cid = chunk.get("chunk_id")

        if isinstance(cid, str) and cid.isdecimal():
            cid = int(cid)

        if type(cid) is not int or cid < 0:
            raise ValueError(
                "chunk_id는 서사 순서를 나타내는 음이 아닌 정수여야 합니다."
            )

        chunk["chunk_id"] = cid

        required_text(chunk, "text")

        for key in ("token_start", "token_end"):
            if (
                type(chunk.get(key)) is not int
                or chunk[key] < 0
            ):
                raise ValueError(
                    f"{key}는 음이 아닌 정수여야 합니다."
                )

        if chunk["token_end"] < chunk["token_start"]:
            raise ValueError(
                "token_end가 token_start보다 작습니다."
            )

    chunks.sort(
        key=lambda chunk: chunk["chunk_id"]
    )

    chunk_by_id = {
        chunk["chunk_id"]: chunk
        for chunk in chunks
    }

    if len(chunk_by_id) != len(chunks):
        raise ValueError(
            "중복된 chunk_id가 있습니다."
        )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    work = input_path.stem

    checkpoint = (
        out_dir / f"{work}.event_checkpoint.json"
    )
    event_file = (
        out_dir / f"{work}.events.jsonl"
    )
    edge_file = (
        out_dir / f"{work}.event_edges.jsonl"
    )
    rejection_file = (
        out_dir / f"{work}.event_rejections.jsonl"
    )

    signature = sid(
        input_text,
        model,
        SYSTEM,
        CHUNK_PROMPT,
        CROSS_PROMPT,
        json.dumps(EVENT_TYPES),
        json.dumps(RELATION_TYPE_EXAMPLES),
        SCHEMA_VERSION,
        prefix="run",
    )

    state = {
        "signature": signature,
        "completed_chunks": [],
        "completed_pairs": [],
        "events": [],
        "edges": [],
        "rejected_records": [],
    }

    if checkpoint.exists() and not reset:
        state = json.loads(
            checkpoint.read_text(encoding="utf-8")
        )

        if state.get("signature") != signature:
            raise ValueError(
                "입력이나 추출 설정이 다른 체크포인트입니다. "
                "새로 실행하려면 --reset을 사용하세요."
            )

        print(
            f"체크포인트 재개: "
            f"{len(state['completed_chunks'])}/{len(chunks)} chunk 완료",
            flush=True,
        )

    # 사건이 없는 chunk도 유지해야 재시작 전후의 인접 쌍이 같습니다.
    by_chunk = {
        cid: []
        for cid in chunk_by_id
    }

    for event in state["events"]:
        by_chunk[event["chunk_id"]].append(event)

    done = set(state["completed_chunks"])

    client = OpenAI(
        timeout=timeout,
        max_retries=0,
    )

    try:
        for n, chunk in enumerate(chunks, 1):
            cid = chunk["chunk_id"]

            if cid in done:
                print(
                    f"[사건] {n}/{len(chunks)} chunk — 완료됨",
                    flush=True,
                )
                continue

            prompt = CHUNK_PROMPT.format(
                chunk_id=cid,
                token_start=chunk["token_start"],
                token_end=chunk["token_end"],
                event_types=EVENT_TYPES,
                relation_type_examples=RELATION_TYPE_EXAMPLES,
                chunk_text=chunk["text"],
            )

            print(
                f"[사건] {n}/{len(chunks)} chunk 요청 중...",
                flush=True,
            )

            data = call(
                client,
                model,
                prompt,
                retries,
                ("events", "relations"),
            )

            local, events, event_rejected = extract_events(
                data["events"],
                chunk,
                work,
                model,
            )

            edges, edge_rejected = extract_relations(
                data["relations"],
                local,
                {cid: chunk["text"]},
                model,
                "within_chunk",
            )

            state["events"].extend(events)
            state["edges"].extend(edges)
            state["rejected_records"].extend(
                event_rejected + edge_rejected
            )
            state["completed_chunks"].append(cid)

            by_chunk[cid] = events
            done.add(cid)

            save_checkpoint(
                checkpoint,
                state,
            )

            print(
                f"  완료 {n / len(chunks):.1%}: "
                f"사건 {len(events)}개 / "
                f"관계 {len(edges)}개 / "
                f"제외 {len(event_rejected) + len(edge_rejected)}개",
                flush=True,
            )

        ids = list(chunk_by_id)
        pairs = list(zip(ids, ids[1:]))

        pairs_done = {
            tuple(pair)
            for pair in state["completed_pairs"]
        }

        for n, (left_id, right_id) in enumerate(pairs, 1):
            pair = (left_id, right_id)

            if pair in pairs_done:
                continue

            left = by_chunk[left_id]
            right = by_chunk[right_id]

            print(
                f"[인접 관계] {n}/{len(pairs)} 쌍: "
                f"{left_id} ↔ {right_id}",
                flush=True,
            )

            if left and right:
                def brief(events: list[dict]) -> str:
                    keys = (
                        "event_id",
                        "label",
                        "summary",
                        "evidence",
                        "participant_names",
                    )

                    return json.dumps(
                        [
                            {
                                key: event[key]
                                for key in keys
                            }
                            for event in events
                        ],
                        ensure_ascii=False,
                    )

                texts = {
                    left_id: chunk_by_id[left_id]["text"],
                    right_id: chunk_by_id[right_id]["text"],
                }

                prompt = CROSS_PROMPT.format(
                    left=brief(left),
                    right=brief(right),
                    left_id=left_id,
                    right_id=right_id,
                    left_text=texts[left_id],
                    right_text=texts[right_id],
                    relation_type_examples=RELATION_TYPE_EXAMPLES,
                )

                data = call(
                    client,
                    model,
                    prompt,
                    retries,
                    ("relations",),
                )

                refs = {
                    event["event_id"]: event["event_id"]
                    for event in left + right
                }

                edges, rejected = extract_relations(
                    data["relations"],
                    refs,
                    texts,
                    model,
                    "adjacent_chunks",
                    {
                        event["event_id"]
                        for event in left
                    },
                    {
                        event["event_id"]
                        for event in right
                    },
                )

                state["edges"].extend(edges)
                state["rejected_records"].extend(rejected)

            state["completed_pairs"].append(
                list(pair)
            )

            save_checkpoint(
                checkpoint,
                state,
            )

        edges = list({
            edge["edge_id"]: edge
            for edge in state["edges"]
        }.values())

        write_jsonl(
            event_file,
            state["events"],
        )
        write_jsonl(
            edge_file,
            edges,
        )
        write_jsonl(
            rejection_file,
            state["rejected_records"],
        )

        checkpoint.unlink(
            missing_ok=True
        )

        print(
            f"\n완료: "
            f"사건 {len(state['events'])}개 / "
            f"관계 {len(edges)}개 / "
            f"제외 {len(state['rejected_records'])}개",
            flush=True,
        )

        print(
            f"{event_file}\n"
            f"{edge_file}\n"
            f"{rejection_file}",
            flush=True,
        )

    except KeyboardInterrupt:
        save_checkpoint(
            checkpoint,
            state,
        )

        print(
            "\n중단됨. 같은 명령으로 이어서 실행할 수 있습니다.\n"
            f"{checkpoint}",
            flush=True,
        )

    except Exception:
        save_checkpoint(
            checkpoint,
            state,
        )

        print(
            f"\n오류 발생 전 상태를 저장했습니다: {checkpoint}",
            flush=True,
        )

        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="소설의 사건과 사건 간 관계를 추출합니다."
    )

    parser.add_argument(
        "input_jsonl",
        type=Path,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/event"),
    )
    parser.add_argument(
        "--model",
        default=None,
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--reset",
        action="store_true",
    )

    args = parser.parse_args()

    main(
        args.input_jsonl,
        args.out_dir,
        args.model,
        args.timeout,
        args.retries,
        args.reset,
    )