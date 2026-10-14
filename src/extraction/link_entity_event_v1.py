"""Entity observation과 Event를 연결하는 bipartite edge(B)를 만든다. Python 3.12 권장. LLM 호출 없음.

E²RAG 원논문·공식 구현을 따르는 연결 (method="name_in_description")
- 논문 식 (1)과 Algorithm 2: 같은 chunk이고 norm(Event 설명)에 norm(Entity 이름)이 포함되면 연결한다.
- Event가 등장한 모든 chunk(occurrence_chunk_ids)를 chunk 조건에 사용한다(공식 구현과 같음).
- method="name_in_label": 공식 examples/interactive_query.py의 construct_bipartite_graph처럼 Event 이름(label)에
  포함될 때의 연결이다. 논문 식 (1)에는 없으므로 paper_method와 분리하고 official_impl_method로 표시한다.

본 연구가 추가한 연결 (원논문에 없음)
- method="name_in_description_bounded": 논문 방식의 포함 위치 중 하나 이상이 단어 경계에 있을 때 추가한다.
  앞 글자가 한글·영문·숫자가 아니고 뒤 글자가 한글·영문·숫자가 아니거나 조사로 시작해야 한다.
  '신'⊂'자신', '성'⊂'완성'처럼 다른 단어의 일부에 걸린 연결을 구분하기 위한 것이다.
  논문 방식 결과(name_in_description)는 그대로 남기고 이 판정은 별도 method와 필드로만 기록한다.
- method="mention_in_evidence": Entity mention의 원문 좌표가 Event 근거 구절의 원문 좌표 안에 있으면 연결한다.
- method="participant": Event 참여자 표기의 원문 좌표가 Entity mention 좌표와 같으면 연결한다.
  두 방법은 문자열 포함이 아니라 같은 원문 위치를 근거로 하므로 설명문 표현에 덜 의존한다.

입력
- --entities : data/entity_observation/<작품>.entities.jsonl  (convert_mentions_to_observations.py 결과)
- --event-run: data/events_new/<작품>/<UTC 실행시각>          (extract_events_v1.py 결과)

출력
- 기본값 data/entity_event_new/<작품>.entity_event_edges.jsonl 과 같은 이름의 .summary.json
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys


def norm(value: str) -> str:
    """연속 공백을 하나로 정리하고 소문자로 바꾼다. 원논문 Norm에 해당한다."""
    return re.sub(r"\s+", " ", value).strip().lower()


def read_jsonl(path: Path) -> list[dict]:
    """UTF-8 JSONL을 읽는다."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


# 단어 경계 판정에 쓰는 조사·어미 목록이다. 긴 것부터 검사한다. 특정 작품과 무관한 한국어 문법 목록이다.
JOSA = sorted(["에게서", "으로서", "으로써", "이라고", "에게", "에서", "한테", "께서", "까지", "부터",
               "처럼", "보다", "조차", "마저", "이나", "이랑", "으로", "하고", "이라", "이다", "이었",
               "였", "이", "가", "은", "는", "을", "를", "의", "에", "와", "과", "도", "로", "께",
               "들", "만", "랑"], key=len, reverse=True)
WORD_CHAR = re.compile(r"[0-9A-Za-z가-힣]")


def bounded_match(name: str, text: str) -> bool:
    """name이 text 안에서 단어 경계에 맞게 한 번 이상 나오는지 검사한다. 둘 다 norm된 값이어야 한다."""
    start = text.find(name)
    while start >= 0:
        end = start + len(name)
        before_ok = start == 0 or not WORD_CHAR.match(text[start - 1])
        rest = text[end:]
        after_ok = not rest or not WORD_CHAR.match(rest[0]) or any(rest.startswith(j) for j in JOSA)
        if before_ok and after_ok:
            return True
        start = text.find(name, start + 1)
    return False


def inside(span: list[int], outer: list[int]) -> bool:
    """span이 outer 구간 안에 완전히 들어 있는지 검사한다."""
    return outer[0] <= span[0] and span[1] <= outer[1]


def link(entities: list[dict], events: list[dict]) -> tuple[list[dict], dict]:
    """같은 chunk의 Entity observation과 Event를 세 가지 방법으로 연결하고 근거를 함께 저장한다."""
    events_by_chunk = defaultdict(list)
    for event in events:
        for chunk_id in event["occurrence_chunk_ids"]:
            events_by_chunk[chunk_id].append(event)
    edges = {}
    stats = Counter({k: 0 for k in ("edges", "only_paper_method", "only_paper_method_unbounded",
                                    "only_extension_methods", "label_only_official",
                                    "recommended", "not_recommended")})
    for entity in entities:
        name = norm(entity["canonical_name"])
        if not name:
            continue
        for event in events_by_chunk.get(entity["chunk_id"], []):
            methods, matched = set(), set()
            description, label = norm(event["description"]), norm(event["label"])
            # 원논문 식 (1): Event 설명(desc)에 이름이 포함될 때만 논문 방식으로 기록한다.
            if name in description:
                methods.add("name_in_description")
                if bounded_match(name, description):
                    methods.add("name_in_description_bounded")
            # 공식 구현(construct_bipartite_graph)은 Event 이름(label) 포함도 연결한다. 논문 식과 구분해 기록한다.
            if name in label:
                methods.add("name_in_label")
                if bounded_match(name, label):
                    methods.add("name_in_label_bounded")
            for mention in entity.get("mentions", []):
                if inside(mention["original_span"], event["original_span"]):
                    methods.add("mention_in_evidence")
                    matched.add(mention["mention_id"])
                for participant in event.get("participants", []):
                    if mention["original_span"] in participant["evidence_spans"]:
                        methods.add("participant")
                        matched.add(mention["mention_id"])
            if not methods:
                continue
            grounded = bool(methods & {"mention_in_evidence", "participant"})
            # 권장 판단은 원문 좌표 근거 또는 설명문(desc)의 단어 경계 포함만 사용한다.
            # label은 LLM이 붙인 짧은 이름이라 설명과 어긋난 경우가 있어 단독 근거로 권장하지 않는다.
            bounded = "name_in_description_bounded" in methods
            official = bool(methods & {"name_in_description", "name_in_label"})
            pair = (entity["observation_id"], event["event_id"])
            digest = hashlib.sha256("\x1f".join(pair).encode("utf-8")).hexdigest()[:16]
            edges[pair] = {
                "edge_id": f"evlink_{digest}", "work_id": entity["work_id"],
                "source_observation_id": entity["observation_id"],
                "target_event_id": event["event_id"], "edge_type": "entity_event_relation",
                "entity_name": entity["canonical_name"], "entity_type": entity["entity_type"],
                "event_label": event["label"], "chunk_id": entity["chunk_id"],
                "available_at_chunk": max(entity["available_at_chunk"], event["available_at_chunk"]),
                "methods": sorted(methods), "paper_method": "name_in_description" in methods,
                "official_impl_method": official,
                "matched_mention_ids": sorted(matched),
                # 원문 좌표 근거가 있거나 단어 경계에 맞는 이름 포함일 때 그래프 구성에 권장한다.
                "name_match_bounded": bounded if "name_in_description" in methods else None,
                "recommended": grounded or bounded,
            }
            stats["edges"] += 1
            for method in methods:
                stats[f"method:{method}"] += 1
            if not grounded:
                stats["only_paper_method"] += 1
                if not bounded:
                    stats["only_paper_method_unbounded"] += 1
            if "name_in_description" not in methods:
                stats["only_extension_methods"] += 1
            if "name_in_label" in methods and "name_in_description" not in methods:
                stats["label_only_official"] += 1
            stats["recommended" if grounded or bounded else "not_recommended"] += 1
    linked_events = {e["target_event_id"] for e in edges.values()}
    linked_entities = {e["source_observation_id"] for e in edges.values()}
    stats["events"] = len(events)
    stats["events_without_entity"] = len(events) - len(linked_events)
    stats["entity_observations"] = len(entities)
    stats["entity_observations_without_event"] = len(entities) - len(linked_entities)
    recommended_events = {e["target_event_id"] for e in edges.values() if e["recommended"]}
    stats["events_without_recommended_entity"] = len(events) - len(recommended_events)
    return sorted(edges.values(), key=lambda e: (e["chunk_id"], e["edge_id"])), dict(stats)


def main() -> None:
    """입력 일치를 검사하고 연결 결과와 요약을 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--event-run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    try:
        entities = read_jsonl(args.entities)
        summary = json.loads((args.event_run / "summary.json").read_text(encoding="utf-8"))
        if summary.get("status") != "completed_with_review":
            raise ValueError(f"Event 추출 상태가 {summary.get('status')}입니다.")
        events = read_jsonl(args.event_run / "events.jsonl")
        if not entities or not events:
            raise ValueError("Entity 또는 Event 결과가 비었습니다.")
        work_ids = {e["work_id"] for e in entities} | {e["work_id"] for e in events}
        if len(work_ids) != 1:
            raise ValueError(f"작품 ID가 일치하지 않습니다: {sorted(work_ids)}")
        hashes = {(e["extractor"].get("document_sha256"), e["extractor"].get("chunks_sha256"))
                  for e in entities} | {(summary["document_sha256"], summary["chunks_sha256"])}
        if len(hashes) != 1:
            raise ValueError("Entity와 Event가 서로 다른 원문 또는 chunk 파일에서 추출되었습니다.")
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"연결 실패: {exc}", file=sys.stderr)
        sys.exit(1)
    work_id = work_ids.pop()
    output = args.output or Path("data/entity_event_new") / f"{work_id}.entity_event_edges.jsonl"
    if output.exists() and not args.overwrite:
        parser.error(f"{output}이 이미 있습니다. 덮어쓰려면 --overwrite를 사용하세요.")
    edges, stats = link(entities, events)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in edges), encoding="utf-8")
    tmp.replace(output)
    report = {"work_id": work_id, "entities": str(args.entities), "event_run": str(args.event_run),
              **stats}
    output.with_suffix(".summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"저장 완료: {output}")


if __name__ == "__main__":
    main()