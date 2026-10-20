"""Stage 4: Relation–Event 연결 확정 (결정적 후처리, LLM 호출 없음).

입력은 stage3 승인 연결이다. 다음 규칙만 적용해 9단계 그래프 입력을 만든다.

1. 양쪽 끝점 매칭 조건
   연결에 쓰인 Entity–Event edge를 원본 edge 파일에서 다시 조회해
   관계의 source observation과 target observation이 모두 그 Event에 연결되어 있을 때만 남긴다.
   (논문 2.3절: "관계를 이루는 두 인물이 Event에 포함되어 있는지 확인")
   stage3의 matched_sides 값은 신뢰하지 않고 edge에서 다시 계산한 값과 일치하는지 검사한다.

2. 연결 유형은 참고값으로만 보존
   stage3의 relation_event_link_type(STATE_EVIDENCE·FORMATION_CAUSE·CHANGE_CAUSE)은
   LLM 판정값이며 원문 대조에서 형성·변화 유형의 정밀도가 낮았다.
   값은 지우지 않고 그대로 두되 link_role을 모든 연결에 "relation_evidence"로 기록하고
   llm_link_type_is_reference=True로 표시한다. 하류 단계는 link_role을 사용한다.

3. relation_type이 없는 관계(null)의 연결은 제외하지 않는다
   관계 자체를 그래프에 넣을지는 9단계에서 한 번에 정한다. 여기서 연결만 빼면
   관계는 남고 근거만 사라지는 불일치가 생기므로 relation_type_missing=True로 표시만 한다.

무결성 검사(ID·작품·chunk·τ)가 하나라도 실패하면 결과를 쓰지 않고 중단한다.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path, PureWindowsPath
import sys

VERSION = "relation_event_finalize.v1"
LINK_ROLE = "relation_evidence"


def load_jsonl(path: Path) -> list[dict]:
    """JSONL 파일을 읽고 모든 행이 JSON 객체인지 확인한다."""
    if not path.is_file():
        raise FileNotFoundError(f"입력 파일 없음: {path}")
    rows = []
    with path.open(encoding="utf-8-sig") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{number} JSON 오류: {exc}") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{path}:{number} JSON 객체가 아님")
            rows.append(data)
    return rows


def index_by(rows: list[dict], field: str, path: Path) -> dict:
    """식별자 누락이나 중복이 있으면 중단한다."""
    result = {}
    for row in rows:
        key = row.get(field)
        if key is None or key in result:
            raise ValueError(f"{path}: {field} 누락 또는 중복: {key!r}")
        result[key] = row
    return result


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """임시 파일에 쓴 뒤 교체해 중간 실패 시 깨진 파일이 남지 않게 한다."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def resolve_report_path(raw: str, root: Path) -> Path:
    """stage2 report에 적힌 Windows 상대 경로를 현재 운영체제 경로로 바꾼다."""
    windows = PureWindowsPath(raw)
    if windows.is_absolute():
        return Path(raw)
    return root.joinpath(*windows.parts)


def verified_sides(link: dict, relation: dict, edges: dict) -> list[str]:
    """연결에 쓰인 edge를 원본에서 다시 조회해 실제로 매칭된 끝점을 계산한다."""
    sides = set()
    for edge_id in link["entity_event_edge_ids"]:
        edge = edges.get(edge_id)
        if edge is None:
            raise ValueError(f"{link['relation_event_edge_id']}: 원본에 없는 Entity–Event edge {edge_id}")
        if edge.get("target_event_id") != link["event_id"]:
            raise ValueError(f"{link['relation_event_edge_id']}: edge {edge_id}의 Event가 연결과 다름")
        if edge.get("chunk_id") != relation["chunk_id"]:
            raise ValueError(f"{link['relation_event_edge_id']}: edge {edge_id}의 chunk가 관계와 다름")
        owner = edge.get("source_observation_id")
        if owner == relation["source_observation_id"]:
            sides.add("source")
        if owner == relation["target_observation_id"]:
            sides.add("target")
        if owner not in (relation["source_observation_id"], relation["target_observation_id"]):
            raise ValueError(f"{link['relation_event_edge_id']}: edge {edge_id}의 인물이 관계 끝점이 아님")
    return [side for side in ("source", "target") if side in sides]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-id", required=True, help="예: 1_눈의 여왕")
    parser.add_argument("--project-root", type=Path, default=None,
                        help="프로젝트 루트 (생략 시 src/event_link 기준 자동 탐지)")
    parser.add_argument("--relation-dir", type=Path, default=None,
                        help="기본: data/event_relation/<작품>")
    parser.add_argument("--entity-event", type=Path, default=None,
                        help="기본: data/entity_event_new/<작품>.entity_event_edges.jsonl")
    parser.add_argument("--events", type=Path, default=None,
                        help="생략 시 stage2 report의 input_events 경로 사용")
    args = parser.parse_args()

    root = (args.project_root or Path(__file__).resolve().parents[2]).resolve()
    work = args.work_id

    def resolve(given: Path | None, default: str) -> Path:
        p = given if given is not None else Path(default)
        return p if p.is_absolute() else root / p

    rel_dir = resolve(args.relation_dir, f"data/event_relation/{work}")
    stage1_path = rel_dir / f"{work}.relation_event_stage1.jsonl"
    stage2_report_path = rel_dir / f"{work}.relation_event_stage2.report.json"
    stage3_path = rel_dir / f"{work}.relation_event_stage3.jsonl"
    stage3_report_path = rel_dir / f"{work}.relation_event_stage3.report.json"
    edge_path = resolve(args.entity_event, f"data/entity_event_new/{work}.entity_event_edges.jsonl")

    for p in (stage2_report_path, stage3_report_path):
        if not p.is_file():
            raise FileNotFoundError(f"입력 파일 없음: {p}")
    stage2_report = json.loads(stage2_report_path.read_text(encoding="utf-8-sig"))
    stage3_report = json.loads(stage3_report_path.read_text(encoding="utf-8-sig"))
    for name, report in (("stage2", stage2_report), ("stage3", stage3_report)):
        if report.get("work_id") != work:
            raise ValueError(f"{name} report의 work_id가 요청 작품과 다름")
    if stage3_report.get("status") == "incomplete":
        raise ValueError("stage3가 incomplete 상태다. stage3를 먼저 정상 완료해야 한다.")
    if args.events is not None:
        events_path = resolve(args.events, "")
    else:
        raw = stage2_report.get("input_events")
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("stage2 report에 input_events가 없다. --events로 지정해야 한다.")
        events_path = resolve_report_path(raw, root)

    relations = index_by(load_jsonl(stage1_path), "relation_observation_id", stage1_path)
    links = load_jsonl(stage3_path)
    edges = index_by(load_jsonl(edge_path), "edge_id", edge_path)
    events = index_by(load_jsonl(events_path), "event_id", events_path)

    if len(links) != stage3_report.get("final_approved_links", len(links)):
        raise ValueError("stage3 결과 건수가 stage3 report와 다름")

    kept, excluded = [], []
    seen = set()
    stats = Counter()
    for link in links:
        lid = link.get("relation_event_edge_id")
        key = (link.get("relation_observation_id"), link.get("event_id"))
        if not lid or key in seen:
            raise ValueError(f"연결 ID 누락 또는 (관계, Event) 중복: {lid} {key}")
        seen.add(key)
        if link.get("work_id") != work or link.get("final_relation_event_link") is not True:
            raise ValueError(f"{lid}: 작품 불일치 또는 승인되지 않은 연결")
        relation = relations.get(key[0])
        event = events.get(key[1])
        if relation is None or event is None:
            raise ValueError(f"{lid}: 원본 관계 또는 Event가 없음")
        if (link.get("source_observation_id") != relation["source_observation_id"]
                or link.get("target_observation_id") != relation["target_observation_id"]
                or link.get("chunk_id") != relation["chunk_id"]):
            raise ValueError(f"{lid}: 관계 끝점 또는 chunk가 원본 관계와 다름")
        if relation["chunk_id"] not in (event.get("occurrence_chunk_ids") or [event.get("chunk_id")]):
            raise ValueError(f"{lid}: Event가 관계 chunk에 등장하지 않음")

        # τ 재계산: 관계·Event·사용 edge 중 가장 늦게 공개되는 위치
        tau = max([int(relation["available_at_chunk"]), int(event["available_at_chunk"])]
                  + [int(edges[e]["available_at_chunk"]) for e in link["entity_event_edge_ids"] if e in edges])
        if tau != link.get("available_at_chunk") or tau != link.get("tau"):
            raise ValueError(f"{lid}: τ 불일치 (재계산 {tau}, 기록 {link.get('tau')})")
        if tau > relation["chunk_id"]:
            raise ValueError(f"{lid}: 연결 τ가 관계 chunk보다 늦음")

        sides = verified_sides(link, relation, edges)
        if set(sides) != set(link.get("matched_sides", [])):
            raise ValueError(f"{lid}: stage3 matched_sides와 edge 재계산 결과가 다름")

        record = {
            **link,
            "matched_sides_verified": sides,
            "link_role": LINK_ROLE,
            "llm_link_type_is_reference": True,
            "relation_type_missing": relation.get("relation_type") is None,
            "finalize_version": VERSION,
        }
        stats["input_links"] += 1
        if len(sides) == 2:
            kept.append(record)
            stats["kept"] += 1
            stats["kept_llm_type:" + str(link.get("relation_event_link_type"))] += 1
            if record["relation_type_missing"]:
                stats["kept_relation_type_missing"] += 1
        else:
            excluded.append({**record, "exclusion_reason": "one_endpoint_only:" + sides[0]})
            stats["excluded_one_endpoint_only"] += 1

    kept.sort(key=lambda x: (x["chunk_id"], x["relation_observation_id"], x["event_id"]))
    excluded.sort(key=lambda x: (x["chunk_id"], x["relation_observation_id"], x["event_id"]))

    linked_relations = {x["relation_observation_id"] for x in kept}
    out_main = rel_dir / f"{work}.relation_event_links.jsonl"
    out_excluded = rel_dir / f"{work}.relation_event_links.excluded.jsonl"
    out_summary = rel_dir / f"{work}.relation_event_links.summary.json"
    summary = {
        "version": VERSION, "work_id": work,
        "rule": "keep link only if both relation endpoints have an Entity–Event edge to the event",
        "link_role": LINK_ROLE,
        "llm_link_type_is_reference": True,
        "relation_type_null_policy": "kept_and_flagged (relation_type_missing)",
        "relations_total": len(relations),
        "relations_with_links": len(linked_relations),
        "relations_without_links": len(relations) - len(linked_relations),
        "counts": dict(stats),
        "inputs": {"stage1": str(stage1_path), "stage3": str(stage3_path),
                   "stage3_version": stage3_report.get("version"),
                   "entity_event_edges": str(edge_path), "events": str(events_path)},
        "outputs": {"links": str(out_main), "excluded": str(out_excluded)},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_jsonl(out_main, kept)
    write_jsonl(out_excluded, excluded)
    write_json(out_summary, summary)
    print(json.dumps({"work_id": work, "input": stats["input_links"], "kept": stats["kept"],
                      "excluded": stats["excluded_one_endpoint_only"],
                      "relations_with_links": f"{len(linked_relations)}/{len(relations)}",
                      "output": str(out_main)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        print(f"Relation–Event 연결 확정 실패: {exc}", file=sys.stderr)
        sys.exit(1)