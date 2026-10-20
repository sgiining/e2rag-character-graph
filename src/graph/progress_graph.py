"""기존 추출 결과를 읽어 진행도별 그래프를 구성한다. 원본 파일은 변경하지 않는다."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

WORKS = ["1_눈의 여왕", "2_엄지 공주", "3_인어 공주", "4_성냥팔이 소녀", "5_미운 아기 오리"]
# GitHub feature/ch에 저장된 실제 최종 입력 파일 경로다.
# Event와 Relation의 실행 시각별 하위 폴더는 find_run_file()에서 찾는다.
SPECS = {
    "chunk": "data/chunk/{w}.jsonl",
    "entity": "data/entity_observation/{w}.entities.jsonl",
    "entity_event": "data/entity_event_new/{w}.entity_event_edges.jsonl",
}
KEYS = {"entity": "observation_id", "event": "event_id", "relation": "relation_observation_id",
        "entity_event": "edge_id", "event_event": "edge_id", "stage3": "relation_event_edge_id"}


def read_jsonl(path: Path) -> list[dict]:
    """JSONL 파일의 모든 행을 읽고 손상된 행은 즉시 오류로 처리한다."""
    rows = []
    with path.open(encoding="utf-8-sig") as fp:
        for line_no, line in enumerate(fp, 1):
            if line.strip():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} JSON 오류: {exc}") from exc
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{line_no}: 객체가 아님")
                rows.append(row)
    return rows


def tau(row: dict) -> int:
    """실제 공개 가능 시점은 명시 필드와 발생 위치 중 가장 늦은 값을 사용한다."""
    vals = []
    for key in ("chunk_id", "available_at_chunk", "tau"):
        value = row.get(key)
        if value is not None and value != "":
            if isinstance(value, bool):
                raise ValueError(f"잘못된 {key}={value}")
            vals.append(int(value))
    if not vals or any(x < 1 for x in vals):
        raise ValueError(f"서사 위치 부재 또는 잘못됨: {row}")
    return max(vals)


def find_run_file(root: Path, subdir: str, work: str, filename: str,
                  selected_run: str | None) -> Path:
    """실행 시각 하위 폴더에서 파일을 찾는다. 다수의 버전은 임의로 고르지 않는다."""
    base = root / "data" / subdir / work
    if selected_run:
        # 실행 폴더명만 허용하여 다른 작품이나 외부 경로를 잘못 선택하지 않는다.
        if Path(selected_run).name != selected_run or selected_run in (".", ".."):
            raise ValueError(f"잘못된 실행 폴더명: {selected_run}")
        target = base / selected_run / filename
        if not target.is_file():
            raise FileNotFoundError(f"선택한 실행 결과 파일이 없습니다: {target}")
        return target

    candidates = sorted(base.glob(f"*/{filename}")) if base.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"입력 파일이 없습니다: {base}/*/{filename}")
    if len(candidates) > 1:
        option = "--event-run" if subdir == "events_new" else "--relation-run"
        versions = "\n".join(f"  - {p.parent.name}" for p in candidates)
        raise ValueError(
            f"{work}: {subdir}의 실행 결과가 {len(candidates)}개입니다.\n"
            f"임의의 결과를 사용하지 않습니다. {option} 옵션으로 선택하세요.\n{versions}"
        )
    return candidates[0]


def load_source(root: Path, work: str, stage_dir: Path,
                event_run: str | None = None,
                relation_run: str | None = None) -> tuple[dict, dict]:
    """feature/ch의 실제 경로에서 7종 데이터를 읽고 SHA-256을 기록한다."""
    paths = {k: root / pat.format(w=work) for k, pat in SPECS.items()}

    event_file = find_run_file(root, "events_new", work, "events.jsonl", event_run)
    # Event와 Event-Event 데이터가 같은 추출 실행 결과인지 보장한다.
    paths["event"] = event_file
    paths["event_event"] = event_file.parent / "event_edges.jsonl"
    paths["relation"] = find_run_file(
        root, "relation_observation", work, "relation_observations.jsonl", relation_run
    )

    stage_name = f"{work}.relation_event_stage3.jsonl"
    stage_candidates = [
        stage_dir / work / stage_name,
        stage_dir / "event_relation" / work / stage_name,
    ]
    paths["stage3"] = next((p for p in stage_candidates if p.is_file()), stage_candidates[0])

    missing = [str(p) for p in paths.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("입력 파일 누락:\n" + "\n".join(missing))

    records = {name: read_jsonl(path) for name, path in paths.items()}
    manifest = {
        name: {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "count": len(records[name]),
        }
        for name, path in paths.items()
    }
    return records, manifest


def build(work: str, records: dict, manifest: dict) -> tuple[dict, dict]:
    """원본 ID와 시점을 검증한다. 실패 시 어떤 그래프도 반환하지 않는다."""
    problems = []
    counts = Counter()

    def issue(kind: str, data_type: str, item: dict, message: str):
        problems.append({"category": kind, "dataset": data_type,
                         "id": str(item.get(KEYS.get(data_type, "id"), "")), "message": message})
        counts[kind] += 1

    chunks = {}
    for c in records["chunk"]:
        try:
            pos = int(c["chunk_id"])
            if pos < 1 or pos in chunks or not isinstance(c.get("text"), str):
                raise ValueError("중복 chunk 또는 text 없음")
            chunks[pos] = c["text"]
        except (KeyError, TypeError, ValueError) as e:
            issue("invalid_chunk", "chunk", c, str(e))
    if not chunks or sorted(chunks) != list(range(1, max(chunks) + 1)):
        issue("invalid_chunk", "chunk", {}, "chunk 1부터 마지막까지 연속적으로 존재해야 함")

    indexed = {}
    for kind in KEYS:
        arr = records[kind]
        key = KEYS[kind]
        indexed[kind] = {}
        for row in arr:
            rid = row.get(key)
            if not isinstance(rid, str) or not rid or rid in indexed[kind]:
                issue("invalid_or_duplicate_id", kind, row, f"{key} 누락/중복")
                continue
            indexed[kind][rid] = row
            if row.get("work_id", work) != work:
                issue("mixed_work", kind, row, "work_id 불일치")
            try:
                point = tau(row)
                if point not in chunks:
                    issue("invalid_position", kind, row, f"τ={point} 이 chunk 범위 밖")
                if row.get("available_at_chunk") is None:
                    issue("missing_position_field", kind, row, "available_at_chunk")
            except (ValueError, TypeError) as ex:
                issue("invalid_position", kind, row, str(ex))

    entities, events, relations = indexed["entity"], indexed["event"], indexed["relation"]
    for rid, r in relations.items():
        for k in ("source_observation_id", "target_observation_id"):
            if r.get(k) not in entities:
                issue("missing_entity_reference", "relation", r, f"{k}={r.get(k)}")
        for k, label in (("source_observation_id", "source_name"), ("target_observation_id", "target_name")):
            e = entities.get(r.get(k))
            if e and r.get(label) and str(r[label]) not in ([str(e.get("canonical_name"))] + list(e.get("aliases") or [])):
                issue("name_disagreement_review", "relation", r, f"{label}과 참조 Entity의 이름이 다름")
    for edge in indexed["entity_event"].values():
        if edge.get("source_observation_id") not in entities:
            issue("missing_entity_reference", "entity_event", edge, "Entity ID 없음")
        if edge.get("target_event_id") not in events:
            issue("missing_event_reference", "entity_event", edge, "Event ID 없음")
    for edge in indexed["event_event"].values():
        for k in ("source_event_id", "target_event_id"):
            if edge.get(k) not in events:
                issue("missing_event_reference", "event_event", edge, f"{k} 없음")

    approved = []
    for edge in indexed["stage3"].values():
        if edge.get("verification_status") != "approved" or edge.get("final_relation_event_link") is not True:
            continue
        approved.append(edge)
        rel = relations.get(edge.get("relation_observation_id"))
        evt = events.get(edge.get("event_id"))
        if rel is None:
            issue("stage3_relation_id_mismatch", "stage3", edge, "Relation 원본 ID 없음")
        if evt is None:
            issue("stage3_event_id_mismatch", "stage3", edge, "Event 원본 ID 없음")
        for k in ("source_observation_id", "target_observation_id"):
            if edge.get(k) not in entities:
                issue("stage3_entity_id_mismatch", "stage3", edge, f"{k} Entity 원본 ID 없음")
            if rel and rel.get(k) != edge.get(k):
                issue("stage3_relation_endpoint_mismatch", "stage3", edge, f"{k}와 Relation 불일치")
        if rel and evt:
            if tau(edge) < max(tau(rel), tau(evt)):
                issue("stage3_premature_release", "stage3", edge, "연결 공개 시점이 노드보다 빠름")
        if edge.get("relation_event_link_type") not in ("STATE_EVIDENCE", "FORMATION_CAUSE", "CHANGE_CAUSE"):
            issue("invalid_link_type", "stage3", edge, "알 수 없는 연결 유형")

    # Stage 3만 아닌 모든 원본 노드/엣지의 공개 시점을 검증한다.
    for kind in ("entity_event", "event_event"):
        for edge in indexed[kind].values():
            if kind == "entity_event":
                ends = [entities.get(edge.get("source_observation_id")), events.get(edge.get("target_event_id"))]
            else:
                ends = [events.get(edge.get("source_event_id")), events.get(edge.get("target_event_id"))]
            if all(ends) and tau(edge) < max(map(tau, ends)):
                issue("premature_edge_release", kind, edge, "엣지 공개 시점이 종단 노드보다 빠름")

    # 원문 근거는 해당 시점까지 공개된 chunk 내 실제 문자열로 존재해야 한다.
    for kind in ("entity", "event", "relation"):
        for row in indexed[kind].values():
            evidence = row.get("evidence")
            if not isinstance(evidence, str) or not evidence.strip():
                issue("missing_evidence", kind, row, "근거 누락")
            elif evidence not in chunks.get(int(row.get("chunk_id") or 0), ""):
                issue("evidence_not_exact_match", kind, row, "해당 chunk와 정확 일치하지 않음 (검토 필요)")

    fatal_prefixes = ("invalid_", "missing_entity_reference", "missing_event_reference", "mixed_work",
                      "stage3_", "premature_", "missing_position_field")
    fatal = [p for p in problems if p["category"].startswith(fatal_prefixes)]
    report = {"work_id": work, "status": "BLOCKED" if fatal else "OK",
              "fatal_count": len(fatal), "issues_by_category": dict(counts),
              "stage3_approved_count": len(approved), "input_manifest": manifest, "issues": problems,
              "source_record_counts": {name: len(rows) for name, rows in records.items()}}
    if fatal:
        return {}, report

    # Entity Observation과 Relation Observation은 동일 인물 여부를 추정하지 않는다.
    nodes = []
    for kind in ("entity", "event", "relation"):
        for rid, row in indexed[kind].items():
            nodes.append({"id": f"{kind}:{rid}", "type": kind, "source_id": rid,
                          "tau": tau(row), "data": row})
    edges = []
    def add(eid, etype, src, dst, row):
        edges.append({"id": f"{etype}:{eid}", "type": etype, "source": src, "target": dst,
                      "tau": tau(row), "data": row})
    for eid, row in indexed["entity_event"].items():
        add(eid, "entity_event", "entity:" + row["source_observation_id"], "event:" + row["target_event_id"], row)
    for eid, row in indexed["event_event"].items():
        add(eid, "event_event", "event:" + row["source_event_id"], "event:" + row["target_event_id"], row)
    for rid, row in relations.items():
        add(rid + ":source", "entity_relation_source", "entity:" + row["source_observation_id"], "relation:" + rid, row)
        add(rid + ":target", "entity_relation_target", "relation:" + rid, "entity:" + row["target_observation_id"], row)
    for row in approved:
        add(row["relation_event_edge_id"], "relation_event", "relation:" + row["relation_observation_id"], "event:" + row["event_id"], row)
    return {"work_id": work, "last_chunk": max(chunks), "nodes": nodes, "edges": edges,
            "source_fingerprints": manifest}, report


def filtered(graph: dict, p: int) -> dict:
    """시점 제한을 먼저 실행한 뒤 UI/QA에 전달할 데이터만 반환한다."""
    if p < 1 or p > graph["last_chunk"]:
        raise ValueError(f"유효한 진행도는 1~{graph['last_chunk']}")
    nodes = [node for node in graph["nodes"] if node["tau"] <= p]
    ids = {node["id"] for node in nodes}
    edges = [edge for edge in graph["edges"] if edge["tau"] <= p
             and edge["source"] in ids and edge["target"] in ids]
    result = {"work_id": graph["work_id"], "p": p, "last_chunk": graph["last_chunk"],
              "nodes": nodes, "edges": edges}
    assert all(n["tau"] <= p for n in nodes)
    assert all(e["tau"] <= p and e["source"] in ids and e["target"] in ids for e in edges)
    return result


def stats(graph: dict) -> dict:
    return {"p": graph["p"], "nodes": dict(Counter(n["type"] for n in graph["nodes"])),
            "edges": dict(Counter(e["type"] for e in graph["edges"]))}


def verify_progress(full: dict) -> dict:
    """미래 정보 유출 여부 및 진행도별 누적 단조성을 검사한다."""
    history = []
    prior_nodes, prior_edges = set(), set()
    for p in range(1, full["last_chunk"] + 1):
        sub = filtered(full, p)
        cur_nodes, cur_edges = {n["id"] for n in sub["nodes"]}, {e["id"] for e in sub["edges"]}
        if not prior_nodes <= cur_nodes or not prior_edges <= cur_edges:
            raise AssertionError(f"p={p} 누적 그래프 단조성 오류")
        prior_nodes, prior_edges = cur_nodes, cur_edges
        if p in {1, 3, 7, full["last_chunk"]}:
            history.append(stats(sub))
    return {"work_id": full["work_id"], "status": "PASS", "checked_chunks": full["last_chunk"],
            "checkpoints": history}


def write_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="E²RAG 독서 진행도 기반 그래프 생성 및 검증")
    parser.add_argument("--work-id", required=True, choices=WORKS)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument("--stage3-root", default="data/event_relation")
    parser.add_argument("--event-run", default=None,
                        help="events_new의 실행 폴더명. 버전이 여럿일 때 지정")
    parser.add_argument("--relation-run", default=None,
                        help="relation_observation의 실행 폴더명. 버전이 여럿일 때 지정")
    parser.add_argument("--output-root", default="data/progress_graph")
    parser.add_argument("--p", type=int, default=None)
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    out = (root / args.output_root).resolve()
    stage = (root / args.stage3_root).resolve()
    report_path = out / args.work_id / "validation_report.json"
    try:
        records, manifest = load_source(
            root, args.work_id, stage, args.event_run, args.relation_run
        )
        graph, report = build(args.work_id, records, manifest)
        write_json(report_path, report)
        if report["status"] != "OK":
            print(f"차단됨: {report['fatal_count']}건 치명적인 입력 문제. {report_path}")
            raise SystemExit(2)
        progress = verify_progress(graph)
        work_out = out / args.work_id
        write_json(work_out / "full_graph.json", graph)
        write_json(work_out / "progress_check.json", progress)
        if args.p is not None:
            visible = filtered(graph, args.p)
            write_json(work_out / f"graph_p{args.p}.json", visible)
            print(json.dumps(stats(visible), ensure_ascii=False))
        print(f"검증 통과: {work_out}")
    except (OSError, ValueError, KeyError) as err:
        write_json(report_path, {"work_id": args.work_id, "status": "BLOCKED", "fatal_error": str(err)})
        print(f"실패: {err}")
        raise SystemExit(2) from err


if __name__ == "__main__":
    main()
