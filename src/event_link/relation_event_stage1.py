
import argparse
import json
from pathlib import Path
from collections import defaultdict, Counter


def read_jsonl(path):
    records = []

    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}:{line_number} JSON 오류: {e}"
                ) from e

            if not isinstance(item, dict):
                raise ValueError(
                    f"{path}:{line_number} JSON 객체가 아님"
                )

            records.append(item)

    return records


def required_id(record, key):
    value = record.get(key)

    if not isinstance(value, str) or not value:
        raise ValueError(f"필수 ID 누락: {key}")

    return value


def collect_links(relations, entity_event_edges):

    # 1. Entity Observation ID 기준 색인
    entity_index = defaultdict(list)

    for edge in entity_event_edges:
        entity_id = required_id(
            edge, "source_observation_id"
        )
        required_id(edge, "target_event_id")

        entity_index[entity_id].append(edge)

    results = []
    stats = Counter()

    # 2. Relation Observation을 기준으로 순회
    for relation in relations:

        relation_id = required_id(
            relation, "relation_observation_id"
        )
        source_id = required_id(
            relation, "source_observation_id"
        )
        target_id = required_id(
            relation, "target_observation_id"
        )

        # 3. 동일 Entity ID에 연결된 Edge 전부 조회
        source_edges = list(
            entity_index.get(source_id, [])
        )
        target_edges = list(
            entity_index.get(target_id, [])
        )

        # 4. Event ID별로 묶기
        # 원본 Edge 목록은 따로 전부 보존
        event_map = {}

        for side, edges in [
            ("source", source_edges),
            ("target", target_edges)
        ]:
            for edge in edges:

                event_id = edge["target_event_id"]
                entity_id = edge["source_observation_id"]

                if event_id not in event_map:
                    event_map[event_id] = {
                        "event_id": event_id,
                        "matched_entity_ids": [],
                        "matched_sides": [],
                        "entity_event_edge_ids": []
                    }

                candidate = event_map[event_id]

                if entity_id not in candidate["matched_entity_ids"]:
                    candidate["matched_entity_ids"].append(
                        entity_id
                    )

                if side not in candidate["matched_sides"]:
                    candidate["matched_sides"].append(side)

                edge_id = edge.get("edge_id")
                if (
                    edge_id
                    and edge_id not in candidate["entity_event_edge_ids"]
                ):
                    candidate["entity_event_edge_ids"].append(
                        edge_id
                    )

        event_candidates = list(event_map.values())

        # 5. Relation Observation별 결과 구성
        result = {
            **relation,

            "source_entity_event_edges": source_edges,
            "target_entity_event_edges": target_edges,

            "event_candidates": event_candidates,
            "linked_event_ids": list(event_map.keys()),

            "source_edge_count": len(source_edges),
            "target_edge_count": len(target_edges),
            "unique_event_count": len(event_candidates),

            "link_method": "exact_entity_observation_id",
            "link_stage": 1,
            "semantic_validation_performed": False
        }

        results.append(result)

        stats["relations_total"] += 1
        stats["source_edge_references"] += len(source_edges)
        stats["target_edge_references"] += len(target_edges)
        stats["unique_relation_event_pairs"] += len(event_candidates)

        if source_edges or target_edges:
            stats["relations_with_events"] += 1
        else:
            stats["relations_without_events"] += 1

    return results, dict(stats)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():

    parser = argparse.ArgumentParser(
        description="Stage 1: Relation 중심 Entity ID 연결 수집"
    )

    parser.add_argument(
        "--relations",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--entity-event",
        type=Path,
        required=True
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True
    )

    args = parser.parse_args()

    if args.output.resolve() in {
        args.relations.resolve(),
        args.entity_event.resolve()
    }:
        parser.error("출력 경로가 입력 경로와 같을 수 없습니다.")

    relations = read_jsonl(args.relations)
    entity_event_edges = read_jsonl(args.entity_event)

    results, stats = collect_links(
        relations,
        entity_event_edges
    )

    write_jsonl(args.output, results)

    stats["entity_event_edges_input"] = len(
        entity_event_edges
    )

    report_path = args.output.with_suffix(".report.json")
    report_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(
        stats,
        ensure_ascii=False,
        indent=2
    ))
    print(f"결과: {args.output}")
    print(f"통계: {report_path}")


if __name__ == "__main__":
    main()
