"""기존 Entity와 Event JSONL을 읽어 같은 chunk의 연결을 저장합니다.

원본 E²RAG examples/interactive_query.py의 construct_bipartite_graph 참고.
Entity 이름이 Event 설명 또는 사건명에 포함되면 연결합니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    """추출 결과를 UTF-8 JSONL로 읽습니다."""
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def connect_entity_event(entities: list[dict], events: list[dict]) -> list[dict]:
    """같은 작품과 chunk에서 이름이 포함된 Entity와 Event를 연결합니다."""
    events_by_chunk = defaultdict(list)
    for event in events:
        key = (event["work_id"], event["chunk_id"])
        events_by_chunk[key].append(event)

    edges = []
    seen = set()
    for entity in entities:
        name = entity["canonical_name"].strip().lower()
        if not name:
            continue

        key = (entity["work_id"], entity["chunk_id"])
        for event in events_by_chunk.get(key, []):
            # 원본의 description과 event_name은 현재 출력의 summary와 label입니다.
            description = event["summary"].lower()
            event_name = event["label"].lower()
            if name not in description and name not in event_name:
                continue

            pair = (entity["observation_id"], event["event_id"])
            if pair in seen:
                continue
            seen.add(pair)

            digest = hashlib.sha256("\x1f".join(pair).encode("utf-8"))
            edges.append({
                "edge_id": f"evlink_{digest.hexdigest()[:16]}",
                "work_id": entity["work_id"],
                "source_observation_id": pair[0],
                "target_event_id": pair[1],
                "edge_type": "entity_event_relation",
                "chunk_id": entity["chunk_id"],
                "available_at_chunk": max(
                    entity["available_at_chunk"], event["available_at_chunk"]
                ),
            })

    return edges


def main(entity_path: Path, event_path: Path, output_path: Path) -> None:
    """연결 결과만 별도의 JSONL 파일에 저장합니다."""
    edges = connect_entity_event(read_jsonl(entity_path), read_jsonl(event_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for edge in edges:
            stream.write(json.dumps(edge, ensure_ascii=False) + "\n")
    print(f"Entity ↔ Event 연결: {len(edges)}개")
    print(f"저장 완료: {output_path}")

    # 저장이 끝난 뒤 연결된 Entity와 Event를 터미널에서 확인합니다.
    entity_by_id = {
        row["observation_id"]: row for row in read_jsonl(entity_path)
    }
    event_by_id = {
        row["event_id"]: row for row in read_jsonl(event_path)
    }
    print("\n===== Entity ↔ Event 연결 상세 =====")
    for number, edge in enumerate(edges, 1):
        entity = entity_by_id[edge["source_observation_id"]]
        event = event_by_id[edge["target_event_id"]]
        print(f"\n[{number}] chunk {edge['chunk_id']}")
        print(f"  Entity: {entity['canonical_name']}")
        print(f"  사건명: {event['label']}")
        print(f"  사건 설명: {event['summary']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entity와 Event를 연결합니다.")
    parser.add_argument("--entities", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.entities, args.events, args.output)
