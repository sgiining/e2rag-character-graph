"""작품별 Entity Observation을 이름 기준 그래프 통합 후보로 묶는다."""

import argparse
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path


WORK_IDS = (
    "1_눈의 여왕",
    "2_엄지 공주",
    "3_인어 공주",
    "4_성냥팔이 소녀",
    "5_미운 아기 오리",
)


def normalize_name(value):
    """유니코드 정규화 및 앞뒤 공백을 처리한다. 다른 이름을 임의로 합치지 않는다."""
    return unicodedata.normalize("NFC", value).strip()


def read_observations(path, requested_work_id):
    """원본 레코드와 ID 및 작품 일치 여부를 검증한다."""
    groups = defaultdict(list)
    seen = set()
    count = 0
    with path.open("r", encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number} JSON 오류: {error}") from error
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number} JSON 객체가 아닙니다.")
            work_id = record.get("work_id")
            observation_id = record.get("observation_id")
            raw_name = record.get("canonical_name")
            if work_id != requested_work_id:
                raise ValueError(f"{path}:{line_number} work_id 불일치: {work_id!r}")
            if not isinstance(observation_id, str) or not observation_id.strip():
                raise ValueError(f"{path}:{line_number} observation_id 누락")
            if observation_id in seen:
                raise ValueError(f"{path}:{line_number} observation_id 중복: {observation_id}")
            if not isinstance(raw_name, str) or not normalize_name(raw_name):
                raise ValueError(f"{path}:{line_number} canonical_name 누락")
            chunk = record.get("available_at_chunk", record.get("chunk_id"))
            if isinstance(chunk, bool) or not isinstance(chunk, int) or chunk < 0:
                raise ValueError(f"{path}:{line_number} 청크 위치 오류: {chunk!r}")
            seen.add(observation_id)
            groups[normalize_name(raw_name)].append(record)
            count += 1
    return groups, count


def build_groups(work_id, observations_by_name):
    """같은 작품·이름에 공통 그래프 ID를 부여하고 관찰 이력을 유지한다."""
    result = []
    for name in sorted(observations_by_name):
        observations = sorted(
            observations_by_name[name],
            key=lambda r: (r.get("available_at_chunk", r["chunk_id"]), r["observation_id"]),
        )
        key = f"{work_id}\x00{name}"
        graph_entity_id = "gentity_" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]
        types = sorted({str(r["entity_type"]) for r in observations if r.get("entity_type") is not None})
        members = [
            {
                "observation_id": r["observation_id"],
                "chunk_id": r["chunk_id"],
                "available_at_chunk": r.get("available_at_chunk", r["chunk_id"]),
                "entity_type": r.get("entity_type"),
                "needs_review": r.get("needs_review", False),
            }
            for r in observations
        ]
        result.append(
            {
                "graph_entity_id": graph_entity_id,
                "work_id": work_id,
                "canonical_name": name,
                "grouping_rule": "same_canonical_name_within_work",
                "identity_status": "name_based_unverified",
                "requires_identity_review": len(types) > 1 or any(m["needs_review"] for m in members),
                "observation_count": len(members),
                "entity_types": types,
                "observation_ids": [m["observation_id"] for m in members],
                "observations": members,
            }
        )
    return result


def write_jsonl(path, records):
    """UTF-8 JSONL 파일을 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def process(work_id, input_dir, output_dir):
    input_file = input_dir / f"{work_id}.entities.jsonl"
    if not input_file.is_file():
        raise FileNotFoundError(f"입력 파일이 없습니다: {input_file}")
    grouped, total = read_observations(input_file, work_id)
    if total == 0:
        raise ValueError(f"비어 있는 입력 파일: {input_file}")
    result = build_groups(work_id, grouped)
    full_path = output_dir / f"{work_id}.entity_identity_groups.jsonl"
    dup_path = output_dir / f"{work_id}.entity_identity_groups.duplicates.jsonl"
    write_jsonl(full_path, result)
    write_jsonl(dup_path, [r for r in result if r["observation_count"] > 1])
    print(f"[{work_id}] Observation={total}, 이름별 그룹={len(result)}, 중복 그룹={sum(r['observation_count'] > 1 for r in result)}")
    print(f"  전체: {full_path}\n  중복: {dup_path}")


def main():
    parser = argparse.ArgumentParser(description="작품별 동일 canonical_name의 Observation ID 통합 매핑 생성")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--work-id", help="예: 1_눈의 여왕")
    target.add_argument("--all", action="store_true", help="정해진 5개 작품 전체 처리")
    parser.add_argument("--input-dir", type=Path, default=Path("data/entity_observation"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/entity_identity"))
    args = parser.parse_args()
    for work_id in (WORK_IDS if args.all else (args.work_id,)):
        process(work_id, args.input_dir, args.output_dir)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"실행 실패: {error}") from error
