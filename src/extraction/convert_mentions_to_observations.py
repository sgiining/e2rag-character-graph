"""Entity mention 결과(extract_entity_mentions_v5.py)를 chunk 단위 Entity observation 형식으로 변환한다. Python 3.12 권장.

목적
- extract_entity_mentions_v5.py의 entities.jsonl은 원문 위치(mention) 단위다.
- E²RAG의 Entity 노드와 이후 Event 연결·Relation 단계는 chunk 하나와 이름 하나당
  observation 한 줄(entity_observation.v2 형식)을 사용한다.
- 이 스크립트는 LLM을 호출하지 않고 v3 결과를 그 형식으로 바꾼다.
  같은 chunk 안의 같은 이름 mention들을 observation 하나로 묶고 각 mention의 원문 위치는 mentions에 보존한다.
  서로 다른 chunk의 observation은 합치지 않는다. 이름이 같아도 동일 인물로 확정하지 않는다.

입력
- --run-dir : data/entity_mentions_new/<작품>/<UTC 실행시각>  (summary.json, entities.jsonl)
- --cleaned : data/cleaned/<작품>.txt
- --chunks  : data/chunk/<작품>.jsonl

출력
- 기본값 data/entity_observation/<작품>.entities.jsonl  (기존 data/entity 결과를 덮어쓰지 않는다)
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_entity_mentions_v5 as v3  # noqa: E402  같은 폴더의 Entity 추출기 입력 검증 함수를 재사용한다.


def norm(value: str) -> str:
    """앞뒤 공백을 제거하고 연속된 공백을 하나로 정리한다. 기존 entity_extraction_v2와 같다."""
    return re.sub(r"\s+", " ", value).strip()


def slug(value: str) -> str:
    """이름을 참고용 키 문자열로 바꾼다. 기존 entity_extraction_v2와 같다."""
    value = re.sub(r"[^0-9A-Za-z가-힣]+", "_", norm(value)).strip("_")
    return value[:80] or "unnamed"


def stable_id(*parts: str, prefix: str) -> str:
    """같은 입력값에 대해 같은 해시 기반 ID를 만든다. 기존 entity_extraction_v2와 같다."""
    return f"{prefix}_{hashlib.sha256(chr(31).join(parts).encode('utf-8')).hexdigest()[:16]}"


def read_jsonl(path: Path) -> list[dict]:
    """UTF-8 JSONL을 읽는다."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def decide_type(mentions: list[dict]) -> tuple[str | None, dict]:
    """observation의 유형을 mention 유형의 단독 다수결로 정한다. 동률이거나 모두 미확정이면 None이다."""
    votes = Counter(m["entity_type"] for m in mentions if m["entity_type"] is not None)
    if not votes:
        return None, {}
    top = max(votes.values())
    leaders = [t for t, c in votes.items() if c == top]
    return (leaders[0] if len(leaders) == 1 else None), dict(votes)


def convert(run_dir: Path, cleaned: Path, chunks_path: Path, allow_incomplete: bool) -> tuple[list[dict], dict]:
    """v3 결과를 검증하고 chunk·이름 단위 observation 목록과 통계를 만든다."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("extractor_version") != "v3":
        raise ValueError("v3 추출 결과 폴더가 아닙니다.")
    if summary.get("status") != "completed_with_review" and not allow_incomplete:
        raise ValueError(f"추출 상태가 {summary.get('status')}입니다. 완료된 실행만 변환합니다.")
    text, chunks, _ = v3.load_inputs(cleaned, chunks_path)
    if v3.digest(text) != summary.get("document_sha256"):
        raise ValueError("cleaned 원문이 추출에 사용한 원문과 다릅니다.")
    if hashlib.sha256(chunks_path.read_bytes()).hexdigest() != summary.get("chunks_sha256"):
        raise ValueError("chunk JSONL이 추출에 사용한 파일과 다릅니다.")
    work_id = chunks_path.stem
    model = summary.get("model")
    entities = read_jsonl(run_dir / "entities.jsonl")

    groups = defaultdict(list)
    for entity in entities:
        a, b = entity["original_span"]
        if text[a:b] != entity["entity_name"]:
            raise ValueError(f"{entity['mention_id']}: 이름과 원문 좌표가 일치하지 않습니다.")
        homes = [c for c in chunks
                 if c["original_start"] <= a and b <= c["original_start"] + len(c["view"])]
        if not homes:
            raise ValueError(f"{entity['mention_id']}: 이 위치를 온전히 포함하는 chunk가 없습니다.")
        first_obs = entity["observations"][0]
        for chunk in homes:
            offset = chunk["chunk_view_start"] + (a - chunk["original_start"])
            groups[(chunk["chunk_id"], norm(entity["entity_name"]))].append({
                "mention_id": entity["mention_id"],
                "original_span": [a, b],
                "chunk_span": [offset, offset + (b - a)],
                "entity_type": entity["entity_type"],
                "mention_kind": entity["mention_kind"],
                "type_resolution": entity["type_resolution"],
                "needs_review": entity["needs_review"],
                "description": entity["description"],
                "evidence": first_obs["evidence"],
                "evidence_original_span": first_obs["evidence_original_span"],
            })

    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    seen_spans = defaultdict(set)
    records, stats = [], Counter()
    for (chunk_id, name) in sorted(groups, key=lambda k: (k[0], min(m["original_span"][0] for m in groups[k]))):
        mentions = sorted(groups[(chunk_id, name)], key=lambda m: m["original_span"])
        chunk = chunk_by_id[chunk_id]
        if name not in norm(chunk["text"]):
            raise ValueError(f"chunk {chunk_id}: 이름 {name!r}이 chunk 원문에 없습니다.")
        spans = {tuple(m["original_span"]) for m in mentions}
        is_overlap_repeat = spans <= seen_spans[name]
        seen_spans[name] |= spans
        entity_type, votes = decide_type(mentions)
        described = next((m for m in mentions if m["description"].strip()), mentions[0])
        record = {
            "observation_id": stable_id(work_id, str(chunk_id), name, prefix="eobs"),
            # 같은 키라고 동일 인물임을 보장하지 않는다.
            "name_key": f"{work_id}:name:{slug(name)}",
            "work_id": work_id,
            "canonical_name": name,
            "aliases": [],
            "entity_type": entity_type,
            "description": norm(described["description"]),
            "evidence": norm(described["evidence"]),
            "chunk_id": chunk_id,
            "available_at_chunk": chunk_id,
            "token_start": chunk["token_start"],
            "token_end": chunk["token_end"],
            "is_overlap_repeat": is_overlap_repeat,
            "needs_review": entity_type is None or any(m["needs_review"] for m in mentions),
            "type_votes": votes,
            "mention_count": len(mentions),
            "mentions": mentions,
            "extractor": {"model": model, "schema_version": "entity_observation.v2",
                          "converted_from": "entity_mention.v3",
                          "source_run": run_dir.name,
                          "document_sha256": summary["document_sha256"],
                          "chunks_sha256": summary["chunks_sha256"]},
        }
        records.append(record)
        stats["observations"] += 1
        stats["mentions_attached"] += len(mentions)
        stats["overlap_repeat"] += is_overlap_repeat
        stats["type_none"] += entity_type is None
        stats["needs_review"] += record["needs_review"]
    ids = [r["observation_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("observation_id가 중복되었습니다.")
    covered = {m["mention_id"] for r in records for m in r["mentions"]}
    if covered != {e["mention_id"] for e in entities}:
        raise RuntimeError("변환 과정에서 빠진 mention이 있습니다.")
    stats["source_mentions"] = len(entities)
    stats["chunks_with_entities"] = len({r["chunk_id"] for r in records})
    return records, dict(stats)


def main() -> None:
    """CLI 인자를 읽고 변환 결과를 새 파일에 저장한다."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--cleaned", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="incomplete 실행도 변환 (권장하지 않음)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = args.output or Path("data/entity_observation") / f"{args.chunks.stem}.entities.jsonl"
    if output.exists() and not args.overwrite:
        parser.error(f"{output}이 이미 있습니다. 덮어쓰려면 --overwrite를 사용하세요.")
    try:
        records, stats = convert(args.run_dir, args.cleaned, args.chunks, args.allow_incomplete)
    except (ValueError, RuntimeError) as exc:
        print(f"변환 실패: {exc}", file=sys.stderr)
        sys.exit(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                   encoding="utf-8")
    tmp.replace(output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"저장 완료: {output}")


if __name__ == "__main__":
    main()