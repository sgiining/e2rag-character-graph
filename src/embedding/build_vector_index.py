"""기존 추출 결과를 변경하지 않고 4종 NumPy 인덱스를 구축·검증한다."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import io
import json
from pathlib import Path
import re
import sys

import numpy as np

if __package__:
    from .vector_index import (DIM, MODEL, STORES, VERSION, Embedder, VectorIndex,
                               digest, read_jsonl, write_json)
else:
    from vector_index import (DIM, MODEL, STORES, VERSION, Embedder, VectorIndex,
                              digest, read_jsonl, write_json)


def require(condition, message):
    """최적화 실행에서도 생략되지 않는 검증이다."""
    if not condition:
        raise ValueError(message)


def norm(value):
    """좌표는 바꾸지 않고 텍스트 비교 때만 공백을 정규화한다."""
    return re.sub(r"\s+", " ", value).strip()


def file_info(root, path):
    """팀원이 다른 루트에서 쓸 수 있도록 상대 경로와 파일 해시를 기록한다."""
    return {"path": path.relative_to(root).as_posix(), "sha256": digest(path.read_bytes())}


def one_file(root, pattern, explicit):
    """실행 결과가 여러 개면 임의로 최신 결과를 고르지 않고 지정하도록 한다."""
    if explicit:
        result = (root / explicit).resolve()
    else:
        matches = sorted(root.glob(pattern))
        require(len(matches) == 1, f"입력 결과를 명시하세요: {pattern} ({len(matches)}개)")
        result = matches[0].resolve()
    require(result.is_file() and result.is_relative_to(root), f"프로젝트 내부 입력 파일 필요: {result}")
    return result


def prepare(root, work, events=None, relations=None):
    """기존 좌표 함수로 검증하고 원본 ID·τ와 1:1인 안전한 검색 레코드를 만든다."""
    require(work and Path(work).name == work and not any(c in work for c in "/\\:"), "work_id 오류")
    inputs = {
        "cleaned": root / "data/cleaned" / f"{work}.txt",
        "chunk": root / "data/chunk" / f"{work}.jsonl",
        "entity": root / "data/entity_observation" / f"{work}.entities.jsonl",
        "event": one_file(root, f"data/events_new/{work}/*/events.jsonl", events),
        "relation": one_file(root, f"data/relation_observation/{work}/*/relation_observations.jsonl", relations),
    }
    source = root / "src/extraction/extract_entity_mentions_v5.py"
    spec = importlib.util.spec_from_file_location("stage8_existing_coordinates", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text, chunks, notices = module.load_inputs(inputs["cleaned"], inputs["chunk"])
    read_end, last = [], 0
    for c in chunks:
        last = max(last, c["original_start"] + len(c["view"]))
        read_end.append(last)
    source_rows = {s: read_jsonl(inputs[s]) for s in STORES}
    rows, stats = {s: [] for s in STORES}, Counter()
    text_hash, chunk_hash = digest(text), digest(inputs["chunk"].read_bytes())
    entity_hash = digest(inputs["entity"].read_bytes())

    def span(value):
        require(isinstance(value, list) and len(value) == 2
                and all(type(x) is int for x in value)
                and 0 <= value[0] < value[1] <= len(text), f"원문 좌표 오류: {value}")
        return value

    def evidence(value, quote):
        a, b = span(value)
        require(norm(text[a:b]) == norm(quote), f"원문 근거 불일치: {[a, b]}")
        return a, b

    for store in STORES:
        ids = set()
        for raw in source_rows[store]:
            id_field = {"entity": "observation_id", "event": "event_id",
                        "relation": "relation_observation_id", "chunk": "chunk_id"}[store]
            node_id = raw[id_field]
            require((type(node_id) is int if store == "chunk" else isinstance(node_id, str) and bool(node_id))
                    and node_id not in ids, f"ID 오류/중복: {store}:{node_id}")
            ids.add(node_id)
            tau = raw["chunk_id"] if store == "chunk" else raw["available_at_chunk"]
            require(type(tau) is int and 1 <= tau <= len(chunks), "τ 범위 오류")
            limit, desc, chosen, partial, alternative = read_end[tau - 1], "", None, False, False
            desc_basis = None
            if store != "chunk":
                require(raw["work_id"] == work, f"다른 작품 혼입: {node_id}")
                ex = raw["extractor"]
                require(ex["document_sha256"] == text_hash and ex["chunks_sha256"] == chunk_hash,
                        f"추출 입력 해시 불일치: {node_id}")
                if store == "relation":
                    require(ex["entities_sha256"] == entity_hash, "Relation의 Entity 버전 불일치")
                require(type(raw["chunk_id"]) is int and 1 <= raw["chunk_id"] <= len(chunks), "chunk_id 오류")
            extra = {}
            if store == "entity":
                mentions = raw["mentions"]
                require(bool(mentions), "Entity mentions 누락")
                for m in mentions:
                    a, b = span(m["original_span"])
                    require(norm(text[a:b]) == norm(raw["canonical_name"]), "mention 이름/좌표 불일치")
                    c = chunks[raw["chunk_id"] - 1]
                    require(c["original_start"] <= a < b <= c["original_start"] + len(c["view"]),
                            "mention이 소속 chunk 안에 없습니다.")
                    require(m["chunk_span"] == [c["chunk_view_start"] + a - c["original_start"],
                                               c["chunk_view_start"] + b - c["original_start"]], "mention chunk 좌표 오류")
                    evidence(m["evidence_original_span"], m["evidence"])
                    stats["all_mention_evidence_crossings"] += int(m["evidence_original_span"][1] > limit)
                representatives = [m for m in mentions if norm(m["description"]) == norm(raw["description"])
                                   and norm(m["evidence"]) == norm(raw["evidence"])]
                require(bool(representatives), "대표 설명과 근거의 출처 mention을 찾지 못했습니다.")
                selected = representatives[0]
                crossed = selected["evidence_original_span"][1] > limit
                stats["representative_crossings"] += int(crossed)
                if crossed:
                    safe = [m for m in mentions if m["evidence_original_span"][1] <= limit
                            and m["evidence_original_span"][0] <= m["original_span"][0]
                            and m["original_span"][1] <= m["evidence_original_span"][1]]
                    if safe:
                        selected, alternative = safe[0], True
                a, b = selected["evidence_original_span"]
                partial = b > limit
                chosen = [a, min(b, limit)]
                require(chosen[0] < chosen[1], "공개된 Entity 근거가 없습니다.")
                if not partial:
                    desc, desc_basis = selected["description"], list(selected["evidence_original_span"])
                stats["alternative_evidence"] += int(alternative)
                stats["description_removed"] += int(partial)
                extra = {"canonical_name": raw["canonical_name"], "name_key": raw["name_key"],
                         "entity_type": raw["entity_type"], "selected_mention_id": selected["mention_id"],
                         "is_overlap_repeat": raw["is_overlap_repeat"]}
                title = [raw["canonical_name"], raw["entity_type"]]
            elif store in ("event", "relation"):
                a, b = evidence(raw["original_span"], raw["evidence"])
                c = chunks[raw["chunk_id"] - 1]
                require(c["original_start"] <= a < b <= c["original_start"] + len(c["view"]),
                        f"{store} 근거가 소속 chunk 밖에 있습니다.")
                require(raw["chunk_span"] == [c["chunk_view_start"] + a - c["original_start"],
                                             c["chunk_view_start"] + b - c["original_start"]], "chunk 좌표 오류")
                require(b <= limit, f"{store} 근거가 τ 이후입니다. 원본 결과를 확인하세요.")
                chosen, desc, desc_basis = [a, b], raw["description"], [a, b]
                if store == "event":
                    extra = {"label": raw["label"], "event_type": raw["event_type"],
                             "narrative_level": raw["narrative_level"]}
                    title = [raw["label"], raw["event_type"], raw["narrative_level"]]
                else:
                    extra = {k: raw[k] for k in ("source_observation_id", "target_observation_id",
                             "source_name", "target_name", "source_name_key", "target_name_key",
                             "relation_type", "direction", "polarity", "modality", "narrative_level")}
                    extra["relation_type_missing"] = raw["relation_type"] is None
                    title = [raw[k] for k in ("source_name", "target_name", "relation_type_label",
                                              "direction", "polarity", "modality", "narrative_level")]
            else:
                c = chunks[raw["chunk_id"] - 1]
                chosen = [c["original_start"], c["original_start"] + len(c["view"])]
                extra = {"read_end_char": limit}
                title = []
            quote = text[chosen[0]:chosen[1]]
            embed_text = "\n".join(str(x) for x in [*title, desc, quote] if x is not None and str(x).strip())
            row = {"node_id": node_id, "node_type": store, "work_id": work,
                   "tau": tau, "available_at_chunk": tau, "chunk_id": raw["chunk_id"],
                   "evidence": quote, "evidence_original_span": chosen, "evidence_partial": partial,
                   "alternative_evidence": alternative, "description": desc,
                   "description_used": bool(desc.strip()), "description_basis_span": desc_basis,
                   "description_basis_complete": desc_basis is not None and desc_basis[1] <= limit,
                   "description_check": "source_evidence_boundary_only_not_semantic_entailment",
                   "read_end_char_tau": limit, "embed_text": embed_text, "text_sha256": digest(embed_text),
                   "needs_review": raw.get("needs_review", False), "flags": raw.get("flags", []), **extra}
            rows[store].append(row)
            stats[f"{store}_partial"] += int(partial)
    entities = {r["node_id"]: r for r in rows["entity"]}
    for r in rows["relation"]:
        require(all(k in entities and entities[k]["tau"] <= r["tau"]
                    for k in (r["source_observation_id"], r["target_observation_id"])), "Relation 끝점 ID/τ 오류")
    info = {k: file_info(root, path) for k, path in inputs.items()}
    info["coordinate_code"] = file_info(root, source)
    return rows, {"inputs": info, "read_end_char": read_end, "counts": {s: len(rows[s]) for s in STORES},
                  "boundary_stats": dict(stats), "utf8_boundary_notices": notices}, text


def validate(root, work, expected, base, text):
    """실제 원본과 재대조하고 모든 p에서 모든 문서 벡터의 검색 범위를 검사한다."""
    index = VectorIndex(root, work)
    for key in ("inputs", "read_end_char", "counts", "boundary_stats", "utf8_boundary_notices"):
        require(index.manifest[key] == base[key], f"입력/설정 버전 불일치: {key}")
    tested, tie_ok = {}, {}
    for s in STORES:
        require(index.meta[s] == expected[s], f"원본에서 재생성한 metadata 불일치: {s}")
        comparisons = 0
        for p in range(1, len(base["read_end_char"]) + 1):
            allowed = {r["node_id"] for r in expected[s] if r["tau"] <= p}
            for i, r in enumerate(expected[s]):
                result = index.search(s, p, max(1, len(expected[s])), query_vector=index.vectors[s][i])
                require({x["node_id"] for x in result} == allowed, "진행도별 후보 집합 불일치")
                require(all(x["tau"] <= p and x["work_id"] == work for x in result), "미래/다른 작품 노출")
                comparisons += 1
        for i, r in enumerate(expected[s]):
            a, b = r["evidence_original_span"]
            require(r["evidence"] == text[a:b] and b <= index.read_end(r["tau"]), "근거 좌표 불일치")
            result = index.search(s, r["tau"], max(1, len(expected[s])), query_vector=index.vectors[s][i])
            own = next(x["score"] for x in result if x["node_id"] == r["node_id"])
            require(abs(own - 1) < 2e-5 and own >= result[0]["score"] - 2e-5, "자기 검색 최대 유사도 실패")
        tested[s], tie_ok[s] = comparisons, len(expected[s])
    query = next(index.vectors[s][0] for s in STORES if len(expected[s]))
    result = index.search_e2rag(1, query_vector=query)
    require(set(result) == {"entity", "event", "relation"}, "E²RAG에 Chunk 저장소 혼입")
    return {"status": "passed", "work_id": work, "build_id": index.manifest["build_id"],
            "counts": base["counts"], "boundary_stats": base["boundary_stats"],
            "all_progress_searches": tested, "self_similarity_tie_accepted": tie_ok,
            "checks": ["input_hashes", "original_ids_and_tau", "metadata_source_rebuild", "coordinates",
                       "description_source_boundary", "finite_unit_vectors", "all_progress_prefilter",
                       "work_isolation", "e2rag_chunk_separation", "file_integrity"],
            "limitations": ["설명의 의미적 근거 충분성은 재판정하지 않음. 기존 추출 결과를 사용함.",
                            "구조 검증 통과는 자연어 검색 품질 또는 최종 QA의 스포일러 차단 평가가 아님."]}


def build(root, work, events=None, relations=None, check_only=False, prepare_only=False,
          batch_size=64, embedder=None):
    """prepare-only는 API 없이 입력을 검사한다. 완성 빌드는 임시 폴더에서 검증 후 교체한다."""
    root = Path(root).resolve()
    rows, base, text = prepare(root, work, events, relations)
    folder = root / "data/vector_index" / work
    folder.parent.mkdir(parents=True, exist_ok=True)
    if prepare_only:
        report = {"status": "inputs_passed_embeddings_not_run", "work_id": work, **base}
        write_json(folder.parent / "_checks" / f"{work}.prepare.json", report)
        return report
    if check_only:
        report = validate(root, work, rows, base, text)
        write_json(folder / "verify_report.json", report)
        return report
    manifest = {"version": VERSION, "model": MODEL, "dimensions": DIM, "work_id": work,
                "stores": list(STORES), "e2rag_stores": list(STORES[:3]),
                "chunk_role": "vanilla_rag_baseline_only", "coordinate_unit": "cleaned_python_char_half_open",
                "progress_unit": "completed_chunk", **base}
    manifest["code_sha256"] = {p.name: digest(p.read_bytes()) for p in
        (Path(__file__), Path(__file__).with_name("vector_index.py"))}
    manifest["build_id"] = digest(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    existing = folder / "manifest.json"
    if existing.exists() and json.loads(existing.read_text(encoding="utf-8"))["build_id"] == manifest["build_id"]:
        report = validate(root, work, rows, base, text)
        write_json(folder / "verify_report.json", report)
        return {**report, "api_calls_this_run": 0, "reused_complete_index": True}
    worker = embedder or Embedder(root, batch_size)
    import shutil
    temp = folder.with_name(folder.name + ".building")
    require(not temp.exists(), f"임시 폴더가 있습니다. 동시 실행 여부 확인 후 제거하세요: {temp}")
    temp.mkdir()
    calls_before, hits_before = worker.calls, worker.hits
    try:
        outputs = {}
        for s in STORES:
            matrix = worker.encode([r["embed_text"] for r in rows[s]])
            buffer = io.BytesIO()
            np.save(buffer, matrix, allow_pickle=False)
            (temp / f"{s}.npy").write_bytes(buffer.getvalue())
            content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows[s])
            (temp / f"{s}.meta.jsonl").write_text(content, encoding="utf-8")
            for name in (f"{s}.npy", f"{s}.meta.jsonl"):
                outputs[name] = digest((temp / name).read_bytes())
        manifest["outputs"] = outputs
        write_json(temp / "manifest.json", manifest)
        # 완성된 파일만 공개한다. 검증 실패 시 기존 인덱스를 복원한다.
        backup = folder.with_name(folder.name + ".backup")
        require(not backup.exists(), f"이전 backup 확인 필요: {backup}")
        if folder.exists():
            folder.rename(backup)
        try:
            temp.rename(folder)
            report = validate(root, work, rows, base, text)
            write_json(folder / "verify_report.json", report)
        except BaseException:
            if folder.exists():
                shutil.rmtree(folder)
            if backup.exists():
                backup.rename(folder)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return {**report, "api_calls_this_run": worker.calls - calls_before,
                "cache_hits_this_run": worker.hits - hits_before, "reused_complete_index": False}
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def main():
    """Windows의 py 명령으로 실행한다. 입력 불일치는 API 호출 전에 중단한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--work-id", required=True)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--relations", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    try:
        result = build(args.project_root, args.work_id, args.events, args.relations,
                       args.check_only, args.prepare_only, args.batch_size)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(f"8단계 실패: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
