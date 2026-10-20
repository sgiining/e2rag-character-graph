"""GitHub data/qa/final의 확정 QA 134문항으로 기본 E²RAG 답변을 생성한다.

- qa_inputs.jsonl의 qa_id, work_id, p, question만 QA 엔진으로 전달한다.
- qa.jsonl은 문항 메타데이터와 진행도 검증에만 사용한다.
- 기준 정답, 미래 사건 정답 후보, gold_evidence는 생성 모델에 제공하지 않는다.
- 평가 파일과 기존 Entity/Event/Relation 파일은 수정하지 않는다.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "qa_eerag" / "e2rag_answers.jsonl"
QA_TYPES = ("answerable", "relation_change", "unanswerable_future")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """UTF-8 JSONL을 읽는다. 손상된 행이나 중복 키가 있으면 중단한다."""
    if not path.is_file():
        raise FileNotFoundError(f"필수 데이터 파일이 없습니다: {path}")
    rows: list[dict[str, Any]] = []

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"중복 JSON 키: {key}")
            obj[key] = value
        return obj

    with path.open("r", encoding="utf-8-sig") as fp:
        for line_no, line in enumerate(fp, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line, object_pairs_hook=reject_duplicate_keys)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: JSON 오류: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"{path}:{line_no}: JSON 객체가 아닙니다.")
            rows.append(item)
    return rows


def read_json(path: Path) -> dict:
    """요약 및 인덱스 manifest 파일을 읽는다."""
    if not path.is_file():
        raise FileNotFoundError(f"필수 데이터 파일이 없습니다: {path}")
    with path.open("r", encoding="utf-8-sig") as fp:
        obj = json.load(fp)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON 객체가 아닙니다: {path}")
    return obj


def sha256(path: Path) -> str:
    """입력 파일의 SHA-256 해시를 계산한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_dataset(root: Path) -> tuple[list[dict], dict[str, str]]:
    """GitHub 확정 QA 입력과 메타데이터를 QA ID 단위로 대조한다."""
    folder = root / "data" / "qa" / "final"
    inputs_path = folder / "qa_inputs.jsonl"
    full_path = folder / "qa.jsonl"
    summary_path = folder / "summary.json"
    inputs = read_jsonl(inputs_path)
    full = read_jsonl(full_path)
    summary = read_json(summary_path)

    if len(inputs) != 134 or len(full) != 134:
        raise ValueError(f"확정 데이터 134문항이 필요합니다: 입력={len(inputs)}, 원본={len(full)}")
    if summary.get("items") != 134:
        raise ValueError("summary.json의 전체 문항 수가 134가 아닙니다.")

    full_by_id = {}
    for item in full:
        qid = item.get("qa_id")
        if not isinstance(qid, str) or not qid or qid in full_by_id:
            raise ValueError(f"qa.jsonl에 QA ID가 없거나 중복되었습니다: {qid}")
        full_by_id[qid] = item

    used = set()
    paired = defaultdict(list)
    counts = Counter()
    clean = []
    for index, row in enumerate(inputs, 1):
        qid = row.get("qa_id")
        if not isinstance(qid, str) or qid not in full_by_id or qid in used:
            raise ValueError(f"qa_inputs.jsonl {index}행: QA ID 불일치/중복: {qid}")
        used.add(qid)
        original = full_by_id[qid]
        for field in ("qa_id", "work_id", "p", "question"):
            if row.get(field) != original.get(field):
                raise ValueError(f"{qid}: 두 JSONL 파일의 {field} 값이 다릅니다.")

        work = row["work_id"]
        p = row["p"]
        question = row["question"]
        qa_type = original.get("qa_type")
        end = original.get("read_end_char")
        pair = original.get("pair_group")
        if (not isinstance(work, str) or not work or type(p) is not int or p < 1
                or not isinstance(question, str) or not question.strip()
                or type(end) is not int or end < 1 or qa_type not in QA_TYPES
                or not isinstance(original.get("ground_truth"), str)
                or not original["ground_truth"].strip()):
            raise ValueError(f"{qid}: 작품/진행도/질문/유형/기준 정답 데이터 오류")
        if qa_type == "relation_change":
            if not isinstance(pair, str) or not pair:
                raise ValueError(f"{qid}: 관계 변화 질문의 pair_group이 없습니다.")
            paired[pair].append((work, p))
        elif pair is not None:
            raise ValueError(f"{qid}: 관계 변화가 아닌 질문에 pair_group이 있습니다.")
        counts[(work, qa_type)] += 1
        # 정답 등 평가 전용 필드는 여기에서 제외해 생성 함수에 전달하지 않는다.
        clean.append({
            "qa_id": qid,
            "work_id": work,
            "p": p,
            "question": question,
            "read_end_char": end,
            "qa_type": qa_type,
            "category": original.get("category"),
            "pair_group": pair,
        })

    if len(used) != len(full_by_id):
        raise ValueError("qa_inputs.jsonl과 qa.jsonl의 QA ID 집합이 다릅니다.")
    for pair, items in paired.items():
        if len(items) != 2 or items[0][0] != items[1][0] or items[0][1] >= items[1][1]:
            raise ValueError(f"{pair}: 관계 변화 전후 문항 구성이 올바르지 않습니다.")
    if len(paired) != 13:
        raise ValueError(f"관계 변화 질문 쌍은 13개여야 합니다: 현재 {len(paired)}개")

    expected = summary.get("by_work_type")
    if not isinstance(expected, dict):
        raise ValueError("summary.json에 by_work_type 항목이 없습니다.")
    actual = {f"{work}|{kind}": count for (work, kind), count in counts.items()}
    if actual != expected:
        raise ValueError("qa.jsonl 문항 분포와 summary.json의 by_work_type이 다릅니다.")

    hashes = {name: sha256(folder / name) for name in
              ("qa_inputs.jsonl", "qa.jsonl", "summary.json")}
    return clean, hashes


def validate_read_progress(root: Path, rows: list[dict]) -> dict[str, str]:
    """각 문항의 진행도가 해당 작품의 벡터 인덱스 공개 범위와 같은지 확인한다."""
    works = sorted({row["work_id"] for row in rows})
    builds = {}
    for work in works:
        manifest = read_json(root / "data" / "vector_index" / work / "manifest.json")
        ends = manifest.get("read_end_char")
        if manifest.get("work_id") != work or not isinstance(ends, list):
            raise ValueError(f"{work}: 벡터 인덱스 manifest 오류")
        for row in rows:
            if row["work_id"] != work:
                continue
            p = row["p"]
            if p > len(ends) or ends[p - 1] != row["read_end_char"]:
                raise ValueError(
                    f"{row['qa_id']}: p={p}의 read_end_char가 벡터 인덱스와 다릅니다. "
                    "진행도와 원문 범위가 일치하지 않아 실행하지 않습니다."
                )
        builds[work] = str(manifest.get("build_id", ""))
    return builds


def atomic_save(path: Path, rows: list[dict], previous: dict[str, dict]) -> None:
    """각 질문 결과를 원자적으로 저장하고 질문 원래 순서를 유지한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as fp:
            for source in rows:
                result = previous.get(source["qa_id"])
                if result is not None:
                    fp.write(json.dumps(result, ensure_ascii=False) + "\n")
        temp.replace(path)
    finally:
        if temp.exists():
            temp.unlink()


def main() -> None:
    """JSONL 검증 → 대상 선택 → 답변 생성 → JSONL 저장을 순서대로 실행한다."""
    parser = argparse.ArgumentParser(description="GitHub 확정 QA 134문항 기본 E²RAG 실행")
    parser.add_argument("--project-root", type=Path, default=ROOT, help="프로젝트 최상위 폴더")
    parser.add_argument("--out", type=Path, default=None, help="답변 저장 JSONL 경로")
    parser.add_argument("--work-id", help="작품 하나만 선택. 생략하면 5편 전체")
    parser.add_argument("--qa-id", help="특정 질문 하나만 선택")
    parser.add_argument("--limit", type=int, default=0, help="이번 실행에서 최대 질문 수(0=전체)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--passage-k", type=int, default=5)
    parser.add_argument("--no-cues", action="store_true", help="단서 추출 생략(본 평가에서는 사용 주의)")
    parser.add_argument("--dry-run", action="store_true", help="데이터 및 진행도 검증만 실행(API 비용 없음)")
    parser.add_argument("--retry-errors", action="store_true", help="오류로 저장된 문항을 다시 실행")
    args = parser.parse_args()
    if args.limit < 0 or args.top_k < 1 or args.passage_k < 1:
        raise ValueError("--limit은 0 이상이고 검색 개수는 1 이상이어야 합니다.")

    root = args.project_root.resolve()
    out_path = (args.out or (root / "data" / "qa_comparison" / "e2rag_answers.jsonl")).resolve()
    rows, hashes = load_dataset(root)
    builds = validate_read_progress(root, rows)
    summary = Counter(row["qa_type"] for row in rows)
    print(f"[데이터 검증] 134문항 | 일반 {summary['answerable']} | "
          f"관계 변화 {summary['relation_change']} | 미래 답변 불가 {summary['unanswerable_future']}")
    for work, count in Counter(row["work_id"] for row in rows).items():
        print(f"  {work}: {count}문항")

    selected = [row for row in rows if (not args.work_id or row["work_id"] == args.work_id)
                and (not args.qa_id or row["qa_id"] == args.qa_id)]
    if not selected:
        raise ValueError("선택한 작품 또는 QA ID에 해당하는 문항이 없습니다.")
    if args.limit:
        selected = selected[:args.limit]
    print(f"[실행 대상] {len(selected)}문항")
    if args.dry_run:
        print("[완료] JSONL 및 인덱스 진행도 검증 통과. API를 호출하지 않았습니다.")
        return

    # 실행 설정이 다른 결과를 섞지 않도록 데이터와 엔진 버전을 저장한다.
    engine_path = Path(__file__).with_name("basic_e2rag_qa.py")
    common_path = root / "src" / "chat" / "progress_qa.py"
    config = {
        "version": "basic_e2rag_final_qa.v1",
        "dataset_sha256": hashes,
        "qa_engine_sha256": sha256(engine_path),
        "shared_qa_sha256": sha256(common_path),
        "index_build_ids": builds,
        "top_k": args.top_k,
        "passage_k": args.passage_k,
        "use_cues": not args.no_cues,
        "generation_model": "gpt-4o-mini",
    }
    config_path = out_path.with_name(out_path.stem + ".config.json")
    if config_path.exists():
        if read_json(config_path) != config:
            raise ValueError(f"기존 실험 설정과 다릅니다. 별도 --out을 지정하세요: {config_path}")
    elif out_path.exists():
        raise ValueError(f"설정 파일 없이 기존 결과가 존재합니다. 백업하거나 별도 --out 사용: {out_path}")
    else:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    previous: dict[str, dict] = {}
    if out_path.exists():
        known = {row["qa_id"]: row for row in rows}
        for result in read_jsonl(out_path):
            qid = result.get("qa_id")
            if qid not in known or qid in previous:
                raise ValueError(f"기존 답변에 없는/중복 QA ID: {qid}")
            source = known[qid]
            if any(result.get(field) != source[field] for field in
                   ("work_id", "p", "question")):
                raise ValueError(f"기존 답변의 입력 내용이 달라졌습니다: {qid}")
            previous[qid] = result

    # dry-run과 구분하여 이 시점부터 실제 QA 엔진 및 API를 사용한다.
    from basic_e2rag_qa import BasicE2RAGQA

    engines: dict[str, BasicE2RAGQA] = {}
    success = skip = error = 0
    for i, row in enumerate(selected, 1):
        qid = row["qa_id"]
        old = previous.get(qid)
        if old and (old.get("generation_status") == "ok" or not args.retry_errors):
            skip += 1
            print(f"[{i}/{len(selected)}] {qid}: 기존 결과 유지")
            continue

        record = {
            **row,
            "qa_system": "basic_e2rag_entity_event_only",
            "generation_model": "gpt-4o-mini",
            "model_answer": "",
            "retrieved_chunks": [],
            "seed_counts": {},
            "retrieved_nodes": 0,
            "graph_links": 0,
            "generation_status": "error",
            "error": "",
        }
        try:
            work = row["work_id"]
            if work not in engines:
                engines[work] = BasicE2RAGQA(work, root=root)
            result = engines[work].ask(
                p=row["p"], question=row["question"],
                top_k=args.top_k, passage_k=args.passage_k,
                use_cues=not args.no_cues,
            )
            if result.get("work_id") != work or result.get("p") != row["p"]:
                raise ValueError("QA 엔진의 작품 또는 진행도가 질문과 일치하지 않습니다.")
            passages = result.get("passages", [])
            if any(int(x["chunk_id"]) > row["p"] or int(x["tau"]) > row["p"]
                   for x in passages):
                raise ValueError("진행도를 초과한 원문이 반환되었습니다.")
            record.update({
                "model_answer": str(result["answer"]),
                "retrieved_chunks": passages,
                "seed_counts": result.get("seed_counts", {}),
                "retrieved_nodes": result.get("retrieved_nodes", 0),
                "graph_links": result.get("graph_links", 0),
                "generation_status": "ok",
            })
            success += 1
            print(f"[{i}/{len(selected)}] {qid}: 답변 생성 완료")
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            error += 1
            print(f"[{i}/{len(selected)}] {qid}: 실패 - {record['error']}", file=sys.stderr)

        previous[qid] = record
        atomic_save(out_path, rows, previous)

    print(f"[완료] 신규 성공 {success} | 기존 유지 {skip} | 오류 {error}")
    print(f"[저장] {len(previous)}/134문항: {out_path}")
    if error:
        print("오류 문항 재실행: 명령어 끝에 --retry-errors 추가")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError, OSError) as exc:
        print(f"실행 실패: {exc}", file=sys.stderr)
        sys.exit(1)
