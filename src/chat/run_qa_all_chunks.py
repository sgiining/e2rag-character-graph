"""고전소설 5편의 고정 질문을 모든 독서 진행도 p에서 반복 실행하고 답변을 저장한다.

기존 src/qa/progress_qa.py의 검색·one-hop 확장·답변 생성 함수를 재사용한다.
원본 그래프, 인덱스, QA 평가셋 파일은 수정하지 않는다.

실행 예:
    python src/qa/run_qa_all_chunks.py --dry-run
    python src/qa/run_qa_all_chunks.py --work-id "1_눈의 여왕" --max-requests 5
    python src/qa/run_qa_all_chunks.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any

from progress_qa import (
    MODEL,
    SEED_STORES,
    ProgressQA,
    extract_cues,
    generate_answer,
    get_client,
    retrieve_context,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data" / "qa" / "progress_qa_batch"

# 질문 문구를 사용자가 제시한 그대로 보존한다. 소설 데이터의 추출 규칙에는 사용하지 않는다.
QUESTIONS: dict[str, list[tuple[str, str, str]]] = {
    "1_눈의 여왕": [
        ("general_1", "일반", "카이 눈에 뭐가 들어간 거야?"),
        ("general_2", "일반", "게르다는 왜 카이를 찾으러 떠났어?"),
        ("general_3", "일반", "카이는 누구 썰매를 따라간 거야?"),
        ("relation_1", "관계", "카이랑 게르다는 어떤 사이야?"),
        ("spoiler_1", "스포일러", "게르다는 결국 카이를 찾아?"),
    ],
    "2_엄지 공주": [
        ("general_1", "일반", "엄지공주는 어디서 태어났어?"),
        ("general_2", "일반", "두꺼비는 왜 엄지공주를 데려갔어?"),
        ("general_3", "일반", "엄지공주는 제비를 어떻게 도와줬어?"),
        ("relation_1", "관계", "엄지공주랑 제비는 어떤 사이야?"),
        ("spoiler_1", "스포일러", "엄지공주는 마지막에 누구랑 결혼해?"),
    ],
    "3_인어 공주": [
        ("general_1", "일반", "인어공주는 왕자를 어떻게 구했어?"),
        ("general_2", "일반", "인어공주는 어떻게 사람이 됐어?"),
        ("general_3", "일반", "인어공주는 왜 말을 못 해?"),
        ("relation_1", "관계", "왕자는 인어공주를 어떻게 생각해?"),
        ("spoiler_1", "스포일러", "인어공주는 결국 왕자랑 결혼해?"),
    ],
    "4_성냥팔이 소녀": [
        ("general_1", "일반", "소녀는 왜 맨발로 다녀?"),
        ("general_2", "일반", "소녀는 왜 집에 안 들어가?"),
        ("general_3", "일반", "소녀는 왜 성냥을 켰어?"),
        ("relation_1", "관계", "소녀한테 할머니는 어떤 사람이야?"),
        ("spoiler_1", "스포일러", "소녀는 마지막에 어떻게 돼?"),
    ],
    "5_미운 아기 오리": [
        ("general_1", "일반", "새끼 오리는 왜 따돌림을 당해?"),
        ("general_2", "일반", "새끼 오리는 왜 집을 나갔어?"),
        ("general_3", "일반", "얼음에 갇힌 새끼 오리를 누가 구해줬어?"),
        ("relation_1", "관계", "다른 오리들은 새끼 오리를 어떻게 대해?"),
        ("spoiler_1", "스포일러", "새끼 오리는 자기가 백조라는 걸 언제 알아?"),
    ],
}

CSV_COLUMNS = [
    "work_id", "qa_id", "category", "question", "p", "last_chunk", "status",
    "answer", "retrieved_chunk_ids", "cited_chunk_ids", "citation_status",
    "retrieved_nodes", "graph_links", "seed_entity", "seed_event", "seed_relation",
    "error",
]


def read_json(path: Path) -> dict:
    """JSON 객체를 읽는다."""
    with path.open(encoding="utf-8-sig") as fp:
        obj = json.load(fp)
    if not isinstance(obj, dict):
        raise ValueError(f"JSON 객체가 아닙니다: {path}")
    return obj


def read_jsonl(path: Path) -> list[dict]:
    """결과 JSONL을 읽고 필수 값의 유효성을 검사한다."""
    if not path.is_file():
        return []
    rows = []
    with path.open(encoding="utf-8-sig") as fp:
        for lineno, line in enumerate(fp, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}행 JSON이 손상되었습니다.") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{lineno}행은 JSON 객체여야 합니다.")
            rows.append(row)
    return rows


def atomic_json(path: Path, obj: dict) -> None:
    """임시 파일을 거쳐 JSON을 안전하게 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, row: dict) -> None:
    """한 건씩 즉시 저장해 중도 종료 시에도 완료 답변을 보존한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(row, ensure_ascii=False) + "\n")
        fp.flush()


def sha256(path: Path) -> str:
    """QA 엔진 및 질문 설정 버전 확인용 파일 해시를 구한다."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_config(engine: ProgressQA, work: str, top_k: int,
                passage_k: int, use_cues: bool) -> dict:
    """중단 후 재개 시 실험 설정과 입력이 바뀌지 않았는지 확인할 정보를 만든다."""
    qa_file = Path(__file__).with_name("progress_qa.py")
    return {
        "batch_version": "progress_qa_batch.v1",
        "work_id": work,
        "model": MODEL,
        "top_k": top_k,
        "passage_k": passage_k,
        "use_cues": use_cues,
        "last_chunk": len(engine.index.manifest["read_end_char"]),
        "index_build_id": engine.index.manifest["build_id"],
        "qa_engine_sha256": sha256(qa_file),
        "questions": [
            {"qa_id": qid, "category": category, "question": question}
            for qid, category, question in QUESTIONS[work]
        ],
    }


def check_config(folder: Path, config: dict) -> None:
    """다른 검색 조건의 답변을 한 실험 폴더에 섞지 않는다."""
    path = folder / "config.json"
    if path.exists():
        old = read_json(path)
        if old != config:
            raise ValueError(
                f"기존 실험 설정이 현재와 다릅니다: {path}\n"
                "다른 --output-dir를 지정해 새 실험을 시작하세요."
            )
    else:
        if (folder / "answers.jsonl").exists():
            raise ValueError(f"config.json 없이 답변 기록만 존재합니다: {folder}")
        atomic_json(path, config)


def existing_answers(folder: Path, work: str, questions: list[tuple[str, str, str]],
                     last_chunk: int) -> dict[tuple[int, str], dict]:
    """이미 처리한 (진행도, 질문 ID)를 읽어 중복 API 호출을 방지한다."""
    qids = {qid for qid, _, _ in questions}
    records: dict[tuple[int, str], dict] = {}
    for row in read_jsonl(folder / "answers.jsonl"):
        if row.get("work_id") != work or row.get("qa_id") not in qids:
            raise ValueError("저장된 답변에 다른 작품이나 질문 ID가 섞여 있습니다.")
        p = row.get("p")
        if type(p) is not int or not 1 <= p <= last_chunk:
            raise ValueError("저장된 답변의 진행도가 올바르지 않습니다.")
        key = (p, row["qa_id"])
        if key in records:
            raise ValueError(f"중복 저장된 답변입니다: {key}")
        records[key] = row
    return records


def latest_errors(folder: Path) -> dict[tuple[int, str], str]:
    """마지막 실패 사유를 가져온다. 성공한 건은 출력에서 성공을 우선한다."""
    result = {}
    for row in read_jsonl(folder / "errors.jsonl"):
        if isinstance(row.get("p"), int) and isinstance(row.get("qa_id"), str):
            result[(row["p"], row["qa_id"])] = str(row.get("error") or "실행 오류")
    return result


def make_rows(work: str, folder: Path, last: int) -> list[dict]:
    """답변한 경우와 미실행인 경우를 포함해 모든 진행도×질문 행을 만든다."""
    questions = QUESTIONS[work]
    saved = existing_answers(folder, work, questions, last)
    errors = latest_errors(folder)
    rows = []
    for p in range(1, last + 1):
        for qid, category, question in questions:
            key = (p, qid)
            result = saved.get(key)
            status = "완료" if result else "오류" if key in errors else "미실행"
            answer = str(result.get("answer", "")) if result else ""
            ids = result.get("retrieved_chunk_ids", []) if result else []
            cited = result.get("cited_chunk_ids", []) if result else []
            seeds = result.get("seed_counts", {}) if result else {}
            rows.append({
                "work_id": work, "qa_id": qid, "category": category, "question": question,
                "p": p, "last_chunk": last, "status": status, "answer": answer,
                "retrieved_chunk_ids": ",".join(map(str, ids)),
                "cited_chunk_ids": ",".join(map(str, cited)),
                "citation_status": result.get("citation_status", "") if result else "",
                "retrieved_nodes": result.get("retrieved_nodes", "") if result else "",
                "graph_links": result.get("graph_links", "") if result else "",
                "seed_entity": seeds.get("entity", ""), "seed_event": seeds.get("event", ""),
                "seed_relation": seeds.get("relation", ""),
                "error": "" if result else errors.get(key, ""),
            })
    return rows


def export_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    """엑셀에서 열 수 있는 UTF-8 BOM CSV를 만든다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def export_all(root_output: Path, selected: list[str]) -> None:
    """작품별 답변표, 통합 상세표, 진행도별 비교표, Markdown을 생성한다."""
    all_rows = []
    for work in selected:
        folder = root_output / work
        config_file = folder / "config.json"
        if not config_file.is_file():
            continue
        last = int(read_json(config_file)["last_chunk"])
        rows = make_rows(work, folder, last)
        export_csv(folder / "answers.csv", rows, CSV_COLUMNS)
        all_rows.extend(rows)

        lines = [f"# {work} — 진행도별 QA 전체 결과", "",
                 "각 진행도는 해당 Chunk까지 읽기를 완료한 상태를 뜻합니다.", ""]
        for qid, category, question in QUESTIONS[work]:
            lines.extend([f"## [{category}] {question}", ""])
            for row in rows:
                if row["qa_id"] != qid:
                    continue
                p, status = row["p"], row["status"]
                lines.append(f"### Chunk 1~{p} · {status}")
                if status == "완료":
                    lines.append(row["answer"])
                    lines.append(f"검색 원문 Chunk: {row['retrieved_chunk_ids'] or '-'}")
                    lines.append(f"인용 Chunk: {row['cited_chunk_ids'] or '-'}")
                elif status == "오류":
                    lines.append(f"실행 오류: {row['error']}")
                else:
                    lines.append("아직 실행하지 않았습니다.")
                lines.append("")
        (folder / "answers.md").write_text("\n".join(lines), encoding="utf-8")

    if not all_rows:
        return
    export_csv(root_output / "all_answers.csv", all_rows, CSV_COLUMNS)
    max_last = max(row["last_chunk"] for row in all_rows)
    matrix_columns = ["work_id", "qa_id", "category", "question", "last_chunk"] + [
        f"p_{i}" for i in range(1, max_last + 1)
    ]
    lookup = {(row["work_id"], row["qa_id"], row["p"]): row for row in all_rows}
    matrix = []
    included_works = [w for w in selected if (root_output / w / "config.json").is_file()]
    for work in included_works:
        last = int(read_json(root_output / work / "config.json")["last_chunk"])
        for qid, category, question in QUESTIONS[work]:
            entry = {"work_id": work, "qa_id": qid, "category": category,
                     "question": question, "last_chunk": last}
            for p in range(1, max_last + 1):
                row = lookup.get((work, qid, p))
                entry[f"p_{p}"] = (
                    "" if p > last else
                    row["answer"] if row and row["status"] == "완료" else
                    "[오류]" if row and row["status"] == "오류" else "[미실행]"
                )
            matrix.append(entry)
    export_csv(root_output / "comparison_matrix.csv", matrix, matrix_columns)
    counts = {status: sum(row["status"] == status for row in all_rows)
              for status in ("완료", "오류", "미실행")}
    atomic_json(root_output / "summary.json", {
        "selected_works": included_works,
        "expected_combinations": len(all_rows),
        "status_counts": counts,
        "note": "완료는 API 응답 저장을 의미하며 정답/무스포일러 판정이 아님",
    })
    print(f"\n결과 갱신: {root_output / 'all_answers.csv'}")
    print(f"질문별 진행도 비교: {root_output / 'comparison_matrix.csv'}")
    print(f"현황: 완료 {counts['완료']} / 오류 {counts['오류']} / 미실행 {counts['미실행']}")


def run(args: argparse.Namespace) -> int:
    """작품별 모든 진행도와 질문을 순회하고 완료 답변은 건너뛴다."""
    selected = [args.work_id] if args.work_id else list(QUESTIONS)
    out = args.output_dir.resolve()
    if args.dry_run:
        total = 0
        for work in selected:
            manifest = ROOT / "data" / "vector_index" / work / "manifest.json"
            if not manifest.is_file():
                print(f"[입력 누락] {manifest}")
                return 2
            last = len(read_json(manifest)["read_end_char"])
            n = last * len(QUESTIONS[work])
            total += n
            print(f"{work}: {last} chunks × {len(QUESTIONS[work])} questions = {n} QA")
        print(f"전체 {total}개 답변 생성 예정. --dry-run에서는 API를 호출하지 않습니다.")
        print(f"기본 설정: top_k={args.top_k}, passage_k={args.passage_k}, 질문 단서 추출={not args.no_cues}")
        return 0

    remaining = args.max_requests
    interrupted = False
    try:
        for work in selected:
            print(f"\n===== {work} =====", flush=True)
            engine = ProgressQA(work, ROOT)
            config = make_config(engine, work, args.top_k, args.passage_k, not args.no_cues)
            folder = out / work
            check_config(folder, config)
            last = config["last_chunk"]
            answers = existing_answers(folder, work, QUESTIONS[work], last)
            planned = last * len(QUESTIONS[work])
            print(f"전체 {planned}개 중 기존 완료 {len(answers)}개. 남은 {planned-len(answers)}개.", flush=True)
            if len(answers) == planned:
                continue

            cue_path = folder / "cue_cache.json"
            cues_cache = read_json(cue_path) if cue_path.is_file() else {}
            client = get_client(ROOT)
            for p in range(1, last + 1):
                for qid, category, question in QUESTIONS[work]:
                    key = (p, qid)
                    if key in answers:
                        continue
                    if remaining is not None and remaining <= 0:
                        print("--max-requests 상한에 도달해 안전하게 중지합니다.")
                        return 0
                    print(f"[{work}] p={p}/{last} | {category} | {question}", flush=True)
                    try:
                        if args.no_cues:
                            cues = {store: question for store in SEED_STORES}
                        else:
                            cached = cues_cache.get(qid)
                            if isinstance(cached, dict) and cached.get("question") == question:
                                cues = cached["cues"]
                            else:
                                cues = extract_cues(question, client)
                                cues_cache[qid] = {"question": question, "cues": cues}
                                atomic_json(cue_path, cues_cache)

                        context = retrieve_context(
                            engine.index, engine.full, p, question, cues,
                            top_k=args.top_k, passage_k=args.passage_k,
                            embedder=engine.embedder,
                        )
                        answer = generate_answer(question, context, client)
                        retrieved = [int(row["chunk_id"]) for row in context["passages"]]
                        cited = sorted({int(x) for x in re.findall(r"\[Chunk\s+(\d+)\]", answer)})
                        if any(x > p for x in retrieved + cited):
                            raise ValueError("진행도 p 이후 Chunk가 결과에 포함되었습니다.")
                        record = {
                            "work_id": work, "qa_id": qid, "category": category,
                            "question": question, "p": p, "last_chunk": last,
                            "answer": answer, "retrieved_chunk_ids": retrieved,
                            "cited_chunk_ids": cited,
                            "citation_status": "표시됨" if cited else "인용 없음 - 검토 필요",
                            "passages": context["passages"],
                            "seed_counts": context["seed_counts"],
                            "retrieved_nodes": len(context["nodes"]),
                            "graph_links": len(context["edges"]),
                        }
                        append_jsonl(folder / "answers.jsonl", record)
                        answers[key] = record
                        print(f"  저장 완료 | 근거 Chunk: {retrieved} | 인용: {cited}", flush=True)
                    except Exception as exc:
                        append_jsonl(folder / "errors.jsonl", {
                            "work_id": work, "qa_id": qid, "category": category,
                            "question": question, "p": p,
                            "error": f"{type(exc).__name__}: {exc}",
                        })
                        print(f"  실행 실패: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                        print("이후 다시 실행하면 저장된 성공 답변은 건너뛰고 실패 건부터 재시도합니다.", flush=True)
                        return 1
                    if remaining is not None:
                        remaining -= 1
    except KeyboardInterrupt:
        interrupted = True
        print("\n사용자가 중단했습니다. 이미 저장된 답변은 보존됩니다.", flush=True)
    except (ValueError, OSError, KeyError) as exc:
        print(f"실행 준비 실패: {exc}", file=sys.stderr)
        return 2
    finally:
        export_all(out, selected)
    return 130 if interrupted else 0


def main() -> None:
    """명령행 옵션을 파싱하고 종료 코드를 반환한다."""
    parser = argparse.ArgumentParser(description="5편 × 작품별 전 Chunk × 각 5문항의 E²RAG QA")
    parser.add_argument("--work-id", choices=list(QUESTIONS), help="한 작품만 실행. 생략하면 5편 모두 실행")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="실험 결과 출력 폴더 (기본 data/qa/progress_qa_batch)")
    parser.add_argument("--top-k", type=int, default=5, help="저장소별 초기 검색 개수")
    parser.add_argument("--passage-k", type=int, default=8, help="답변에 사용할 원문 Chunk 수")
    parser.add_argument("--no-cues", action="store_true", help="LLM 질문 단서 추출을 생략 (앱 기본 동작과 달라짐)")
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 작품별 질문 수만 표시")
    parser.add_argument("--max-requests", type=int, default=None,
                        help="이번 실행에서 새로 처리할 답변 개수 제한")
    args = parser.parse_args()
    if args.top_k < 1 or args.passage_k < 1:
        parser.error("--top-k와 --passage-k는 1 이상이어야 합니다.")
    if args.max_requests is not None and args.max_requests < 1:
        parser.error("--max-requests는 1 이상이어야 합니다.")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
