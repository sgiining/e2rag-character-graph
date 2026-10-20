"""QA 후보의 사람 검수·일치도·최종 평가셋 동결 (qa_finalize.v1). Python 3.12 권장. LLM 호출 없음.

세 가지 명령으로 나뉜다.

1) export : 검증을 통과한 후보(status=verifier_accepted)를 모아 사람 검수표를 만든다.
   - review_main.csv   : 주 검수자용. 모든 문항이 들어 있고 decision(keep/fix/drop)을 채운다.
   - review_second.csv : 두 번째 검수자용 독립 검수표. 작품×유형별로 --second-ratio만큼 표집한다.
                         관계 변화 쌍은 두 문항을 함께 표집한다.
   - items.jsonl       : 내보낸 후보 원본(build가 읽는다)
   - sentences/<작품>.csv : 문장 번호표(id, tau, text). 검수자가 근거 문장 번호를 찾을 때 쓴다.
   검수표에는 시스템 답변이 없다. 검수 중 시스템 답변을 보지 않는다.

2) agreement : 주 검수와 두 번째 검수의 decision 일치도(Cohen's κ)를 계산한다.
   불일치 해결 전에 실행한다. 결과는 agreement.json, disagreements.csv.

3) build : 주 검수표의 decision을 반영해 최종 평가셋을 만든다.
   - keep : 그대로 채택
   - fix  : fixed_* 칸의 값으로 바꾼다. 빈 칸은 원래 값을 쓴다. evidence_rechecked가 y여야 한다
            (정답을 고쳤다면 수정본의 근거도 다시 확인했다는 표시).
            근거 문장 번호를 바꾸면 좌표와 τ를 코드로 다시 계산하고 유형별 코드 검사를 다시 한다.
   - drop : 제외
   관계 변화 쌍은 한 문항이 제외되면 다른 문항도 제외한다(pair_incomplete).
   같은 작품에서 근거 구간이 겹치고 질문이 비슷한 문항과 관계 쌍을 한 묶음으로 보고 같은 split에 넣는다.
   batch=dev 문항이 하나라도 있는 묶음은 dev, 나머지는 test다.
   결과는 data/qa/final/ 에 저장하고 기존 결과가 있으면 --overwrite 없이는 덮어쓰지 않는다(평가셋 동결).

출력 (build)
- data/qa/final/qa.jsonl        : 최종 평가셋(정답·근거·미래 근거 포함, 평가 전용)
- data/qa/final/qa_inputs.jsonl : 시스템 입력용(qa_id, work_id, p, question)만 담은 파일
- data/qa/final/summary.json    : 집계·검수 결과·split 묶음·파일 해시
"""
from __future__ import annotations

import argparse
import bisect
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys

FINALIZER_VERSION = "qa_finalize.v1"
DECISIONS = ("keep", "fix", "drop")
YES = {"y", "yes", "예", "o", "true", "1"}
CSV_FIELDS = ["qa_id", "work_id", "batch", "qa_type", "pair_group", "p", "read_end_char", "category",
              "question", "ground_truth", "gold_sentence_ids", "gold_evidence_text",
              "future_sentence_ids", "future_evidence_text", "candidate_future_answer",
              "relation_source", "relation_target", "verifier_reason",
              "decision", "fixed_category", "fixed_question", "fixed_ground_truth",
              "fixed_gold_sentence_ids", "fixed_future_sentence_ids", "evidence_rechecked",
              "reviewer", "note"]
FINAL_FIELDS = ["qa_id", "work_id", "split", "p", "read_end_char", "qa_type", "category", "question",
                "ground_truth", "gold_evidence", "future_evidence", "pair_group",
                "document_sha256", "chunks_sha256", "generator", "verification"]
CATEGORIES = [
    "Character and Behavioral Consistency", "Causal Consistency",
    "Setting, Environment and Atmosphere", "Symbolism, Imagery and Motifs",
    "Thematic, Philosophical and Moral", "Narrative and Plot Structure",
    "Social, Cultural and Political", "Emotional and Psychological",
]


def read_jsonl(path: Path) -> list[dict]:
    """UTF-8 JSONL을 읽는다."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """JSONL을 임시 파일에 쓴 뒤 교체한다."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


def write_json(path: Path, value: object) -> None:
    """JSON을 임시 파일에 쓴 뒤 교체한다."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    """Excel에서 한글이 깨지지 않도록 UTF-8 BOM으로 CSV를 쓴다."""
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    """검수표 CSV를 읽는다."""
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(stream)]


def sha256_file(path: Path) -> str:
    """파일 SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- 원문 좌표 (후보 실행의 sentences.jsonl 기준)

class RunText:
    """후보 실행 폴더의 문장 번호표와 진행도로 근거 좌표와 τ를 다시 계산한다."""

    def __init__(self, run_dir: Path):
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        self.work_id = summary["work_id"]
        self.document_sha256 = summary["document_sha256"]
        self.read_end = [row["read_end_char"] for row in summary["progress_points"]]
        self.n = len(self.read_end)
        self.sentences = read_jsonl(run_dir / "sentences.jsonl")

    def tau(self, end: int) -> int | None:
        """τ([a,b)) = min{k | b ≤ read_end(k)}."""
        k = bisect.bisect_left(self.read_end, end)
        return k + 1 if k < self.n else None

    def visible_count(self, p: int) -> int:
        """p에서 일부라도 공개된 문장 수."""
        limit = self.read_end[p - 1]
        return sum(1 for s in self.sentences if s["original_span"][0] < limit)

    def evidence(self, ids: list[int], limit: int | None) -> list[dict]:
        """build_qa_candidates.Document.evidence와 같은 규칙으로 근거 항목을 만든다."""
        items, run = [], []
        for index in sorted(set(ids)):
            if run and index != run[-1] + 1:
                items.append(self._item(run, limit))
                run = []
            run.append(index)
        if run:
            items.append(self._item(run, limit))
        return items

    def _item(self, run: list[int], limit: int | None) -> dict:
        first, last = self.sentences[run[0] - 1], self.sentences[run[-1] - 1]
        a = first["original_span"][0]
        la, lb = last["original_span"]
        partial = False
        if limit is not None and lb > limit:
            text = last["text"][:limit - la].rstrip()
            lb, partial = la + len(text), True
        # text는 fill_text가 cleaned 원문 좌표로 채운다(문장 사이 공백까지 원문 그대로).
        return {"sentence_ids": [f"S{i}" for i in run], "original_span": [a, lb],
                "text": None, "tau": self.tau(lb), "partial_sentence": partial}


def load_text(cleaned_dir: Path, work_id: str, document_sha256: str) -> str:
    """cleaned 원문을 기존 추출 코드와 같은 방식(utf-8-sig)으로 읽고 해시를 확인한다."""
    path = cleaned_dir / f"{work_id}.txt"
    text = path.read_text(encoding="utf-8-sig")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != document_sha256:
        raise ValueError(f"{path}: 후보 생성 때와 원문 해시가 다릅니다.")
    return text


def parse_ids(value: str, upper: int) -> list[int] | None:
    """'S3, S4' 또는 'S3;S4' 형태를 문장 번호 목록으로 바꾼다. 범위 밖이면 None."""
    parts = [x for x in re.split(r"[,;\s]+", value) if x]
    result = []
    for part in parts:
        match = re.fullmatch(r"[Ss]?(\d+)", part)
        if not match or not 1 <= int(match.group(1)) <= upper:
            return None
        result.append(int(match.group(1)))
    return result


def ids_text(items: list[dict]) -> str:
    """근거 항목의 문장 번호를 쉼표로 잇는다."""
    return ", ".join(i for item in items for i in item["sentence_ids"])


def evidence_text(items: list[dict]) -> str:
    """근거 항목의 원문을 구분선으로 잇는다."""
    return "\n---\n".join(item["text"] for item in items)


# ---------------------------------------------------------------- export

def collect(runs: list[str]) -> tuple[list[dict], dict]:
    """여러 실행 폴더에서 verifier_accepted 후보를 모은다. 같은 qa_id는 먼저 나온 것만 쓴다."""
    items, seen, run_of_work = [], set(), defaultdict(set)
    hashes = {}
    for raw in runs:
        run_dir = Path(raw).resolve()
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        work = summary["work_id"]
        if hashes.setdefault(work, (summary["document_sha256"], summary["chunks_sha256"])) != (
                summary["document_sha256"], summary["chunks_sha256"]):
            raise ValueError(f"{work}: 실행 폴더마다 원문 또는 chunk 해시가 다릅니다.")
        run_of_work[work].add(str(run_dir))
        rows = [r for r in read_jsonl(run_dir / "candidates.jsonl") if r["status"] == "verifier_accepted"]
        groups = Counter(r["pair_group"] for r in rows if r["pair_group"])
        for row in rows:
            if row["pair_group"] and groups[row["pair_group"]] != 2:
                continue
            if row["qa_id"] in seen:
                continue
            seen.add(row["qa_id"])
            items.append({**row, "source_run": str(run_dir)})
    return items, {k: sorted(v) for k, v in run_of_work.items()}


def review_row(item: dict) -> dict:
    """검수표 한 행."""
    stage2 = item["verification"].get("stage2", {})
    relation = item["generator"].get("relation", {})
    return {"qa_id": item["qa_id"], "work_id": item["work_id"], "batch": item["batch"],
            "qa_type": item["qa_type"], "pair_group": item["pair_group"] or "", "p": item["p"],
            "read_end_char": item["read_end_char"], "category": item["category"] or "",
            "question": item["question"], "ground_truth": item["ground_truth"],
            "gold_sentence_ids": ids_text(item["gold_evidence"]),
            "gold_evidence_text": evidence_text(item["gold_evidence"]),
            "future_sentence_ids": ids_text(item["future_evidence"]),
            "future_evidence_text": evidence_text(item["future_evidence"]),
            "candidate_future_answer": item["generator"].get("candidate_future_answer", ""),
            "relation_source": relation.get("source", ""), "relation_target": relation.get("target", ""),
            "verifier_reason": stage2.get("reason", "")}


def sample_second(items: list[dict], ratio: float, seed: int) -> list[dict]:
    """작품×유형별로 ratio만큼 표집한다. 관계 변화 쌍은 한 단위로 표집한다."""
    units = defaultdict(list)
    by_group = defaultdict(list)
    for item in items:
        by_group[item["pair_group"] or item["qa_id"]].append(item)
    for members in by_group.values():
        units[(members[0]["work_id"], members[0]["qa_type"])].append(members)
    rng = random.Random(seed)
    chosen = []
    for key in sorted(units):
        group = sorted(units[key], key=lambda m: m[0]["qa_id"])
        count = round(len(group) * ratio)
        if ratio > 0 and count == 0:
            count = 1
        for members in rng.sample(group, min(count, len(group))):
            chosen.extend(members)
    return sorted(chosen, key=lambda r: (r["work_id"], r["qa_type"], r["pair_group"] or "", r["p"]))


def cmd_export(args) -> None:
    """검수표를 만든다."""
    root = Path(args.project_root).resolve()
    items, runs = collect(args.runs)
    if not items:
        raise ValueError("verifier_accepted 후보가 없습니다.")
    out = Path(args.out).resolve() if args.out else (
        root / "data" / "qa" / "review" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    out.mkdir(parents=True, exist_ok=False)
    (out / "sentences").mkdir()
    items.sort(key=lambda r: (r["work_id"], r["qa_type"], r["pair_group"] or "", r["p"], r["qa_id"]))
    write_jsonl(out / "items.jsonl", items)
    write_csv(out / "review_main.csv", CSV_FIELDS, [review_row(i) for i in items])
    second = sample_second(items, args.second_ratio, args.seed)
    write_csv(out / "review_second.csv", CSV_FIELDS, [review_row(i) for i in second])
    for work, run_dirs in runs.items():
        rows = read_jsonl(Path(run_dirs[0]) / "sentences.jsonl")
        write_csv(out / "sentences" / f"{work}.csv", ["id", "tau", "text"], rows)
    summary = {"finalizer_version": FINALIZER_VERSION, "command": "export", "runs": runs,
               "items": len(items), "second_review_items": len(second),
               "second_ratio": args.second_ratio, "seed": args.seed,
               "counts": dict(Counter(f'{r["work_id"]}|{r["qa_type"]}' for r in items))}
    write_json(out / "export_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"검수 폴더: {out}")


# ---------------------------------------------------------------- agreement

def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """두 판정 목록의 Cohen's κ. 기대 일치가 1이면 None."""
    n = len(a)
    if n == 0:
        return None
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb[k] for k in set(ca) | set(cb)) / (n * n)
    if expected == 1:
        return None
    return (observed - expected) / (1 - expected)


def cmd_agreement(args) -> None:
    """주 검수와 두 번째 검수의 decision 일치도를 계산한다."""
    review = Path(args.review_dir).resolve()
    main_rows = {r["qa_id"]: r for r in read_csv(review / "review_main.csv")}
    second_rows = read_csv(Path(args.second_csv).resolve() if args.second_csv
                           else review / "review_second.csv")
    pairs, missing = [], []
    for row in second_rows:
        first = main_rows.get(row["qa_id"])
        if not first or first["decision"] not in DECISIONS or row["decision"] not in DECISIONS:
            missing.append(row["qa_id"])
            continue
        pairs.append((first, row))
    a = [x["decision"] for x, _ in pairs]
    b = [y["decision"] for _, y in pairs]
    keep_a = ["drop" if d == "drop" else "use" for d in a]
    keep_b = ["drop" if d == "drop" else "use" for d in b]
    disagreements = [{"qa_id": x["qa_id"], "work_id": x["work_id"], "qa_type": x["qa_type"],
                      "p": x["p"], "question": x["question"], "main_decision": x["decision"],
                      "second_decision": y["decision"], "main_note": x["note"], "second_note": y["note"]}
                     for x, y in pairs if x["decision"] != y["decision"]]
    result = {"finalizer_version": FINALIZER_VERSION, "command": "agreement",
              "compared": len(pairs), "not_compared": missing,
              "observed_agreement": (sum(x == y for x, y in zip(a, b)) / len(a)) if a else None,
              "kappa_keep_fix_drop": cohen_kappa(a, b),
              "kappa_use_vs_drop": cohen_kappa(keep_a, keep_b),
              "main_counts": dict(Counter(a)), "second_counts": dict(Counter(b)),
              "by_type": {t: {"compared": sum(1 for x, _ in pairs if x["qa_type"] == t),
                              "agree": sum(1 for x, y in pairs if x["qa_type"] == t
                                           and x["decision"] == y["decision"])}
                          for t in sorted({x["qa_type"] for x, _ in pairs})},
              "disagreements": len(disagreements)}
    write_json(review / "agreement.json", result)
    write_csv(review / "disagreements.csv", list(disagreements[0]) if disagreements else ["qa_id"],
              disagreements)
    print(json.dumps(result, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------- build

def char_bigrams(text: str) -> set[str]:
    """질문 유사도용 글자 바이그램."""
    clean = re.sub(r"[\s\W_]+", "", text)
    return {clean[i:i + 2] for i in range(len(clean) - 1)} or {clean}


def jaccard(x: set[str], y: set[str]) -> float:
    """자카드 유사도."""
    return len(x & y) / len(x | y) if x | y else 0.0


def spans(item: dict) -> list[tuple[int, int]]:
    """문항의 근거 구간(gold와 future)."""
    return [tuple(e["original_span"]) for e in item["gold_evidence"] + item["future_evidence"]]


def overlap(x: list[tuple[int, int]], y: list[tuple[int, int]]) -> bool:
    """두 구간 목록이 겹치는가."""
    return any(a < d and c < b for a, b in x for c, d in y)


def group_items(items: list[dict], threshold: float) -> dict[str, str]:
    """관계 쌍과 비슷한 질문(같은 작품, 근거 겹침, 질문 유사도 ≥ threshold)을 한 묶음으로 만든다."""
    parent = {i["qa_id"]: i["qa_id"] for i in items}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    by_pair = defaultdict(list)
    for item in items:
        if item["pair_group"]:
            by_pair[item["pair_group"]].append(item["qa_id"])
    for members in by_pair.values():
        for other in members[1:]:
            union(members[0], other)
    grams = {i["qa_id"]: char_bigrams(i["question"]) for i in items}
    by_work = defaultdict(list)
    for item in items:
        by_work[item["work_id"]].append(item)
    for rows in by_work.values():
        for i, x in enumerate(rows):
            for y in rows[i + 1:]:
                if overlap(spans(x), spans(y)) and jaccard(grams[x["qa_id"]], grams[y["qa_id"]]) >= threshold:
                    union(x["qa_id"], y["qa_id"])
    return {qa_id: find(qa_id) for qa_id in parent}


def apply_fix(item: dict, row: dict, run_text: RunText, text: str) -> list[str]:
    """fix 행을 적용하고 코드 검사를 다시 한다. 실패 사유 목록을 돌려준다."""
    failures, changed = [], []
    if row["evidence_rechecked"].lower() not in YES:
        failures.append("evidence_not_rechecked")
    if row["fixed_category"]:
        if row["fixed_category"] in CATEGORIES:
            item["category"] = row["fixed_category"]
        elif row["fixed_category"].lower() in {"null", "none", "-"}:
            item["category"] = None
        else:
            failures.append("fixed_category_not_in_list")
        changed.append("category")
    if row["fixed_question"]:
        item["question"] = row["fixed_question"]
        changed.append("question")
    if row["fixed_ground_truth"]:
        item["ground_truth"] = row["fixed_ground_truth"]
        changed.append("ground_truth")
    p = item["p"]
    if row["fixed_gold_sentence_ids"]:
        upper = run_text.visible_count(p)
        ids = parse_ids(row["fixed_gold_sentence_ids"], upper)
        if ids is None:
            failures.append("fixed_gold_ids_invalid_or_after_p")
        else:
            item["gold_evidence"] = fill_text(run_text.evidence(ids, run_text.read_end[p - 1]), text)
        changed.append("gold_evidence")
    if row["fixed_future_sentence_ids"]:
        ids = parse_ids(row["fixed_future_sentence_ids"], len(run_text.sentences))
        if ids is None:
            failures.append("fixed_future_ids_invalid")
        else:
            item["future_evidence"] = fill_text(run_text.evidence(ids, None), text)
        changed.append("future_evidence")
    item["_changed"] = changed
    return failures


def fill_text(items: list[dict], text: str) -> list[dict]:
    """근거 항목의 text를 cleaned 원문 좌표로 채운다."""
    for item in items:
        a, b = item["original_span"]
        item["text"] = text[a:b]
    return items


def type_checks(item: dict) -> list[str]:
    """유형별 코드 검사(build_qa_candidates와 같은 조건)."""
    failures = []
    if not item["question"]:
        failures.append("question_missing")
    if item["qa_type"] in {"answerable", "relation_change"}:
        if not item["ground_truth"]:
            failures.append("answer_missing")
        if not item["gold_evidence"]:
            failures.append("gold_evidence_missing")
        if any(e["tau"] is None or e["tau"] > item["p"] for e in item["gold_evidence"]):
            failures.append("gold_tau_after_p")
    if item["qa_type"] == "unanswerable_future":
        if item["gold_evidence"]:
            failures.append("gold_evidence_not_empty")
        if not item["future_evidence"]:
            failures.append("future_evidence_missing")
        if any(e["tau"] is None or e["tau"] <= item["p"] for e in item["future_evidence"]):
            failures.append("future_evidence_not_after_p")
    return failures


def pair_checks(members: list[dict]) -> list[str]:
    """관계 변화 쌍 검사: 같은 질문, p1 < p2, after 근거 중 τ > p1."""
    if len(members) != 2:
        return ["pair_incomplete"]
    before, after = sorted(members, key=lambda m: m["p"])
    failures = []
    if before["question"] != after["question"]:
        failures.append("pair_question_differs")
    if not before["p"] < after["p"]:
        failures.append("p1_not_before_p2")
    if not any(e["tau"] > before["p"] for e in after["gold_evidence"]):
        failures.append("no_after_evidence_after_p1")
    return failures


def cmd_build(args) -> None:
    """검수 결과를 반영해 최종 평가셋을 만들고 동결한다."""
    root = Path(args.project_root).resolve()
    review = Path(args.review_dir).resolve()
    out = Path(args.out).resolve() if args.out else root / "data" / "qa" / "final"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "qa.jsonl").exists() and not args.overwrite:
        raise FileExistsError(f"{out / 'qa.jsonl'}가 이미 있습니다. 동결된 평가셋은 --overwrite 없이 바꾸지 않습니다.")
    items = {r["qa_id"]: r for r in read_jsonl(review / "items.jsonl")}
    rows = {r["qa_id"]: r for r in read_csv(review / "review_main.csv")}
    missing = [q for q in items if q not in rows or rows[q]["decision"] not in DECISIONS]
    if missing:
        raise ValueError(f"decision이 keep/fix/drop이 아닌 문항 {len(missing)}개: {missing[:10]}")
    run_cache, text_cache = {}, {}
    kept, dropped = [], []
    for qa_id, item in items.items():
        row = rows[qa_id]
        item = json.loads(json.dumps(item))
        human = {"decision": row["decision"], "reviewer": row["reviewer"], "note": row["note"],
                 "evidence_rechecked": row["evidence_rechecked"], "changed_fields": []}
        if row["decision"] == "drop":
            dropped.append({"qa_id": qa_id, "reason": "human_drop"})
            continue
        if row["decision"] == "fix":
            run_dir = item["source_run"]
            run_text = run_cache.setdefault(run_dir, RunText(Path(run_dir)))
            text = text_cache.setdefault(item["work_id"], load_text(
                root / "data" / "cleaned", item["work_id"], item["document_sha256"]))
            failures = apply_fix(item, row, run_text, text)
            human["changed_fields"] = item.pop("_changed")
            failures += type_checks(item)
            if failures:
                raise ValueError(f"{qa_id}: 수정 내용이 코드 검사를 통과하지 못했습니다: {failures}")
        item["verification"]["human"] = human
        kept.append(item)
    by_pair = defaultdict(list)
    for item in kept:
        if item["pair_group"]:
            by_pair[item["pair_group"]].append(item)
    removed = set()
    for group, members in by_pair.items():
        failures = pair_checks(members)
        if failures == ["pair_incomplete"]:
            for m in members:
                removed.add(m["qa_id"])
                dropped.append({"qa_id": m["qa_id"], "reason": "pair_incomplete"})
        elif failures:
            raise ValueError(f"{group}: 관계 변화 쌍 검사 실패: {failures}")
    kept = [i for i in kept if i["qa_id"] not in removed]
    groups = group_items(kept, args.similarity)
    dev_groups = {groups[i["qa_id"]] for i in kept if i["batch"] == "dev"}
    final = []
    for item in sorted(kept, key=lambda r: (r["work_id"], r["qa_type"], r["pair_group"] or "", r["p"], r["qa_id"])):
        item["split"] = "dev" if groups[item["qa_id"]] in dev_groups else "test"
        final.append({k: item[k] for k in FINAL_FIELDS})
    inputs = [{"qa_id": r["qa_id"], "work_id": r["work_id"], "p": r["p"], "question": r["question"]}
              for r in final]
    write_jsonl(out / "qa.jsonl", final)
    write_jsonl(out / "qa_inputs.jsonl", inputs)
    group_members = defaultdict(list)
    for item in final:
        group_members[groups[item["qa_id"]]].append(item["qa_id"])
    pairs = Counter(r["split"] for r in final if r["qa_type"] == "relation_change")
    summary = {"finalizer_version": FINALIZER_VERSION, "command": "build",
               "review_dir": str(review), "created_at": datetime.now(timezone.utc).isoformat(),
               "items": len(final), "dropped": dropped,
               "by_split_type": dict(Counter(f'{r["split"]}|{r["qa_type"]}' for r in final)),
               "by_work_type": dict(Counter(f'{r["work_id"]}|{r["qa_type"]}' for r in final)),
               "relation_change_pairs_by_split": {k: v // 2 for k, v in pairs.items()},
               "decisions": dict(Counter(rows[q]["decision"] for q in items)),
               "similarity_threshold": args.similarity,
               "split_groups": [m for m in group_members.values() if len(m) > 1],
               "qa_sha256": sha256_file(out / "qa.jsonl"),
               "qa_inputs_sha256": sha256_file(out / "qa_inputs.jsonl")}
    write_json(out / "summary.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k not in {"dropped", "split_groups"}},
                     ensure_ascii=False, indent=2))
    print(f"최종 평가셋: {out / 'qa.jsonl'}")


def build_parser() -> argparse.ArgumentParser:
    """CLI 인자를 정의한다."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-root", default=".")
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export", help="검수표 만들기")
    export.add_argument("--runs", nargs="+", required=True, help="data/qa_candidates/<작품>/<실행시각> 폴더들")
    export.add_argument("--out", default=None)
    export.add_argument("--second-ratio", type=float, default=0.25, help="두 번째 검수 표집 비율")
    export.add_argument("--seed", type=int, default=0)
    agreement = sub.add_parser("agreement", help="검수 일치도 계산")
    agreement.add_argument("--review-dir", required=True)
    agreement.add_argument("--second-csv", default=None, help="기본값: <review-dir>/review_second.csv")
    build = sub.add_parser("build", help="최종 평가셋 만들기")
    build.add_argument("--review-dir", required=True)
    build.add_argument("--out", default=None, help="기본값: data/qa/final")
    build.add_argument("--similarity", type=float, default=0.5,
                       help="같은 묶음으로 볼 질문 글자 바이그램 자카드 유사도 하한")
    build.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    """명령을 실행한다."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "export" and not 0 <= args.second_ratio <= 1:
        parser.error("second-ratio는 0~1이어야 합니다.")
    try:
        {"export": cmd_export, "agreement": cmd_agreement, "build": cmd_build}[args.command](args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"실행 실패: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
