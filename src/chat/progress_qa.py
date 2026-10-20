"""독서 진행도 기반 E²RAG 확장 QA 엔진.

기존 벡터 인덱스, Entity–Event–Relation 그래프와 원문 근거를 읽기 전용으로 사용한다.
독자가 완료한 chunk 번호 p를 입력받고 공개 시점 τ <= p를 엄격히 적용한다.

원논문의 Entity/Event 검색 및 Event 중심 one-hop 확장을 참고한다.
본 프로젝트의 Relation 검색과 Relation–Event 확장은 별도의 추가 설계이다.
평가용 QA 정답 파일을 읽지 않는다.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_DIR = ROOT / "src" / "embedding"
if str(EMBEDDING_DIR) not in sys.path:
    sys.path.insert(0, str(EMBEDDING_DIR))

from vector_index import Embedder, VectorIndex  # noqa: E402

MODEL = "gpt-4o-mini"
SEED_STORES = ("entity", "event", "relation")
EXPAND_EDGE_TYPES = {"entity_event", "event_event", "relation_event",
                     "entity_relation_source", "entity_relation_target"}


def read_json(path: Path) -> dict:
    """파일의 JSON 객체를 읽는다."""
    with path.open(encoding="utf-8-sig") as fp:
        value = json.load(fp)
    if not isinstance(value, dict):
        raise ValueError(f"JSON 객체가 아닙니다: {path}")
    return value


def get_client(root: Path):
    """프로젝트 루트의 .env 또는 OS 환경 변수에서 API 키를 읽는다."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(root / ".env", override=False)
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(f"OPENAI_API_KEY가 없습니다. {root / '.env'}를 확인하세요.")
    return OpenAI(max_retries=1, timeout=60.0)


def node_key(row: dict) -> str:
    """인덱스 node_id를 그래프 노드 id로 변환한다."""
    return f"{row['node_type']}:{row['node_id']}"


def validate_sources(index: VectorIndex, graph: dict, report: dict) -> None:
    """벡터 인덱스와 그래프의 작품·원본 버전·노드 ID 일치를 검증한다."""
    if report.get("status") != "OK":
        raise ValueError("그래프 검증 보고서가 OK가 아닙니다.")
    if graph.get("work_id") != index.work_id or report.get("work_id") != index.work_id:
        raise ValueError("그래프와 벡터 인덱스의 작품이 다릅니다.")
    if graph.get("source_fingerprints") != report.get("input_manifest"):
        raise ValueError("그래프 결과와 검증 보고서가 일치하지 않습니다.")
    if graph.get("last_chunk") != len(index.manifest["read_end_char"]):
        raise ValueError("그래프와 벡터 인덱스의 총 Chunk 수가 다릅니다.")

    graph_inputs = graph["source_fingerprints"]
    vector_inputs = index.manifest["inputs"]
    for store in ("chunk", "entity", "event", "relation"):
        graph_hash = graph_inputs.get(store, {}).get("sha256")
        vector_hash = vector_inputs.get(store, {}).get("sha256")
        if not graph_hash or graph_hash != vector_hash:
            raise ValueError(f"그래프와 벡터 인덱스의 {store} 원본 버전이 다릅니다.")
    graph_ids = {n["id"] for n in graph["nodes"]}
    for store in SEED_STORES:
        ids = {node_key(row) for row in index.meta[store]}
        actual = {name for name in graph_ids if name.startswith(store + ":")}
        if actual != ids:
            raise ValueError(f"{store} 검색 레코드와 그래프 노드가 다릅니다.")


def filter_graph(full: dict, p: int) -> dict:
    """QA 검색 전에 노드 및 엣지의 τ와 양쪽 종단을 함께 제한한다."""
    if type(p) is not int or not 1 <= p <= full["last_chunk"]:
        raise ValueError("독서 진행도 p가 범위를 벗어났습니다.")
    nodes = {n["id"]: n for n in full["nodes"] if n["tau"] <= p}
    edges = [e for e in full["edges"]
             if e["tau"] <= p and e["source"] in nodes and e["target"] in nodes]
    return {"nodes": nodes, "edges": edges}


def extract_cues(question: str, client: Any) -> dict[str, str]:
    """질문에 실제로 등장하는 개체·사건·관계 검색 단서를 분리한다."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": (
                "한국어 문학 QA 질문의 검색 단서를 분리한다. 질문에 없는 작품 정보나 "
                "이후 전개를 추측해 추가하지 않는다. 반드시 JSON 객체만 출력한다. "
                "키는 entity, event, relation이며 값은 각각 간단한 검색용 한국어 문자열이다. "
                "해당 단서가 없으면 질문의 관련 표현을 그대로 사용한다."
            )},
            {"role": "user", "content": question},
        ],
    )
    try:
        obj = json.loads(response.choices[0].message.content or "")
    except (ValueError, IndexError, AttributeError) as exc:
        raise ValueError("질문 단서 추출 결과가 JSON 형식이 아닙니다.") from exc
    if not isinstance(obj, dict):
        raise ValueError("질문 단서 추출 결과가 객체가 아닙니다.")
    return {store: (str(obj.get(store) or "").strip()[:300] or question)
            for store in SEED_STORES}


def retrieve_context(index: VectorIndex, full: dict, p: int, question: str,
                     cues: dict[str, str], top_k: int = 5, passage_k: int = 5,
                     embedder: Embedder | None = None) -> dict:
    """τ<=p 검색 → Event/Relation 시드 one-hop 확장 → 원문 근거를 조립한다."""
    index.check_progress(p)
    if not question.strip() or top_k < 1 or passage_k < 1:
        raise ValueError("질문과 검색 개수를 확인하세요.")

    visible = filter_graph(full, p)
    nodes, edges = visible["nodes"], visible["edges"]
    query_texts = [cues[store] for store in SEED_STORES]
    vectors = (embedder or Embedder(index.root)).encode(query_texts)
    seed_results = {store: index.search(store, p, top_k=top_k,
                                        query_vector=vectors[i])
                    for i, store in enumerate(SEED_STORES)}

    scores: dict[str, float] = {}
    seeds: dict[str, set[str]] = defaultdict(set)
    for store, results in seed_results.items():
        for item in results:
            key = node_key(item)
            if key not in nodes:
                raise ValueError(f"인덱스 검색 결과가 그래프에 없습니다: {key}")
            score = max(0.0, float(item["score"]))
            if score <= 0:
                continue
            scores[key] = max(scores.get(key, 0.0), score)
            seeds[store].add(key)

    # 확장은 단 한 번만 실행한다. 새로 얻은 노드에서 연쇄 확장하지 않는다.
    frontier = seeds["event"] | seeds["relation"]
    expansion_types = {"entity_event", "event_event", "relation_event"}
    for edge in edges:
        kind = edge["type"]
        if kind not in EXPAND_EDGE_TYPES:
            continue
        source, target = edge["source"], edge["target"]
        if kind not in expansion_types and not (
            kind.startswith("entity_relation_") and (source in seeds["relation"] or target in seeds["relation"])
        ):
            continue
        if source in frontier:
            scores[target] = max(scores.get(target, 0.0), 0.85 * scores[source])
        if target in frontier:
            scores[source] = max(scores.get(source, 0.0), 0.85 * scores[target])

    # 원문 Passage는 별도 Chunk 벡터 검색 결과를 섞지 않고 선택 노드의 chunk_id로 조회한다.
    chunk_scores: dict[int, float] = {}
    for key, score in scores.items():
        chunk_id = int(nodes[key]["data"]["chunk_id"])
        chunk_scores[chunk_id] = max(chunk_scores.get(chunk_id, 0.0), score)
    ranked_chunks = sorted(chunk_scores, key=lambda cid: (-chunk_scores[cid], cid))[:passage_k]
    chunk_lookup = {int(row["node_id"]): row for row in index.meta["chunk"] if row["tau"] <= p}
    passages = []
    read_end = index.read_end(p)
    for cid in ranked_chunks:
        row = chunk_lookup.get(cid)
        if row is None or row["evidence_original_span"][1] > read_end:
            raise ValueError(f"진행도에 맞는 Chunk 근거가 없습니다: {cid}")
        passages.append({"chunk_id": cid, "tau": row["tau"],
                         "text": row["evidence"], "score": round(chunk_scores[cid], 6)})

    # 선택된 Passage에 실제로 대응되는 노드와 링크만 전달한다.
    chosen_ids = {key for key in scores if int(nodes[key]["data"]["chunk_id"]) in ranked_chunks}
    selected_nodes = []
    for key in sorted(chosen_ids, key=lambda k: (-scores[k], k)):
        row = nodes[key]
        desc = str(row["data"].get("description") or row["data"].get("summary") or "")
        # 인덱스에 기록된 근거 경계와 설명 공개 조건을 QA에서도 다시 적용한다.
        store, ident = key.split(":", 1)
        meta = next((m for m in index.meta[store] if str(m["node_id"]) == ident), None)
        if meta is None:
            raise ValueError(f"노드 metadata가 없습니다: {key}")
        if meta["tau"] > p or meta["evidence_original_span"][1] > read_end:
            raise ValueError(f"비공개 근거가 포함되었습니다: {key}")
        if not meta["description_used"]:
            desc = ""
        else:
            # 원본 그래프 description 대신 경계 검증된 인덱스 설명을 사용한다.
            desc = meta["description"]
        item = {"id": key, "type": store, "tau": row["tau"],
                "chunk_id": int(row["data"]["chunk_id"]),
                "description": desc, "evidence": meta["evidence"]}
        if store == "entity":
            item["name"] = meta.get("canonical_name", "")
        elif store == "event":
            item["name"] = meta.get("label", "")
        else:
            item["name"] = f"{meta.get('source_name', '')} - {meta.get('target_name', '')}"
            item["relation_type"] = meta.get("relation_type")
        selected_nodes.append(item)
    selected_edges = [{"type": e["type"], "source": e["source"], "target": e["target"],
                       "tau": e["tau"]} for e in edges
                      if e["source"] in chosen_ids and e["target"] in chosen_ids]
    assert all(item["tau"] <= p for item in selected_nodes + selected_edges + passages)
    return {"passages": passages, "nodes": selected_nodes, "edges": selected_edges,
            "seed_counts": {s: len(seeds[s]) for s in SEED_STORES}, "p": p}


def generate_answer(question: str, context: dict, client: Any) -> str:
    """검색된 근거만 전달하고 Chunk 번호를 인용하는 답변을 요청한다."""
    if not context["passages"]:
        return "현재 읽은 범위에서 질문에 답할 수 있는 근거를 찾지 못했습니다."
    payload = {
        "passages": context["passages"],
        "entity_event_relation": context["nodes"][:45],
        "graph_links": context["edges"][:70],
    }
    answer = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": (
                "너는 고전문학 독자를 위한 근거 기반 QA 도우미다. "
                "제공된 검색 문맥 이외의 줄거리, 외부 지식, 이후 전개를 절대 사용하지 마라. "
                "Passage 원문을 최우선 근거로 삼고 구조화된 노드 설명은 보조 단서로만 사용하라. "
                "질문이 미래를 전제하더라도 현재 공개된 근거로 확인되는 범위만 답하라. "
                "근거로 뒷받침되는 사실만 간결하게 설명하고 문장 끝에 [Chunk N] 형식으로 "
                "실제 제공된 chunk를 인용하라. 근거가 부족하면 확인할 수 없다고 대답하라. "
                "검색 문맥에 없는 chunk 번호를 만들지 마라."
            )},
            {"role": "user", "content": (
                f"독서 진행도 p={context['p']} (여기까지 완료).\n"
                f"질문: {question}\n\n검색 문맥(JSON):\n"
                + json.dumps(payload, ensure_ascii=False)
            )},
        ],
    ).choices[0].message.content
    if not answer or not answer.strip():
        raise ValueError("답변 모델에서 빈 응답이 반환되었습니다.")
    cited = [int(v) for v in re.findall(r"\[Chunk\s+(\d+)\]", answer)]
    allowed = {row["chunk_id"] for row in context["passages"]}
    if any(x not in allowed for x in cited):
        raise ValueError("GPT 답변에 검색되지 않은 Chunk 출처가 포함되었습니다.")
    return answer.strip()


class ProgressQA:
    """선택한 작품의 원본 호환성을 검사한 후 여러 질문을 처리하는 엔진."""

    def __init__(self, work_id: str, root: Path = ROOT):
        self.root = Path(root).resolve()
        self.index = VectorIndex(self.root, work_id)
        folder = self.root / "data" / "progress_graph" / work_id
        self.full = read_json(folder / "full_graph.json")
        report = read_json(folder / "validation_report.json")
        validate_sources(self.index, self.full, report)
        self.embedder = Embedder(self.root)

    def ask(self, p: int, question: str, top_k: int = 5, passage_k: int = 5,
            use_cues: bool = True) -> dict:
        """한 번의 질문을 처리한다. 평가 정답이나 미래 노드는 사용하지 않는다."""
        self.index.check_progress(p)
        question = question.strip()
        if not question:
            raise ValueError("질문을 입력하세요.")
        if len(question) > 1000:
            raise ValueError("질문은 최대 1000자까지 가능합니다.")
        client = get_client(self.root)
        cues = extract_cues(question, client) if use_cues else {s: question for s in SEED_STORES}
        context = retrieve_context(self.index, self.full, p, question, cues,
                                   top_k=top_k, passage_k=passage_k, embedder=self.embedder)
        answer = generate_answer(question, context, client)
        return {"work_id": self.index.work_id, "p": p, "question": question,
                "answer": answer, "passages": context["passages"],
                "seed_counts": context["seed_counts"],
                "retrieved_nodes": len(context["nodes"]),
                "graph_links": len(context["edges"])}


def main() -> None:
    """UI 없이 QA 엔진만 단독으로 확인할 때 쓰는 명령행 진입점."""
    parser = argparse.ArgumentParser(description="독서 진행도 기반 E²RAG QA")
    parser.add_argument("--work-id", required=True)
    parser.add_argument("--p", required=True, type=int)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--passage-k", type=int, default=5)
    parser.add_argument("--no-cues", action="store_true", help="질문 단서 추출을 생략")
    args = parser.parse_args()
    engine = ProgressQA(args.work_id)
    result = engine.ask(args.p, args.query, args.top_k, args.passage_k,
                        use_cues=not args.no_cues)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
