"""독서 진행도 기반 기본 E²RAG 비교용 QA 엔진.

기존 Entity·Event 인덱스와 Entity–Event / Event–Event 그래프 엣지만 검색한다.
Relation Observation, Relation–Event 연결, Relation 인덱스는 검색하지 않는다.
원논문 전체 코드를 그대로 복제한 구현은 아니며 기존 데이터 기반의 Entity–Event 축소 비교 모델이다.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
EMBEDDING_DIR = ROOT / "src" / "embedding"
for directory in (ROOT, EMBEDDING_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from src.embedding.vector_index import Embedder, VectorIndex  # noqa: E402
from src.chat.progress_qa import (  # noqa: E402
    MODEL,
    filter_graph,
    generate_answer,
    get_client,
    node_key,
    read_json,
    validate_sources,
)

SEED_STORES = ("entity", "event")
ALLOWED_EDGE_TYPES = {"entity_event", "event_event"}


def extract_cues(question: str, client: Any) -> dict[str, str]:
    """기본 E²RAG와 같이 질문에서 Entity·Event 단서만 추출한다."""
    response = client.chat.completions.create(
        model=MODEL,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": (
                "한국어 문학 QA 질문에서 Entity와 Event 검색 단서를 추출한다. "
                "질문에 없는 인물이나 사건을 추측하거나 이후 줄거리를 추가하지 않는다. "
                "반드시 entity, event 두 키를 가진 JSON 객체만 출력한다. "
                "값은 각각 간단한 한국어 검색 문자열이며 단서가 없으면 질문의 관련 표현을 사용한다."
            )},
            {"role": "user", "content": question},
        ],
    )
    try:
        obj = json.loads(response.choices[0].message.content or "")
    except (ValueError, IndexError, AttributeError) as exc:
        raise ValueError("Entity·Event 단서 추출 응답이 JSON이 아닙니다.") from exc
    if not isinstance(obj, dict):
        raise ValueError("단서 추출 결과는 JSON 객체여야 합니다.")
    return {kind: (str(obj.get(kind) or "").strip()[:300] or question)
            for kind in SEED_STORES}


def retrieve_context(
    index: VectorIndex,
    full: dict,
    p: int,
    question: str,
    cues: dict[str, str],
    top_k: int = 5,
    passage_k: int = 5,
    embedder: Embedder | None = None,
) -> dict:
    """진행도 필터 → Entity/Event 시드 → Event 중심 one-hop → Chunk 근거 조립."""
    index.check_progress(p)
    if not question.strip() or top_k < 1 or passage_k < 1:
        raise ValueError("질문과 검색 개수를 확인하세요.")

    visible = filter_graph(full, p)
    # 확장 기능이 검색 결과에 유입되지 않도록 노드와 엣지를 명시적으로 화이트리스트로 제한한다.
    nodes = {key: row for key, row in visible["nodes"].items()
             if row["type"] in SEED_STORES}
    edges = [row for row in visible["edges"]
             if row["type"] in ALLOWED_EDGE_TYPES
             and row["source"] in nodes and row["target"] in nodes]

    vectors = (embedder or Embedder(index.root)).encode([cues[s] for s in SEED_STORES])
    results = {store: index.search(store, p, top_k=top_k, query_vector=vectors[i])
               for i, store in enumerate(SEED_STORES)}

    scores: dict[str, float] = {}
    seeds: dict[str, set[str]] = defaultdict(set)
    for store, rows in results.items():
        for item in rows:
            key = node_key(item)
            if key not in nodes:
                raise ValueError(f"인덱스 검색 결과가 기본 그래프에 없습니다: {key}")
            score = max(0.0, float(item["score"]))
            if score <= 0:
                continue
            scores[key] = max(scores.get(key, 0.0), score)
            seeds[store].add(key)

    # 원논문에서 Event 시드만을 확장 출발점으로 사용하는 one-hop 방식을 따른다.
    # 새로 도달한 노드는 다시 확장하지 않는다. 이벤트와 연결된 Entity/Event에 점수를 전파한다.
    event_seed_scores = {key: scores[key] for key in seeds["event"]}
    for edge in edges:
        source, target = edge["source"], edge["target"]
        for start, neighbor in ((source, target), (target, source)):
            if start in event_seed_scores:
                scores[neighbor] = max(scores.get(neighbor, 0.0),
                                       0.85 * event_seed_scores[start])

    # 원문 Chunk는 선택된 Entity/Event의 실제 chunk_id에서만 가져온다.
    chunk_scores: dict[int, float] = {}
    for key, score in scores.items():
        cid = int(nodes[key]["data"]["chunk_id"])
        if cid > p:
            raise ValueError(f"미래 노드가 검색되었습니다: {key}")
        chunk_scores[cid] = max(chunk_scores.get(cid, 0.0), score)
    ranked_chunks = sorted(chunk_scores, key=lambda cid: (-chunk_scores[cid], cid))[:passage_k]
    read_end = index.read_end(p)
    chunk_lookup = {int(row["node_id"]): row for row in index.meta["chunk"] if row["tau"] <= p}
    passages = []
    for cid in ranked_chunks:
        row = chunk_lookup.get(cid)
        if row is None or row["evidence_original_span"][1] > read_end:
            raise ValueError(f"진행도 내 Chunk 근거를 확인할 수 없습니다: {cid}")
        passages.append({"chunk_id": cid, "tau": row["tau"],
                         "text": row["evidence"], "score": round(chunk_scores[cid], 6)})

    chosen_ids = {key for key in scores
                  if int(nodes[key]["data"]["chunk_id"]) in ranked_chunks}
    metadata = {store: {str(item["node_id"]): item for item in index.meta[store]}
                for store in SEED_STORES}
    selected_nodes = []
    for key in sorted(chosen_ids, key=lambda x: (-scores[x], x)):
        kind, ident = key.split(":", 1)
        graph_row = nodes[key]
        item = metadata[kind].get(ident)
        if item is None:
            raise ValueError(f"벡터 metadata에서 노드를 찾을 수 없습니다: {key}")
        if item["tau"] > p or item["evidence_original_span"][1] > read_end:
            raise ValueError(f"미래 근거가 포함되었습니다: {key}")
        row = {"id": key, "type": kind, "tau": graph_row["tau"],
               "chunk_id": int(graph_row["data"]["chunk_id"]),
               "description": item["description"] if item["description_used"] else "",
               "evidence": item["evidence"],
               "name": item.get("canonical_name", "") if kind == "entity"
               else item.get("label", "")}
        selected_nodes.append(row)

    selected_edges = [{"type": e["type"], "source": e["source"],
                       "target": e["target"], "tau": e["tau"]}
                      for e in edges if e["source"] in chosen_ids and e["target"] in chosen_ids]
    if not all(row["tau"] <= p for row in passages + selected_nodes + selected_edges):
        raise ValueError("검색 문맥의 공개 시점이 진행도를 초과했습니다.")
    return {"passages": passages, "nodes": selected_nodes, "edges": selected_edges,
            "seed_counts": {s: len(seeds[s]) for s in SEED_STORES}, "p": p}


class BasicE2RAGQA:
    """기존 제안 시스템과 동일한 호출 인터페이스를 제공하는 비교용 QA 엔진."""

    def __init__(self, work_id: str, root: Path = ROOT):
        self.root = Path(root).resolve()
        self.index = VectorIndex(self.root, work_id)
        folder = self.root / "data" / "progress_graph" / work_id
        self.full = read_json(folder / "full_graph.json")
        report = read_json(folder / "validation_report.json")
        # 버전 검증은 기존 제안 시스템과 동일하게 실시한다.
        validate_sources(self.index, self.full, report)
        self.embedder = Embedder(self.root)

    def ask(self, p: int, question: str, top_k: int = 5, passage_k: int = 5,
            use_cues: bool = True) -> dict:
        """진행도 p의 현재 원문만 근거로 답변을 생성한다."""
        self.index.check_progress(p)
        question = question.strip()
        if not question or len(question) > 1000:
            raise ValueError("질문은 1~1000자여야 합니다.")
        client = get_client(self.root)
        cues = (extract_cues(question, client) if use_cues else
                {store: question for store in SEED_STORES})
        context = retrieve_context(self.index, self.full, p, question, cues,
                                   top_k=top_k, passage_k=passage_k,
                                   embedder=self.embedder)
        # 제안 시스템에서 실제 사용하는 생성 함수를 공유하여 최종 답변 지침을 동일하게 유지한다.
        answer = generate_answer(question, context, client)
        return {"work_id": self.index.work_id, "p": p, "question": question,
                "answer": answer, "passages": context["passages"],
                "seed_counts": context["seed_counts"],
                "retrieved_nodes": len(context["nodes"]),
                "graph_links": len(context["edges"])}


def main() -> None:
    """명령행에서 작품·진행도·질문을 지정해 기본 E²RAG를 테스트한다."""
    parser = argparse.ArgumentParser(description="기본 E²RAG 진행도 기반 QA 비교 모델")
    parser.add_argument("--work-id", required=True)
    parser.add_argument("--p", required=True, type=int)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--passage-k", type=int, default=5)
    parser.add_argument("--no-cues", action="store_true", help="질문 단서 추출 생략")
    args = parser.parse_args()
    engine = BasicE2RAGQA(args.work_id)
    result = engine.ask(args.p, args.query, args.top_k, args.passage_k,
                        use_cues=not args.no_cues)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
