"""8단계 벡터 검색. τ는 정보 위치이고 p는 검색 요청의 독서 진행도다."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

MODEL = "text-embedding-3-small"
DIM = 1536
STORES = ("entity", "event", "relation", "chunk")
E2_STORES = STORES[:3]
VERSION = "embedding.v1"


def digest(value):
    """텍스트 또는 파일 바이트의 SHA-256을 반환한다."""
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def read_jsonl(path):
    """빈 줄을 제외한 UTF-8 JSONL 객체를 읽는다."""
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def write_json(path, value):
    """완성된 JSON 파일을 원자적으로 교체한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def normalize(values):
    """유한한 비영벡터를 검사하고 행별 L2 정규화한다."""
    a = np.asarray(values, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != DIM or not np.isfinite(a).all():
        raise ValueError("벡터 차원 또는 값 오류")
    norms = np.linalg.norm(a.astype(np.float64), axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("영벡터가 있습니다.")
    return (a / norms).astype(np.float32)


class Embedder:
    """OpenAI 배치 호출과 디스크 캐시. 같은 캐시에는 한 프로세스만 쓴다."""

    def __init__(self, root, batch_size=64, client=None):
        if not 1 <= batch_size <= 256:
            raise ValueError("batch_size는 1~256이어야 합니다.")
        self.root, self.batch_size = Path(root), batch_size
        self.path = self.root / "data/vector_index/_cache/embeddings.json"
        self.cache = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.client = client
        self.calls = self.hits = 0

    def encode(self, texts):
        """입력 순서를 보존한다. 토큰 한도와 응답 index 및 차원을 검증한다."""
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        keys = [f"{MODEL}:{DIM}:{digest(s)}" for s in texts]
        missing = {}
        for key, value in zip(keys, texts):
            if not value.strip() or len(enc.encode(value, disallowed_special=())) > 8191:
                raise ValueError("임베딩 입력이 비었거나 8191토큰을 초과했습니다.")
            if key not in self.cache:
                missing[key] = value
            else:
                self.hits += 1
        if missing and self.client is None:
            from dotenv import load_dotenv
            from openai import OpenAI
            load_dotenv(self.root / ".env", override=False)
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("프로젝트 .env에 OPENAI_API_KEY가 필요합니다.")
            self.client = OpenAI(max_retries=0, timeout=60.0)
        pairs, batches, batch, tokens = list(missing.items()), [], [], 0
        for pair in pairs:
            size = len(enc.encode(pair[1], disallowed_special=()))
            if batch and (len(batch) >= self.batch_size or tokens + size > 200000):
                batches.append(batch)
                batch, tokens = [], 0
            batch.append(pair)
            tokens += size
        if batch:
            batches.append(batch)
        done = 0
        for batch in batches:
            for attempt in range(5):
                try:
                    self.calls += 1
                    response = self.client.embeddings.create(
                        model=MODEL, dimensions=DIM, encoding_format="float",
                        input=[s for _, s in batch])
                    break
                except Exception as exc:
                    status = getattr(exc, "status_code", None)
                    retry = status in (408, 409, 429) or (status is not None and status >= 500)
                    retry |= type(exc).__name__ in ("APIConnectionError", "APITimeoutError")
                    if not retry or attempt == 4:
                        raise
                    time.sleep(2 ** attempt)
            ordered = sorted(response.data, key=lambda x: x.index)
            if [x.index for x in ordered] != list(range(len(batch))):
                raise ValueError("Embedding API 응답 index가 입력과 다릅니다.")
            values = [x.embedding for x in ordered]
            normalize(values)
            for (key, _), value in zip(batch, values):
                self.cache[key] = value
            write_json(self.path, self.cache)
            done += len(batch)
            print(f"[embedding] {done}/{len(pairs)}", flush=True)
        if not keys:
            return np.empty((0, DIM), dtype=np.float32)
        return normalize([self.cache[k] for k in keys])


class VectorIndex:
    """작품 한 편의 인덱스를 읽는다. 그래프 확장이나 답변 생성은 하지 않는다."""

    def __init__(self, root, work_id):
        self.root = Path(root).resolve()
        if not work_id or Path(work_id).name != work_id or any(c in work_id for c in "/\\:"):
            raise ValueError("work_id는 작품 파일의 stem이어야 합니다.")
        self.folder = self.root / "data/vector_index" / work_id
        self.manifest = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        if (self.manifest["work_id"] != work_id or self.manifest["version"] != VERSION
                or self.manifest["model"] != MODEL or self.manifest["dimensions"] != DIM):
            raise ValueError("작품 또는 인덱스 버전/모델 불일치")
        self.work_id = work_id
        self.vectors, self.meta = {}, {}
        for store in STORES:
            for name in (f"{store}.npy", f"{store}.meta.jsonl"):
                if digest((self.folder / name).read_bytes()) != self.manifest["outputs"][name]:
                    raise ValueError(f"인덱스 파일 해시 불일치: {name}")
            a = np.load(self.folder / f"{store}.npy", allow_pickle=False)
            rows = read_jsonl(self.folder / f"{store}.meta.jsonl")
            if a.dtype != np.float32 or a.shape != (len(rows), DIM):
                raise ValueError(f"벡터와 metadata 불일치: {store}")
            if len(rows) != self.manifest["counts"][store]:
                raise ValueError("manifest 건수 불일치")
            normalize(a)
            if not np.allclose(np.linalg.norm(a, axis=1), 1, atol=2e-6):
                raise ValueError("정규화되지 않은 벡터")
            ids = [r["node_id"] for r in rows]
            if len(ids) != len(set(ids)):
                raise ValueError("중복 ID")
            for r in rows:
                if r["work_id"] != work_id or r["node_type"] != store:
                    raise ValueError("다른 작품 또는 저장소 혼입")
                self.check_progress(r["tau"])
                if r["available_at_chunk"] != r["tau"]:
                    raise ValueError("τ 필드 불일치")
                if r["evidence_original_span"][1] > self.read_end(r["tau"]):
                    raise ValueError("근거 공개 경계 초과")
                if r["description_used"] and not r["description_basis_complete"]:
                    raise ValueError("불완전 근거의 description 사용")
                if r["text_sha256"] != digest(r["embed_text"]):
                    raise ValueError("검색 텍스트 해시 오류")
            self.vectors[store], self.meta[store] = a, rows

    def check_progress(self, p):
        """현재 구현의 p는 공개 완료 chunk 번호다. t와 동일시하지 않는다."""
        if type(p) is not int or not 1 <= p <= len(self.manifest["read_end_char"]):
            raise ValueError("p는 1부터 마지막 chunk까지의 정수여야 합니다.")

    def read_end(self, p):
        """해당 진행도에서 공개된 원문의 끝 문자 좌표를 반환한다."""
        self.check_progress(p)
        return self.manifest["read_end_char"][p - 1]

    def search(self, store, p, top_k=10, query_text=None, query_vector=None, embedder=None):
        """작품 선택 후 τ≤p 사전 필터를 적용하고 정확 코사인 검색한다."""
        self.check_progress(p)
        if store not in STORES or type(top_k) is not int or top_k < 1:
            raise ValueError("저장소 또는 top_k 오류")
        if (query_text is None) == (query_vector is None):
            raise ValueError("query_text와 query_vector 중 하나만 지정하세요.")
        candidates = [i for i, r in enumerate(self.meta[store]) if r["tau"] <= p]
        if not candidates:
            return []
        if query_vector is None:
            query_vector = (embedder or Embedder(self.root)).encode([query_text])[0]
        q = normalize([query_vector])[0]
        scores = self.vectors[store][candidates] @ q
        order = sorted(range(len(candidates)),
                       key=lambda j: (-float(scores[j]), self.meta[store][candidates[j]]["node_id"]))
        return [{**self.meta[store][candidates[j]], "score": float(scores[j]),
                 "p": p, "read_end_char_p": self.read_end(p),
                 "index_build_id": self.manifest["build_id"]} for j in order[:top_k]]

    def search_e2rag(self, p, query_text=None, query_vector=None, top_k=10, embedder=None):
        """세 종류의 seed 후보를 반환한다. Chunk는 이 경로에 섞지 않는다."""
        self.check_progress(p)
        if (query_text is None) == (query_vector is None):
            raise ValueError("질문 또는 벡터 중 하나만 지정하세요.")
        if query_vector is None:
            query_vector = (embedder or Embedder(self.root)).encode([query_text])[0]
        return {s: self.search(s, p, top_k, query_vector=query_vector) for s in E2_STORES}


def main():
    """질문 또는 기존 문서 벡터로 검색을 실행한다."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--work-id", required=True)
    p.add_argument("--store", choices=(*STORES, "e2rag"), default="e2rag")
    p.add_argument("--p", type=int, required=True)
    p.add_argument("--top-k", type=int, default=10)
    q = p.add_mutually_exclusive_group(required=True)
    q.add_argument("--query")
    q.add_argument("--self-id", help="API 없이 저장된 문서 벡터로 검색하는 구조 점검")
    args = p.parse_args()
    index = VectorIndex(args.project_root, args.work_id)
    kwargs = {"query_text": args.query} if args.query is not None else {}
    if args.self_id:
        matches = [(s, i) for s in STORES for i, r in enumerate(index.meta[s])
                   if str(r["node_id"]) == args.self_id]
        if len(matches) != 1:
            raise ValueError("self-id가 없거나 여러 저장소에서 중복됩니다.")
        s, i = matches[0]
        kwargs["query_vector"] = index.vectors[s][i]
    result = (index.search_e2rag(args.p, top_k=args.top_k, **kwargs) if args.store == "e2rag"
              else index.search(args.store, args.p, args.top_k, **kwargs))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
