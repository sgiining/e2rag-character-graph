"""제안 시스템: p 사전 필터와 Relation Observation 검색·확장."""
from retrieval import graph_retrieve


def retrieve(index, graph, question, cues, embedder, cfg, p):
    """현재 공개 범위 안에서만 시드 검색과 one-hop을 수행한다."""
    result = graph_retrieve(index, graph, p, cues, embedder, cfg,
                            stores=('entity', 'event', 'relation'), proposed=True)
    result['scope'] = 'read_prefix'
    return result
