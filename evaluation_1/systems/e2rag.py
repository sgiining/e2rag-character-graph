"""기본 E²RAG 기준선: Entity/Event 시드와 Event 중심 one-hop. Relation 제외."""
from retrieval import graph_retrieve


def retrieve(index, graph, question, cues, embedder, cfg):
    """독자 p 없이 전체 작품에서 검색한다. 가상 답변은 사용하지 않는다."""
    last = len(index.manifest['read_end_char'])
    result = graph_retrieve(index, graph, last, cues, embedder, cfg,
                            stores=('entity', 'event'), proposed=False)
    result['scope'] = 'full_story'
    return result
