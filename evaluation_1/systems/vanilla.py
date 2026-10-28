"""Vanilla RAG: 질문만으로 작품 전체 chunk를 검색한다."""
def retrieve(index, graph, question, cues, embedder, cfg):
    """독자 p를 받지 않는다. 마지막 chunk는 전체 인덱스 선택용 상한이다."""
    last = len(index.manifest['read_end_char'])
    results = index.search('chunk', last, top_k=cfg['passage_k'],
                           query_text=question, embedder=embedder)
    return {'passages': [{'chunk_id': int(r['chunk_id']), 'tau': r['tau'],
                          'text': r['evidence'], 'score': r['score'],
                          'original_span': r['evidence_original_span']} for r in results],
            'nodes': [], 'edges': [], 'seed_counts': {'chunk': len(results)},
            'search_bound': last, 'scope': 'full_story'}
