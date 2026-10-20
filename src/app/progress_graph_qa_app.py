"""기존 인물 관계도와 별도 구현한 QA 엔진을 한 화면에서 실행한다.

기존 progress_graph_app.py / progress_graph.py 파일은 변경하지 않는다.
소설과 독서 진행도 p를 두 기능에서 공유한다.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GRAPH_DIR = ROOT / "src" / "graph"
if str(GRAPH_DIR) not in sys.path:
    sys.path.insert(0, str(GRAPH_DIR))

from src.graph import progress_graph_app as graph_ui
from src.graph.progress_graph import WORKS, filtered
from src.chat.progress_qa import ProgressQA


@st.cache_resource(show_spinner=False)
def get_qa_engine(work: str) -> ProgressQA:
    """작품별 인덱스·그래프 로딩을 한 번만 수행한다."""
    return ProgressQA(work, ROOT)


def qa_section(work: str, p: int) -> None:
    """그래프에 적용된 독서 진행도를 그대로 QA에 전달한다."""
    st.divider()
    st.subheader("진행도 기반 QA 챗봇")
    st.caption(f"{work} · 완료한 범위: Chunk 1~{p} · 이후 전개는 검색하지 않습니다.")

    # 진행도를 바꾸더라도 이전 진행도의 답변이 새 화면에 노출되지 않도록 분리한다.
    history_key = (work, p)
    if "qa_histories" not in st.session_state:
        st.session_state.qa_histories = {}
    history = st.session_state.qa_histories.setdefault(history_key, [])

    with st.expander("검색 설정", expanded=False):
        top_k = st.slider("저장소별 초기 검색 개수", 1, 10, 5, key="qa_top_k")
        passage_k = st.slider("사용할 원문 Chunk 수", 1, 8, 5, key="qa_passage_k")
        use_cues = st.checkbox("질문에서 Entity·Event·Relation 단서 추출", value=True,
                               key="qa_use_cues")
        st.caption("기본 그래프 확장은 one-hop입니다. 답변 생성에는 OpenAI API 비용이 발생합니다.")

    for msg in history:
        with st.chat_message("user"):
            st.write(msg["question"])
        with st.chat_message("assistant"):
            st.write(msg["answer"])
            if msg["passages"]:
                with st.expander("답변 검색에 사용된 원문 근거"):
                    for passage in msg["passages"]:
                        st.markdown(f"**Chunk {passage['chunk_id']}**")
                        st.write(passage["text"])

    question = st.chat_input("읽은 범위의 인물·사건·관계를 질문하세요")
    if question:
        with st.chat_message("user"):
            st.write(question)
        try:
            with st.spinner("읽은 범위에서 근거를 검색하고 답변을 생성하는 중입니다..."):
                engine = get_qa_engine(work)
                result = engine.ask(p, question, top_k=top_k,
                                    passage_k=passage_k, use_cues=use_cues)
        except Exception as exc:
            # API 오류/원본 버전 불일치는 추측한 답변으로 대체하지 않는다.
            st.error(f"QA 실행 실패: {exc}")
            return
        history.append({"question": question, "answer": result["answer"],
                        "passages": result["passages"]})
        with st.chat_message("assistant"):
            st.write(result["answer"])
            if result["passages"]:
                with st.expander("답변 검색에 사용된 원문 근거"):
                    for passage in result["passages"]:
                        st.markdown(f"**Chunk {passage['chunk_id']}**")
                        st.write(passage["text"])


def main() -> None:
    """기존 graph 앱의 시각화 함수를 재사용하고 QA 패널만 추가한다."""
    st.set_page_config(page_title="고전소설 인물 관계도 + QA", layout="wide")
    st.title("독서 진행도별 동적 인물 관계도 및 QA")
    st.caption("인물 관계도와 질의응답에 동일한 작품과 독서 진행도 p를 적용합니다.")

    work = st.sidebar.selectbox("작품", WORKS)
    folder = graph_ui.RESULT_ROOT / work
    report_file = folder / "validation_report.json"
    full_file = folder / "full_graph.json"
    if not report_file.is_file() or not full_file.is_file():
        st.error("그래프 결과가 없습니다. 먼저 progress_graph.py를 실행하세요.")
        st.code(f'python src/graph/progress_graph.py --work-id "{work}" --p 3',
                language="powershell")
        st.stop()

    try:
        report = graph_ui.load_json(str(report_file), report_file.stat().st_mtime_ns)
        if report.get("status") != "OK":
            raise ValueError("그래프 검증 보고서가 OK가 아닙니다.")
        full = graph_ui.load_json(str(full_file), full_file.stat().st_mtime_ns)
        if full.get("work_id") != work or full.get("source_fingerprints") != report.get("input_manifest"):
            raise ValueError("그래프 및 검증 보고서의 원본 정보가 서로 다릅니다.")
        last = int(full["last_chunk"])
        if last < 1:
            raise ValueError("유효하지 않은 마지막 Chunk 번호")
        p = st.sidebar.slider("독서 진행도 p", 1, last, 1) if last > 1 else 1
        sub = filtered(full, p)
        identity_file = graph_ui.identity_path(work)
        if not identity_file.is_file():
            raise FileNotFoundError(f"Entity Identity 파일이 없습니다: {identity_file}")
        mapping, identity_info = graph_ui.read_identity_groups(
            str(identity_file), identity_file.stat().st_mtime_ns, work
        )
        graph_ui.check_identity_coverage(mapping, full)
        snapshot = graph_ui.build_snapshot(sub, mapping)
    except (OSError, ValueError, KeyError, TypeError, AssertionError, json.JSONDecodeError) as exc:
        st.error(f"관계도 데이터 읽기 실패: {exc}")
        st.stop()

    st.write(f"**현재 진행도: Chunk 1~{p} / {last}**")
    counts = st.columns(3)
    counts[0].metric("화면 인물 노드", len(snapshot["characters"]))
    counts[1].metric("현재 관계 (인물쌍)", len(snapshot["pairs"]))
    counts[2].metric("현재 Entity-Event 사건 그룹", len(snapshot["events_by_participants"]))
    st.caption("인물 노드에는 Identity 그룹의 graph_entity_id를 사용합니다. 원본 Observation은 보존됩니다.")
    if "name_based_unverified" in identity_info["statuses"]:
        st.warning("이름에 따라 묶인 미검증 Entity Identity가 있어 동명이인 결합 가능성이 있습니다.")

    st.sidebar.caption(f"Identity 그룹 파일: {identity_file.name}")
    mode = st.sidebar.radio("시각화 유형", ["인물 관계도", "통합 그래프"])
    center = None
    if snapshot["characters"]:
        opts = ["전체"] + sorted(snapshot["characters"],
                                key=lambda k: (snapshot["characters"][k]["name"], k))
        selection = st.sidebar.selectbox(
            "중심 인물 (기본 1-hop)", opts,
            format_func=lambda k: "전체" if k == "전체" else snapshot["characters"][k]["name"]
        )
        center = None if selection == "전체" else selection
    show_all = st.sidebar.checkbox("관계 없는 인물도 표시", value=False)
    show_events = mode == "통합 그래프" and st.sidebar.checkbox(
        "Entity–Event의 별도 최신 사건 표시", value=False
    )
    net, n_nodes, n_edges = graph_ui.render_network(snapshot, mode, center, show_events, show_all)
    if n_nodes:
        st.components.v1.html(net.generate_html(), height=650, scrolling=True)
        st.caption(f"현재 화면: 노드 {n_nodes}개 · 연결 {n_edges}개")
    else:
        st.info("표시할 노드가 없습니다. '관계 없는 인물도 표시'를 선택해 보세요.")

    st.divider()
    st.subheader("현재 관계와 사건 원문 근거")
    graph_ui.relation_details(snapshot, center)
    st.caption("그래프는 화면에만 최신 상태를 반영합니다. 원본 추출 결과는 변경하지 않습니다.")

    qa_section(work, p)


if __name__ == "__main__":
    main()
