"""독서 진행도에 따른 최신 인물·관계·사건 상태를 표시하는 Streamlit 앱.

- 기존 progress_graph.py에서 만든 full_graph.json을 읽는다.
- 원본 Entity Observation을 병합/수정하지 않는다.
- data/entity_identity/{작품명}.entity_identity_groups.jsonl에서 graph_entity_id를 읽는다.
- observation_id → graph_entity_id 매핑으로 인물 노드를 구성한다.
- 현재 진행도 p까지 등장한 Observation만 그래프 표시용 최신 상태로 접는다.
- 원본의 인물 식별 그룹은 이름 기반 임시 그룹일 수 있으므로 확정된 동일성으로 주장하지 않는다.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from html import escape
from pathlib import Path

import streamlit as st
from pyvis.network import Network

from progress_graph import WORKS, filtered

ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "data" / "progress_graph"


@st.cache_data(show_spinner=False)
def load_json(path: str, mtime_ns: int) -> dict:
    """수정 시각이 바뀌면 캐시를 새로 읽는다."""
    del mtime_ns
    with open(path, encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 객체가 아닙니다: {path}")
    return data


def identity_path(work: str) -> Path:
    """프로젝트의 기존 Entity Identity 그룹 파일 경로를 만든다."""
    return ROOT / "data" / "entity_identity" / f"{work}.entity_identity_groups.jsonl"


@st.cache_data(show_spinner=False)
def read_identity_groups(path: str, mtime_ns: int, work: str) -> tuple[dict[str, str], dict]:
    """Observation ID를 graph_entity_id에 연결하고 매핑의 중복을 검사한다.

    그룹 파일의 전체 내용은 화면에 직접 노출하지 않는다. 이후 진행도 p에
    포함된 Observation ID만 조회하므로 미래 노드가 미리 등장하지 않는다.
    """
    del mtime_ns
    mapping: dict[str, str] = {}
    seen_graph_ids: set[str] = set()
    status_counts: dict[str, int] = defaultdict(int)

    with Path(path).open(encoding="utf-8-sig") as fp:
        for line_no, line in enumerate(fp, 1):
            if not line.strip():
                continue
            try:
                group = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Entity Identity {line_no}행 JSON 오류: {exc}") from exc
            if not isinstance(group, dict):
                raise ValueError(f"Entity Identity {line_no}행: JSON 객체가 아닙니다.")
            if group.get("work_id") != work:
                raise ValueError(f"Entity Identity {line_no}행: work_id가 다릅니다.")

            graph_id = group.get("graph_entity_id")
            ids = group.get("observation_ids")
            if not isinstance(graph_id, str) or not graph_id.strip():
                raise ValueError(f"Entity Identity {line_no}행: graph_entity_id가 없습니다.")
            if graph_id in seen_graph_ids:
                raise ValueError(f"Entity Identity {line_no}행: 중복 graph_entity_id {graph_id}")
            if not isinstance(ids, list) or not ids:
                raise ValueError(f"Entity Identity {line_no}행: observation_ids가 비었습니다.")
            if group.get("observation_count") != len(ids):
                raise ValueError(f"Entity Identity {line_no}행: observation_count가 불일치합니다.")
            details = group.get("observations")
            if details is not None:
                if not isinstance(details, list) or {x.get("observation_id") for x in details if isinstance(x, dict)} != set(ids):
                    raise ValueError(f"Entity Identity {line_no}행: observations와 observation_ids가 다릅니다.")
            for obs_id in ids:
                if not isinstance(obs_id, str) or not obs_id.strip():
                    raise ValueError(f"Entity Identity {line_no}행: 올바르지 않은 Observation ID")
                if obs_id in mapping:
                    raise ValueError(f"Entity Identity {line_no}행: Observation ID 중복 등록 {obs_id}")
                mapping[obs_id] = graph_id
            seen_graph_ids.add(graph_id)
            status_counts[str(group.get("identity_status") or "unknown")] += 1

    if not mapping:
        raise ValueError("Entity Identity 그룹 파일에 유효한 데이터가 없습니다.")
    # 수량은 전체 작품에 대한 메타데이터이므로 화면에는 노출하지 않는다.
    return mapping, {"statuses": sorted(status_counts)}


def check_identity_coverage(mapping: dict[str, str], full: dict) -> None:
    """그래프 Entity Observation과 Identity 파일의 ID 호환성을 확인한다."""
    entity_ids = {
        node["source_id"]
        for node in full.get("nodes", [])
        if node.get("type") == "entity"
    }
    missing = entity_ids - set(mapping)
    extra = set(mapping) - entity_ids
    if missing or extra:
        raise ValueError(
            "Identity 매핑과 full_graph.json의 Entity Observation ID가 다릅니다. "
            f"누락={len(missing)}건, 그래프에 없는 ID={len(extra)}건. "
            f"첫 누락={sorted(missing)[:3]}, 첫 초과={sorted(extra)[:3]}. "
            "같은 Entity 결과를 기반으로 그룹 파일을 생성했는지 확인하세요."
        )


def name(node: dict) -> str:
    data = node.get("data", {})
    return str(data.get("canonical_name") or node.get("source_id") or "이름 없음")


def latest(nodes: list[dict]) -> list[dict]:
    """최신 τ에 있는 기록만 남긴다. 같은 τ는 우열을 임의로 정하지 않는다."""
    when = max(n["tau"] for n in nodes)
    return sorted((n for n in nodes if n["tau"] == when), key=lambda n: n["source_id"])


def build_snapshot(sub: dict, identity: dict[str, str]) -> dict:
    """p 범위 내에서 한 인물 노드 및 한 인물쌍 최신 관계/사건을 만든다.

    동일한 graph_entity_id를 가진 Observation만 하나의 인물로 표시한다.
    Identity 파일에 없는 ID는 사전에 검증 단계에서 차단한다.
    """
    entities = [node for node in sub["nodes"] if node["type"] == "entity"]
    events = {node["source_id"]: node for node in sub["nodes"] if node["type"] == "event"}
    relations = [node for node in sub["nodes"] if node["type"] == "relation"]
    entity_to_character: dict[str, str] = {}
    members: dict[str, list[dict]] = defaultdict(list)
    for node in entities:
        obs_id = node["source_id"]
        char_id = identity[obs_id]
        entity_to_character[obs_id] = char_id
        members[char_id].append(node)

    characters = {}
    for char_id, observations in members.items():
        current = latest(observations)
        first_chunk = min(n["tau"] for n in observations)
        characters[char_id] = {
            "id": char_id,
            "name": name(current[0]) if len({name(n) for n in current}) == 1 else "/".join(sorted({name(n) for n in current})),
            "first_tau": first_chunk,
            "tau": current[0]["tau"],
            "current": current,
            "observations": sorted(observations, key=lambda n: (n["tau"], n["source_id"])),
        }

    pair_records: dict[tuple[str, str], list[dict]] = defaultdict(list)
    relation_pair: dict[str, tuple[str, str]] = {}
    for relation in relations:
        row = relation["data"]
        a = entity_to_character.get(row.get("source_observation_id"))
        b = entity_to_character.get(row.get("target_observation_id"))
        if a is None or b is None:
            continue
        # 같은 graph_entity_id 사이의 자기 관계는 인물 간 관계 선으로 표시하지 않는다.
        if a == b:
            continue
        pair = tuple(sorted((a, b)))
        pair_records[pair].append(relation)
        relation_pair[relation["source_id"]] = pair

    # 동일 인물쌍에 대해 최신 Relation Observation만 화면에 적용한다.
    pairs = {}
    for pair, observations in pair_records.items():
        current = latest(observations)
        pairs[pair] = {
            "pair": pair,
            "tau": current[0]["tau"],
            "relations": current,
            "events": [],
        }

    # 최신 관계 Observation에 연결된 Stage 3 승인 사건만 표시한다.
    # 이전 관계의 연결 사건이 현재 관계의 근거처럼 남지 않도록 한다.
    current_relations_by_pair = {
        pair: {relation["source_id"] for relation in item["relations"]}
        for pair, item in pairs.items()
    }
    stage3_candidates: dict[tuple[str, str], list[tuple[int, dict, dict]]] = defaultdict(list)
    for edge in sub["edges"]:
        if edge["type"] != "relation_event":
            continue
        rel_key = edge["source"]
        pair = relation_pair.get(rel_key.split(":", 1)[-1]) if rel_key.startswith("relation:") else None
        event_id = edge["data"].get("event_id")
        event = events.get(event_id)
        if (pair is not None and event is not None
                and rel_key.split(":", 1)[-1] in current_relations_by_pair[pair]):
            stage3_candidates[pair].append((edge["tau"], event, edge))
    for pair, candidates in stage3_candidates.items():
        newest = max(tau for tau, _, _ in candidates)
        pairs[pair]["events"] = sorted(
            ((event, link) for tau, event, link in candidates if tau == newest),
            key=lambda pair_item: (pair_item[0]["source_id"], pair_item[1]["id"]),
        )

    # Entity-Event의 사건은 참여 인물 ID 집합별로 최신 기록을 표시한다.
    # 참여자가 2명 이상이어도 이 사실만으로 두 인물 사이의 '관계'를 만들지 않는다.
    event_participants: dict[str, set[str]] = defaultdict(set)
    for edge in sub["edges"]:
        if edge["type"] != "entity_event":
            continue
        row = edge["data"]
        char_id = entity_to_character.get(row.get("source_observation_id"))
        event_id = row.get("target_event_id")
        if char_id and event_id in events:
            event_participants[event_id].add(char_id)

    event_groups: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for event_id, participants in event_participants.items():
        signature = tuple(sorted(participants))
        if signature:
            event_groups[signature].append(events[event_id])

    latest_events = {}
    for signature, observations in event_groups.items():
        current = latest(observations)
        latest_events[signature] = {"participants": signature, "events": current,
                                    "tau": current[0]["tau"]}

    return {"characters": characters, "pairs": pairs,
            "events_by_participants": latest_events,
            "visible_entities": entities}


def safe(value: object, limit: int = 800) -> str:
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value)
    value = str(value if value is not None else "-")
    return escape(value[:limit], quote=True).replace("\n", "<br>")


def short(value: object, length: int = 23) -> str:
    text = str(value if value is not None else "-").replace("\n", " ")
    return text if len(text) <= length else text[:length - 1] + "…"


def info(title: str, value: object) -> str:
    return f"<b>{safe(title)}</b>: {safe(value)}"


def char_tooltip(char: dict) -> str:
    result = [info("인물", char["name"]), info("Graph Entity ID", char["id"]),
              info("최초 등장 τ", char["first_tau"]),
              info("최신 관찰 τ", char["tau"]),
              info("묶인 관찰 수", len(char["observations"]))]
    for node in char["current"]:
        data = node["data"]
        result.extend([info("현재 관찰", node["source_id"]),
                       info("설명", data.get("description")),
                       info("원문 근거", data.get("evidence"))])
    return "<br>".join(result)


def relation_label(item: dict) -> str:
    labels = {
        str(r["data"].get("relation_type_label") or r["data"].get("relation_type") or "관계")
        for r in item["relations"]
    }
    return next(iter(labels)) if len(labels) == 1 else "동시 관찰 관계(복수)"


def relation_tooltip(item: dict) -> str:
    lines = [info("최신 관계 τ", item["tau"]), info("관계 유형", relation_label(item))]
    for rel in item["relations"]:
        lines.extend([info("설명", rel["data"].get("description")),
                      info("원문 근거", rel["data"].get("evidence"))])
    for event, link in item["events"]:
        lines.extend([info("해당 인물쌍 최신 승인 사건", event["data"].get("label")),
                      info("사건 근거", event["data"].get("evidence")),
                      info("연결 유형", link["data"].get("relation_event_link_type"))])
    return "<br>".join(lines)


def make_net() -> Network:
    net = Network(height="620px", width="100%", bgcolor="#ffffff",
                  font_color="#222222", directed=True, cdn_resources="in_line")
    net.barnes_hut(gravity=-3200, central_gravity=0.22, spring_length=170,
                   spring_strength=0.035, damping=0.42, overlap=0.3)
    return net


def pair_id(pair: tuple[str, str]) -> str:
    return hashlib.sha256(json.dumps(pair, ensure_ascii=False).encode()).hexdigest()[:16]


def add_character(net: Network, char: dict) -> None:
    net.add_node("char:" + char["id"], label=short(char["name"]),
                 title=char_tooltip(char), color="#A6CEF0", shape="dot", size=24)


def render_network(snapshot: dict, mode: str, center: str | None,
                   include_single_events: bool, show_all: bool) -> tuple[Network, int, int]:
    """최신 상태 스냅샷만 PyVis로 넘긴다. 과거 노드/선은 렌더링하지 않는다."""
    net = make_net()
    chars = snapshot["characters"]
    pairs = snapshot["pairs"]
    chosen_pairs = {
        k: v for k, v in pairs.items()
        if center is None or center in k
    }
    chosen_events = {
        k: v for k, v in snapshot["events_by_participants"].items()
        if center is None or center in k
    } if mode == "통합 그래프" and include_single_events else {}

    chosen_chars = set(chars) if show_all and center is None else set()
    for pair in chosen_pairs:
        chosen_chars.update(pair)
    for signature in chosen_events:
        chosen_chars.update(signature)
    if center:
        chosen_chars.add(center)
    for char_id in sorted(chosen_chars):
        add_character(net, chars[char_id])

    edge_count = 0
    displayed_events: set[str] = set()

    def add_event(event: dict) -> str:
        """하나의 원본 Event는 화면에도 한 번만 노드로 추가한다."""
        event_key = "event:" + event["source_id"]
        if event_key not in displayed_events:
            data = event["data"]
            net.add_node(
                event_key, label=short(data.get("label") or "사건", 19),
                title="<br>".join([info("사건", data.get("label")),
                                    info("최신 공개 τ", event["tau"]),
                                    info("내용", data.get("summary") or data.get("description")),
                                    info("원문 근거", data.get("evidence"))]),
                color="#F5D39E", shape="box",
            )
            displayed_events.add(event_key)
        return event_key

    for pair, item in chosen_pairs.items():
        a, b = pair
        key = pair_id(pair)
        label = relation_label(item)
        if mode == "인물 관계도":
            # 두 인물 사이 최신 관계 Observation은 선 하나로 표시한다.
            row = item["relations"][0]["data"]
            src = "char:" + a
            dst = "char:" + b
            # 원본 방향을 보존할 수 있는 경우에만 방향을 표시한다.
            source_char = next((c for c in (a, b)
                                if any(n["source_id"] == row.get("source_observation_id")
                                       for n in chars[c]["observations"])), None)
            target_char = next((c for c in (a, b)
                                if any(n["source_id"] == row.get("target_observation_id")
                                       for n in chars[c]["observations"])), None)
            if source_char and target_char:
                src, dst = "char:" + source_char, "char:" + target_char
            sym = row.get("is_symmetric") is True or row.get("direction") in (
                "undirected", "symmetric", "bidirectional")
            net.add_edge(src, dst, label=short(label, 18), title=relation_tooltip(item),
                         arrows="" if sym or len(item["relations"]) > 1 else "to",
                         color="#7970AB", width=2.5)
            edge_count += 1
        else:
            rel_key = "pair_relation:" + key
            net.add_node(rel_key, label=short(label, 17), title=relation_tooltip(item),
                         color="#CCBCE9", shape="diamond", size=19)
            net.add_edge("char:" + a, rel_key, color="#8874B7", arrows="")
            net.add_edge(rel_key, "char:" + b, color="#8874B7", arrows="")
            edge_count += 2
            for event, _link in item["events"]:
                event_key = add_event(event)
                net.add_edge(rel_key, event_key, color="#D47575", arrows="to")
                edge_count += 1

    for signature, item in chosen_events.items():
        for event in item["events"]:
            event_key = add_event(event)
            for char_id in signature:
                net.add_edge("char:" + char_id, event_key, color="#8EA9BC", arrows="")
                edge_count += 1

    return net, len(net.nodes), edge_count



def relation_details(snapshot: dict, center: str | None) -> None:
    pairs = snapshot["pairs"]
    chars = snapshot["characters"]
    options = [pair for pair in pairs if center is None or center in pair]
    options.sort(key=lambda pair: (pairs[pair]["tau"], chars[pair[0]]["name"], chars[pair[1]]["name"]))
    if not options:
        st.info("현재 범위에 확인된 관계가 없습니다.")
        return
    pair = st.selectbox("현재 관계 선택", options,
                        format_func=lambda p: f"{chars[p[0]]['name']} ↔ {chars[p[1]]['name']} | {relation_label(pairs[p])}")
    item = pairs[pair]
    st.write(f"**현재 관계:** {relation_label(item)}")
    st.caption(f"최신 관계 Observation의 τ={item['tau']} · 같은 Chunk의 동시 관찰은 함께 표시")
    for relation in item["relations"]:
        data = relation["data"]
        st.write(f"**관계 설명:** {data.get('description') or '-'}")
        st.write(f"**원문 근거:** {data.get('evidence') or '-'}")
        st.caption(f"관계 Observation ID: {relation['source_id']}")
    if item["events"]:
        st.markdown("**해당 인물쌍의 최신 Stage 3 승인 사건**")
        for event, link in item["events"]:
            d = event["data"]
            st.write(f"**사건:** {d.get('label') or '-'}")
            st.write(f"**사건 내용:** {d.get('summary') or d.get('description') or '-'}")
            st.write(f"**사건 근거:** {d.get('evidence') or '-'}")
            st.caption(f"연결 유형: {link['data'].get('relation_event_link_type') or '-'}")
    else:
        st.caption("현재 인물쌍에 연결된 승인 사건이 없습니다.")


def main() -> None:
    st.set_page_config(page_title="독서 진행도별 인물 관계", layout="wide")
    st.title("독서 진행도별 동적 인물 관계도")
    st.caption("화면에는 진행도 p까지 공개된 최신 인물·관계·사건 상태만 표시합니다.")
    work = st.sidebar.selectbox("작품", WORKS)
    folder = RESULT_ROOT / work
    report_file = folder / "validation_report.json"
    full_file = folder / "full_graph.json"
    if not report_file.is_file() or not full_file.is_file():
        st.error("그래프 결과가 없습니다. progress_graph.py로 먼저 생성하세요.")
        st.code(f'python src/chae/progress_graph.py --work-id "{work}" --p 3', language="powershell")
        st.stop()
    try:
        report = load_json(str(report_file), report_file.stat().st_mtime_ns)
        if report.get("status") != "OK":
            st.error("입력 데이터 검증에 실패했습니다. validation_report.json을 확인하세요.")
            st.json({"status": report.get("status"), "fatal_error": report.get("fatal_error"),
                     "fatal_count": report.get("fatal_count")})
            st.stop()
        full = load_json(str(full_file), full_file.stat().st_mtime_ns)
        if full.get("work_id") != work or full.get("source_fingerprints") != report.get("input_manifest"):
            raise ValueError("그래프와 보고서가 일치하지 않습니다. 그래프를 재생성하세요.")
        last = int(full["last_chunk"])
        if last < 1:
            raise ValueError("올바르지 않은 마지막 Chunk 번호")
        p = st.sidebar.slider("독서 진행도 p", 1, last, 1) if last > 1 else 1
        sub = filtered(full, p)
        identity_file = identity_path(work)
        if not identity_file.is_file():
            raise FileNotFoundError(
                f"Entity Identity 그룹 파일이 없습니다: {identity_file}"
            )
        mapping, identity_info = read_identity_groups(
            str(identity_file), identity_file.stat().st_mtime_ns, work
        )
        check_identity_coverage(mapping, full)
        snapshot = build_snapshot(sub, mapping)
    except (OSError, ValueError, KeyError, TypeError, AssertionError, json.JSONDecodeError) as exc:
        st.error(f"데이터 읽기/변환 실패: {exc}")
        st.stop()

    st.write(f"**현재 진행도: Chunk 1~{p} / {last}**")
    counts = st.columns(3)
    counts[0].metric("화면 인물 노드", len(snapshot["characters"]))
    counts[1].metric("현재 관계 (인물쌍)", len(snapshot["pairs"]))
    counts[2].metric("현재 Entity-Event 사건 그룹", len(snapshot["events_by_participants"]))

    st.caption("인물 노드는 data/entity_identity의 graph_entity_id로 통합합니다. "
               "원본 Observation ID는 보존됩니다.")
    if "name_based_unverified" in identity_info["statuses"]:
        st.warning("현재 Entity Identity 그룹에는 이름 기반의 미검증 동일성 연결이 포함되어 있습니다. "
                   "동명이인 또는 다른 개체가 합쳐질 가능성이 있으므로 인물 동일성 검토가 필요합니다.")

    st.sidebar.caption(f"Identity 그룹 파일: {identity_file.name}")
    mode = st.sidebar.radio("시각화 유형", ["인물 관계도", "통합 그래프"])
    center = None
    if snapshot["characters"]:
        opts = ["전체"] + sorted(snapshot["characters"],
                              key=lambda k: (snapshot["characters"][k]["name"], k))
        selection = st.sidebar.selectbox("중심 인물 (기본 1-hop)", opts,
                                         format_func=lambda k: "전체" if k == "전체" else snapshot["characters"][k]["name"])
        center = None if selection == "전체" else selection
    show_all = st.sidebar.checkbox("관계 없는 인물도 표시", value=False)
    show_events = mode == "통합 그래프" and st.sidebar.checkbox("Entity–Event의 별도 최신 사건 표시", value=False)
    net, n_nodes, n_edges = render_network(snapshot, mode, center, show_events, show_all)
    if n_nodes:
        st.components.v1.html(net.generate_html(), height=650, scrolling=True)
        st.caption(f"현재 화면: 노드 {n_nodes}개 · 연결 {n_edges}개. 인물의 Chunk 번호는 노드에 표시하지 않습니다.")
    else:
        st.info("현재 조건에 표시할 노드가 없습니다. '관계 없는 인물도 표시'를 선택해 보세요.")
    st.divider()
    st.subheader("현재 관계와 사건 원문 근거")
    relation_details(snapshot, center)
    st.caption("최신 상태 갱신은 화면에만 적용합니다. Entity Identity·Observation·Stage 3 원본은 변경하지 않습니다.")


if __name__ == "__main__":
    main()
