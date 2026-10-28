"""문항별 답변·평가자 점수와 유형별/전체 지표를 CSV·HTML로 내보낸다."""
from __future__ import annotations
import csv
import html
from collections import defaultdict
from statistics import mean
from common import SYSTEMS, TYPES, save_json, keyed, digest

LABELS = {'answerable': '일반', 'relation_change': '관계 변화', 'unanswerable_future': '미래'}


def csv_write(path, rows, fields):
    """Excel용 UTF-8 BOM CSV를 생성한다. 답변 문자열을 변경하지 않는다."""
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def majority(values):
    """세 평가자가 모두 완료된 0/1 판정만 다수결 집계한다."""
    if len(values) != 3 or any(type(v) is not int or v not in (0, 1) for v in values):
        raise ValueError('다수결은 완료된 평가자 3명의 0/1 판정 필요')
    return int(sum(values) >= 2)


def combine(row, system, answer, judges):
    """문항별 점수 평균과 다수결 및 평가자별 결합 유보 성공을 계산한다."""
    future = row['qa_type'] == 'unanswerable_future'
    return {**{f: row.get(f) for f in ('qa_id', 'work_id', 'p', 'qa_type', 'pair_group', 'question', 'ground_truth')},
            'system': system, 'answer': answer['answer'],
            'mean_score': None if future else mean(j['score'] for j in judges),
            'correct': None if future else majority([j['correct'] for j in judges]),
            'abstained': majority([j['abstained'] for j in judges]),
            'spoiler': majority([j['spoiler'] for j in judges]),
            'abstention_success': majority([int(j['abstained'] == 1 and j['spoiler'] == 0) for j in judges]) if future else None,
            'context_tokens': answer['context_tokens'],
            'future_chunk_ratio': answer['future_chunk_ratio']}


def rate(rows, key):
    """분모가 없는 비율은 0 대신 null로 표현한다."""
    values = [r[key] for r in rows if r[key] is not None]
    return mean(values) if values else None


def summarize(rows, expected_pairs):
    """전체 및 유형/작품 부분집합 지표를 계산한다. 미래 점수는 평균에서 제외한다."""
    scored = [r for r in rows if r['mean_score'] is not None]
    future = [r for r in rows if r['qa_type'] == 'unanswerable_future']
    pairmap = defaultdict(list)
    for row in rows:
        if row['qa_type'] == 'relation_change':
            pairmap[(row['work_id'], row['pair_group'])].append(row)
    complete = [items for key, items in pairmap.items() if
                len(items) == 2 and {r['qa_id'] for r in items} == expected_pairs[key]]
    success_pairs = sum(all(r['correct'] == 1 for r in items) for items in complete)
    return {'items': len(rows), 'scored_items': len(scored),
            'mean_score': mean(r['mean_score'] for r in scored) if scored else None,
            'correct_count': sum(r['correct'] for r in scored),
            'accuracy': rate(scored, 'correct'),
            'pair_count': len(complete), 'correct_pair_count': success_pairs,
            'pair_accuracy': success_pairs/len(complete) if complete else None,
            'future_items': len(future), 'abstention_success_count': sum(r['abstention_success'] for r in future),
            'abstention_success_rate': rate(future, 'abstention_success'),
            'spoiler_count': sum(r['spoiler'] for r in rows), 'spoiler_rate': rate(rows, 'spoiler'),
            'over_abstention_count': sum(r['abstained'] for r in scored),
            'over_abstention_rate': rate(scored, 'abstained'),
            'mean_context_tokens': rate(rows, 'context_tokens'),
            'mean_future_chunk_ratio': rate(rows, 'future_chunk_ratio')}


def export(root, run, dataset, cfg, fingerprint, allow_partial=False):
    """불완전 결과는 기본 중단한다. 명시한 부분 집계는 별도 폴더에 저장한다."""
    models = [j['model'] for j in cfg['judges']]
    answers = {s: keyed(run / 'answers' / s / 'answers.jsonl', ('qa_id',)) for s in SYSTEMS}
    judgments = {m: keyed(run / 'judges' / m / 'judgments.jsonl', ('qa_id',)) for m in models}
    long_rows, wide_rows, final_rows, missing = [], [], [], []
    judge_stats = []
    for row in dataset:
        qid = row['qa_id']
        a = {s: answers[s].get((qid,)) for s in SYSTEMS}
        j = {m: judgments[m].get((qid,)) for m in models}
        if any(x is None for x in a.values()) or any(x is None for x in j.values()):
            missing.append(qid)
            continue
        hashes = {s: digest(a[s]) for s in SYSTEMS}
        for m in models:
            if j[m]['answer_hashes'] != hashes or j[m]['run_fingerprint'] != fingerprint:
                raise ValueError('채점 대상 답변/실험 설정 변경: ' + qid)
        wide = {f: row.get(f) for f in ('qa_id', 'work_id', 'p', 'qa_type', 'pair_group', 'question', 'ground_truth')}
        for system in SYSTEMS:
            scores = []
            for model in models:
                verdict = j[model]['verdicts'][system]
                scores.append(verdict)
                long_rows.append({**{f: row.get(f) for f in ('qa_id','work_id','p','qa_type','question','ground_truth')},
                                  'system': system, 'answer': a[system]['answer'], 'judge': model,
                                  **{f: verdict[f] for f in ('score','correct','abstained','spoiler','reason')}})
                for field in ('score', 'correct', 'abstained', 'spoiler', 'reason'):
                    wide[f'{system}|{model}|{field}'] = verdict[field]
            final = combine(row, system, a[system], scores)
            final_rows.append(final)
            for field in ('answer','mean_score','correct','abstained','spoiler','abstention_success'):
                wide[f'{system}|{field}'] = final[field]
        wide_rows.append(wide)
    if missing and not allow_partial:
        raise ValueError(f'미완료 {len(missing)}/{len(dataset)} 문항. 전체 평균은 아직 계산하지 않습니다. '
                         '진행 중 확인은 report --allow-partial 사용')
    out = run / ('reports_partial' if missing else 'reports')
    out.mkdir(parents=True, exist_ok=True)
    pairs = defaultdict(set)
    for row in dataset:
        if row['qa_type'] == 'relation_change':
            pairs[(row['work_id'], row['pair_group'])].add(row['qa_id'])
    metrics = []
    for system in SYSTEMS:
        selected = [r for r in final_rows if r['system'] == system]
        groups = [('전체', '전체', selected)]
        groups += [('유형', LABELS[t], [r for r in selected if r['qa_type'] == t]) for t in TYPES]
        groups += [('작품', w, [r for r in selected if r['work_id'] == w]) for w in sorted({r['work_id'] for r in dataset})]
        for scope, name, group in groups:
            metrics.append({'system': system, 'scope': scope, 'group': name, **summarize(group, pairs)})
        for model in models:
            for kind in ('all', *TYPES):
                chosen = [r for r in long_rows if r['system'] == system and r['judge'] == model
                          and (kind == 'all' or r['qa_type'] == kind)]
                scored = [r['score'] for r in chosen if r['score'] is not None]
                judge_stats.append({'system':system, 'judge':model, 'qa_type':kind,
                                    'items':len(chosen), 'scored_items':len(scored),
                                    'mean_score':mean(scored) if scored else None})
    agreement = []
    for system in SYSTEMS:
        for field in ('correct', 'abstained', 'spoiler'):
            all_same, count = 0, 0
            for qid in [r['qa_id'] for r in wide_rows]:
                vals = [judgments[m][(qid,)]['verdicts'][system][field] for m in models]
                if vals[0] is None:
                    continue
                count += 1
                all_same += len(set(vals)) == 1
            agreement.append({'system':system, 'field':field, 'items':count,
                              'unanimous_count':all_same, 'unanimous_rate':all_same/count if count else None})
    save_json(out / 'summary.json', {'status':'PARTIAL' if missing else 'COMPLETE',
        'run_fingerprint':fingerprint, 'expected_items':len(dataset), 'completed_items':len(wide_rows),
        'missing_qa_ids':missing, 'score_denominator':'answerable + relation_change only',
        'metrics':metrics, 'judge_scores':judge_stats, 'agreement':agreement})
    long_fields = ['qa_id','work_id','p','qa_type','question','ground_truth','system','answer','judge',
                   'score','correct','abstained','spoiler','reason']
    csv_write(out/'questions_long.csv', long_rows, long_fields)
    base = ['qa_id','work_id','p','qa_type','pair_group','question','ground_truth']
    wide_fields = base + [f'{s}|{f}' for s in SYSTEMS for f in ('answer','mean_score','correct','abstained','spoiler','abstention_success')]
    wide_fields += [f'{s}|{m}|{f}' for s in SYSTEMS for m in models for f in ('score','correct','abstained','spoiler','reason')]
    csv_write(out/'questions_wide.csv', wide_rows, wide_fields)
    csv_write(out/'summary.csv', metrics, list(metrics[0]))
    csv_write(out/'judge_scores.csv', judge_stats, list(judge_stats[0]))
    for kind in TYPES:
        csv_write(out/f'{kind}.csv', [r for r in long_rows if r['qa_type']==kind], long_fields)
    esc = lambda x: html.escape(str(x))
    def fmt(value):
        return '—' if value is None else (f'{value:.4f}' if isinstance(value,float) else esc(value))
    headers = ['system','scope','group','items','scored_items','mean_score','accuracy','pair_accuracy',
               'abstention_success_rate','spoiler_rate']
    table = '<table><tr>'+''.join('<th>'+esc(k)+'</th>' for k in headers)+'</tr>'
    table += ''.join('<tr>'+''.join('<td>'+fmt(m[k])+'</td>' for k in headers)+'</tr>' for m in metrics)+'</table>'
    cards = []
    finals = {(r['qa_id'],r['system']):r for r in final_rows}
    for row in dataset:
        qid = row['qa_id']
        if (qid,SYSTEMS[0]) not in finals:
            continue
        block = f'<article data-kind="{esc(row["qa_type"])}"><h3>{esc(qid)} · {esc(row["work_id"])} · p={row["p"]} · {LABELS[row["qa_type"]]}</h3>'
        block += '<p>'+esc(row['question'])+'</p><details><summary>기준 정답</summary><p>'+esc(row['ground_truth'])+'</p></details>'
        for system in SYSTEMS:
            final = finals[(qid,system)]
            block += f'<section><h4>{system} · 평균 {fmt(final["mean_score"])} · 정답 {fmt(final["correct"])} · 유보 {final["abstained"]} · 스포일러 {final["spoiler"]}</h4><p>{esc(final["answer"])}</p>'
            for model in models:
                v = judgments[model][(qid,)]['verdicts'][system]
                block += f'<details><summary>{esc(model)} · 점수 {fmt(v["score"])} · 정답 {fmt(v["correct"])} · 유보 {v["abstained"]} · 스포일러 {v["spoiler"]}</summary><p>{esc(v["reason"])}</p></details>'
            block += '</section>'
        cards.append(block+'</article>')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>진행도 QA 비교 평가</title>
<style>body{font:16px/1.65 system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#182331;background:#f6f8fa}table{border-collapse:collapse;background:white;font-size:13px;display:block;overflow:auto}th,td{padding:9px;border:1px solid #ddd}article{background:white;border:1px solid #ddd;border-radius:10px;margin:22px 0;padding:20px}section{border-top:1px solid #ddd;padding:10px 0}p{white-space:pre-wrap}summary,select{cursor:pointer}h1,h3{color:#184b64}</style>
<h1>독서 진행도 기반 QA 비교 평가</h1>'''
    page += f'<p>상태: {"부분 집계" if missing else "전체 완료"} · {len(wide_rows)}/{len(dataset)}문항. 전체 평균은 일반·관계 변화만 포함하며 미래 문항은 점수 없음(—). 비율은 0~1.</p>'+table
    page += '<h2>문항별 답변과 평가자 판정</h2><label>유형 <select id="kind"><option value="all">전체</option>'+''.join(f'<option value="{t}">{LABELS[t]}</option>' for t in TYPES)+'</select></label>'
    page += ''.join(cards)+'''<script>document.getElementById('kind').addEventListener('change',e=>{document.querySelectorAll('article').forEach(a=>a.hidden=e.target.value!=='all'&&a.dataset.kind!==e.target.value)})</script></html>'''
    (out/'report.html').write_text(page,encoding='utf-8')
    return {'status':'PARTIAL' if missing else 'COMPLETE','completed_items':len(wide_rows),'folder':str(out)}
