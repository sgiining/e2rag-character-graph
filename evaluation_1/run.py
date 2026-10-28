"""② 방식: check → prepare → generate → judge → report."""
from __future__ import annotations
import argparse
from collections import Counter
from pathlib import Path
import importlib
import re
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
from common import (API,Embedder,SYSTEMS,RunLock,config_load,digest,dumps,read_json,read_jsonl,
                    save_json,save_rows,keyed,load_dataset,failure_record,permanent_error)
from prepare import Assets,source_bundle,prepare_assets,input_signature
from retrieval import cues_for,generate,pack_context,assert_prefix,tokens
from judging import policy_for,judgment_schema,validate_judgment,make_payload,JUDGE_PROTOCOL
from common import validate_answer_records,validate_saved
from reporting import export


def name_only(value):
    if not value or Path(value).name!=value or value in ('.','..') or any(c in value for c in '/\\:'):
        raise ValueError('폴더 이름만 사용: '+value)
    return value


def dataset_fingerprint(root,rows,docs):
    """QA·원문·chunk를 고정한다. 코드 경로나 타임스탬프는 데이터 해시에 넣지 않는다."""
    paths=[Path(root)/'data/qa/final'/n for n in ('qa.jsonl','qa_inputs.jsonl','summary.json')]
    paths += [Path(root)/'data/chunk'/f'{w}.jsonl' for w in sorted(docs)]
    return digest({'files':{str(p.relative_to(root)):digest(p.read_bytes()) for p in paths},
                   'texts':{w:digest(t) for w,t in docs.items()}})


def code_hashes():
    """신규 실행에 영향을 주는 코드를 기록한다. 채점만 변경한 이력은 별도로 허용한다."""
    return {p.relative_to(HERE).as_posix():digest(p.read_bytes()) for p in sorted(HERE.rglob('*.py'))
            if not any(x in p.relative_to(HERE).parts for x in ('runs','assets','__pycache__'))}


def bind_new(run,base,cfg,data_hash,works):
    """새 검색 실험의 입력과 생성 코드는 고정하고 채점 수정 이력은 보존한다."""
    codes=code_hashes()
    generation={k:v for k,v in codes.items() if k in ('run.py','retrieval.py','prepare.py','common.py') or k.startswith('systems/')}
    asset_hashes={w:digest((base/w/'manifest.json').read_bytes()) for w in works}
    manifest={'mode':'new_method','config':cfg,'dataset_hash':data_hash,
              'assets':asset_hashes,'assets_name':base.name,'generation_code':generation}
    manifest['run_fingerprint']=digest(manifest)
    path=run/'manifest.json'
    if path.exists():
        if read_json(path)!=manifest:raise ValueError('입력/검색/생성 코드가 바뀜. 새로운 --run 이름으로 실행하세요.')
    else:
        if run.exists() and any(p.name!='.running.lock' for p in run.iterdir()):raise ValueError('manifest 없는 실행 폴더에 결과가 있음')
        save_json(path,manifest)
    history=keyed(run/'code_history.jsonl',('hash',))
    sha=digest(codes)
    if (sha,) not in history:
        history[(sha,)]= {'hash':sha,'code':codes,'judge_protocol':JUDGE_PROTOCOL}
        save_rows(run/'code_history.jsonl',list(history.values()))
    return manifest['run_fingerprint']


def generate_stage(root,run,dataset,selected,engines,cfg,fingerprint,systems,api_factory=API):
    """검색 함수에는 q만 주고 S4에만 p를 준다. QA 정답·유형은 검색/생성 입력에서 배제한다."""
    api=None;embedder=None
    for system in systems:
        path=run/'answers'/system/'answers.jsonl';previous=keyed(path,('qa_id',))
        validate_answer_records(previous,dataset,fingerprint)
        module=importlib.import_module('systems.'+system)
        for row in selected:
            qid=row['qa_id']
            if (qid,) in previous:continue
            try:
                if api is None:api=api_factory(root,run);embedder=Embedder(api,run,cfg)
                index,graph=engines[row['work_id']];p=row['p']
                cues=cues_for(row['question'],api,run,system=='proposed') if system!='vanilla' else {}
                kwargs={'p':p} if system=='proposed' else {}
                context=module.retrieve(index,graph,row['question'],cues,embedder,cfg,**kwargs)
                packed=pack_context(context,cfg['context_tokens'],cfg['graph_context_tokens'])
                if system=='proposed':assert_prefix(context,index,p);assert_prefix(packed,index,p)
                answer,meta,size,prompt_size=generate(row['question'],p,packed,api,cfg,f'answer:{system}:{qid}')
                ids={x['chunk_id'] for x in packed['passages']}
                future={x['chunk_id'] for x in packed['passages'] if x['tau']>p}
                all_evidence=[x for part in ('passages','nodes','edges') for x in packed[part]]
                record={f:row.get(f) for f in ('qa_id','work_id','p','qa_type','pair_group','question')}
                record.update(system=system,answer=answer,run_fingerprint=fingerprint,
                    search_scope=context['scope'],search_bound=context['search_bound'],seed_counts=context['seed_counts'],
                    context=packed,context_tokens=size,estimated_prompt_tokens=prompt_size,
                    future_chunk_count=len(future),chunk_count=len(ids),future_chunk_ratio=len(future)/len(ids) if ids else None,
                    future_evidence_ratio=sum(x['tau']>p for x in all_evidence)/len(all_evidence) if all_evidence else None,
                    context_counts_before={k:len(context[k]) for k in ('passages','nodes','edges')},
                    context_counts_after={k:len(packed[k]) for k in ('passages','nodes','edges')},
                    citation_outside_retrieved=sorted(set(map(int,re.findall(r'\[Chunk\s+(\d+)\]',answer)))-
                        (ids|{x['chunk_id'] for x in packed['nodes']})),**meta)
                previous[(qid,)]=record;save_rows(path,list(previous.values()))
                failure_record(run,'generate',system,qid)
                print(f'[답변] {system} {qid} p={p}',flush=True)
            except Exception as exc:
                failure_record(run,'generate',system,qid,exc)
                print(f'[답변 실패] {system} {qid}: {type(exc).__name__}: {exc}',flush=True)
                if permanent_error(exc):raise


def judge_stage(root,run,dataset,selected,docs,cfg,fingerprint,judge_filter=None,api_factory=API):
    """완료 결과를 먼저 검증하고 없는 평가자·문항만 요청한다. 미래 점수는 모델에게 요청하지 않는다."""
    answers={s:keyed(run/'answers'/s/'answers.jsonl',('qa_id',)) for s in SYSTEMS}
    for records in answers.values():validate_answer_records(records,dataset,fingerprint)
    available=[r for r in selected if all((r['qa_id'],) in answers[s] for s in SYSTEMS)]
    if len(available)<len(selected):print(f'[채점 보류] 답변 미완료 {len(selected)-len(available)}문항',flush=True)
    api=None
    for judge in cfg['judges']:
        model=judge['model']
        if judge_filter and model!=judge_filter:continue
        path=run/'judges'/model/'judgments.jsonl';previous=keyed(path,('qa_id',))
        for row in available:
            qid=row['qa_id'];future=row['qa_type']=='unanswerable_future'
            sources={s:answers[s][(qid,)] for s in SYSTEMS}
            payload,mapping=make_payload(row,sources,docs[row['work_id']],cfg['shuffle_seed'])
            policy=policy_for(future);schema=judgment_schema(future)
            request_hash=digest([policy,schema,payload,mapping,judge,JUDGE_PROTOCOL])
            old=previous.get((qid,))
            if old:
                validate_saved(old,row,sources,fingerprint)
                if old['request_hash']!=request_hash:raise ValueError('채점 정책이 바뀜. 새 코드 설치 시 이전 evaluation_1 전체를 삭제하고 시작하세요.')
                continue
            try:
                if api is None:api=api_factory(root,run)
                opts={k:v for k,v in judge.items() if k not in ('model','max_output_tokens','output_mode')}
                # 4.1-mini는 strict schema. 다른 두 모델은 기존 JSON mode+동일한 엄격한 로컬 검증.
                mode=judge.get('output_mode','json_schema' if model=='gpt-4.1-mini' else 'json_object')
                value,meta=api.chat(model,[{'role':'system','content':policy},
                    {'role':'user','content':dumps(payload)}],f'judge:{model}:{qid}',judge['max_output_tokens'],
                    json_mode=True,options=opts,schema=schema if mode=='json_schema' else None,
                    validate=lambda obj:validate_judgment(obj,future))
                record={'qa_id':qid,'judge':model,'run_fingerprint':fingerprint,'request_hash':request_hash,
                    'answer_hashes':{s:digest(a) for s,a in sources.items()},'anonymous_mapping':mapping,
                    'verdicts':{system:value[label] for label,system in mapping.items()},
                    'judge_protocol':JUDGE_PROTOCOL,'response_schema':schema,'output_mode':mode,**meta}
                validate_saved(record,row,sources,fingerprint)
                previous[(qid,)]=record;save_rows(path,list(previous.values()))
                failure_record(run,'judge',model,qid)
                print(f'[채점] {model} {qid} p={row["p"]}',flush=True)
            except Exception as exc:
                failure_record(run,'judge',model,qid,exc)
                print(f'[채점 실패] {model} {qid}: {type(exc).__name__}: {exc}',flush=True)
                if permanent_error(exc):raise


def validate_outputs(run,rows,docs,cfg,fingerprint,engines=None):
    """완료·누락 수와 저장 판정을 검사한다. 확인된 실패를 감추지 않는다."""
    answers={s:keyed(run/'answers'/s/'answers.jsonl',('qa_id',)) for s in SYSTEMS}
    judges={j['model']:keyed(run/'judges'/j['model']/'judgments.jsonl',('qa_id',)) for j in cfg['judges']}
    known={r['qa_id']:r for r in rows}
    for s,records in answers.items():
        validate_answer_records(records,rows,fingerprint)
        for (qid,),a in records.items():
            if a['system']!=s:raise ValueError('시스템 답변 경로 불일치')
            if engines and s=='proposed':assert_prefix(a['context'],engines[a['work_id']][0],a['p'])
    for model,records in judges.items():
        for (qid,),j in records.items():
            if qid not in known or j['judge']!=model:raise ValueError('채점 QA/모델 경로 불일치')
            if not all((qid,) in answers[s] for s in SYSTEMS):raise ValueError('답변 없이 채점 존재')
            validate_saved(j,known[qid],{s:answers[s][(qid,)] for s in SYSTEMS},fingerprint)
    missing_a={s:[r['qa_id'] for r in rows if (r['qa_id'],) not in v] for s,v in answers.items()}
    missing_j={m:[r['qa_id'] for r in rows if (r['qa_id'],) not in v] for m,v in judges.items()}
    status={'status':'COMPLETE' if not any(missing_a.values()) and not any(missing_j.values()) else 'INCOMPLETE',
        'answer_counts':{s:len(v) for s,v in answers.items()},'judge_counts':{m:len(v) for m,v in judges.items()},
        'missing_answers':missing_a,'missing_judgments':missing_j,
        'future_score_policy':'null excluded from mean; abstention+spoiler separately',
        'mode':'new_method'}
    save_json(run/'status.json',status);return status


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['check','preflight','prepare','generate','judge','report','verify','all'])
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--config',type=Path,default=HERE/'config.json')
    parser.add_argument('--run',default='method_v1')
    parser.add_argument('--assets',default='method_v1')
    parser.add_argument('--judge',choices=['gpt-4.1-mini','gpt-6.1-sol','gpt-5.4-mini'])
    parser.add_argument('--qa-type',choices=['answerable','relation_change','unanswerable_future'])
    parser.add_argument('--work-id');parser.add_argument('--qa-id');parser.add_argument('--limit',type=int,default=0)
    parser.add_argument('--system',choices=['all',*SYSTEMS],default='all')
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args(argv)
    name_only(args.run);name_only(args.assets)
    if args.limit<0:raise ValueError('--limit은 0 이상')
    if args.system!='all' and args.stage!='generate':raise ValueError('--system은 generate 전용')
    if args.judge and args.stage!='judge':raise ValueError('--judge는 judge 전용')
    if args.allow_partial and args.stage!='report':raise ValueError('--allow-partial은 report 전용')
    root=args.project_root.resolve();cfg=config_load(args.config)
    rows,docs=load_dataset(root);works=sorted(docs)
    if Counter(r['qa_type'] for r in rows)!=Counter(answerable=75,relation_change=26,unanswerable_future=33):
        raise ValueError('확정 QA 75/26/33 구성 불일치')
    data_hash=dataset_fingerprint(root,rows,docs)
    selected=[r for r in rows if (not args.work_id or r['work_id']==args.work_id) and
              (not args.qa_id or r['qa_id']==args.qa_id) and (not args.qa_type or r['qa_type']==args.qa_type)]
    if args.limit:selected=selected[:args.limit]
    if not selected:raise ValueError('선택 문항 없음')
    base=HERE/'assets'/args.assets;run=HERE/'runs'/args.run
    if args.stage in ('check','preflight'):
        stats={}
        for work in works:
            manifest,graph,meta,_=source_bundle(root,work)
            stats[work]={'chunks':len(meta['chunk']),'nodes':{s:len(meta[s]) for s in ('entity','event','relation')},
                'paper_entity_event':sum(e['type']=='entity_event' and e['data'].get('paper_method') is True for e in graph['edges']),
                'multi_occurrence_events':sum(len(n['data'].get('occurrence_chunk_ids',[]))>1 for n in graph['nodes'] if n['type']=='event')}
        result={'status':'OK','qa':len(rows),'types':dict(Counter(r['qa_type'] for r in rows)),
                'dataset_hash':data_hash,'works':stats}
        save_json(HERE/'input_check.json',result);print(dumps(result))
        if args.stage=='preflight':
            api=API(root,HERE/'preflight')
            for model in [cfg['generation_model'],cfg['embedding_model']]+[j['model'] for j in cfg['judges']]:
                obj=api.client.models.retrieve(model);print('[모델 접근] '+obj.id,flush=True)
            # 접근 확인은 생성/구조화 응답 인자 호환성을 보장하지 않는다.
        return 0
    if args.stage in ('prepare','all'):
        with RunLock(base):prepare_assets(root,base,works,docs,cfg)
        if args.stage=='prepare':return 0
    with RunLock(run):
        if (not (base/'summary.json').exists() or read_json(base/'summary.json').get('status')!='COMPLETE'
            or read_json(base/'source_manifest.json')!=input_signature(root,works,cfg)):
            raise ValueError('자산이 없거나 입력/코드가 다릅니다. 새 설치 후 prepare를 먼저 실행하세요.')
        engines={w:(Assets(base,w),None) for w in works}
        engines={w:(a,a.graph) for w,(a,_) in engines.items()}
        fingerprint=bind_new(run,base,cfg,data_hash,works)
        if args.stage in ('generate','all'):
            generate_stage(root,run,rows,selected,engines,cfg,fingerprint,SYSTEMS if args.system=='all' else (args.system,))
        if args.stage in ('judge','all'):
            judge_stage(root,run,rows,selected,docs,cfg,fingerprint,args.judge)
        status=validate_outputs(run,rows,docs,cfg,fingerprint,engines)
        print(dumps({'status':status['status'],'answers':status['answer_counts'],'judges':status['judge_counts']}),flush=True)
        if args.stage in ('report','all'):
            print(dumps(export(root,run,rows,cfg,fingerprint,args.allow_partial)))
        if args.stage in ('generate','judge'):
            missing=status['missing_answers'] if args.stage=='generate' else status['missing_judgments']
            actors=([args.system] if args.system!='all' else list(SYSTEMS)) if args.stage=='generate' else ([args.judge] if args.judge else list(missing))
            incomplete=any(r['qa_id'] in missing[a] for a in actors for r in selected)
        else:incomplete=status['status']!='COMPLETE'
        if incomplete:print('[미완료] status.json과 failures.jsonl 확인. 같은 명령 재실행 시 완료 항목은 건너뜁니다.',flush=True)
        return 1 if incomplete else 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as exc:
        print(f'실행 중단: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)
