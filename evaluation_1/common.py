"""평가 데이터 검증과 원자 저장 및 API 요청을 담당한다."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time
from collections import Counter, defaultdict

SYSTEMS = ('vanilla', 'e2rag', 'proposed')
TYPES = ('answerable', 'relation_change', 'unanswerable_future')
HERE = Path(__file__).resolve().parent


def dumps(value):
    """해시와 저장에 사용할 결정적 JSON 문자열을 반환한다."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def digest(value):
    """직렬화된 입력의 SHA-256을 구한다."""
    return hashlib.sha256(value if isinstance(value, bytes) else dumps(value).encode()).hexdigest()


def read_json(path):
    """UTF-8 JSON을 읽는다."""
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_jsonl(path):
    """행 번호를 포함해 JSONL 오류를 보고한다."""
    rows = []
    for n, line in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        if line.strip():
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError('객체가 아님')
                rows.append(row)
            except ValueError as exc:
                raise ValueError(f'{path}:{n}: {exc}') from exc
    return rows


def save_json(path, obj):
    """JSON을 임시 파일로 쓴 뒤 원자적으로 교체한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


def save_rows(path, rows):
    """완료 결과를 원자 저장한다. 부분 마지막 행을 만들지 않는다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(''.join(dumps(row) + '\n' for row in rows), encoding='utf-8')
    tmp.replace(path)


def keyed(path, fields):
    """중복 키를 거부하고 기존 완료 레코드를 읽는다."""
    result = {}
    if Path(path).exists():
        for row in read_jsonl(path):
            key = tuple(row[f] for f in fields)
            if key in result:
                raise ValueError(f'중복 저장 결과: {path}: {key}')
            result[key] = row
    return result


def config_load(path):
    """실험 수치와 세 평가자의 고유 모델 설정을 확인한다."""
    cfg = read_json(path)
    for f in ('top_k', 'passage_k', 'context_tokens', 'answer_tokens', 'dimensions'):
        if type(cfg[f]) is not int or cfg[f] < 1:
            raise ValueError(f'양의 정수 설정 필요: {f}')
    if cfg['generation_model'] != 'gpt-4o-mini' or cfg['embedding_model'] != 'text-embedding-3-small' or cfg['dimensions'] != 1536:
        raise ValueError('확정 모델/기존 인덱스 차원과 다른 설정')
    if not 0 <= cfg['propagation_decay'] <= 1:
        raise ValueError('전파 계수는 0~1')
    models = [j['model'] for j in cfg['judges']]
    if len(models) != 3 or len(set(models)) != 3:
        raise ValueError('서로 다른 평가자 모델 3개 필요')
    if set(models) != {'gpt-4.1-mini','gpt-6.1-sol','gpt-5.4-mini'}:
        raise ValueError('확정한 세 평가자 모델만 허용')
    if not 0 < cfg.get('graph_context_tokens',4000) < cfg['context_tokens']:
        raise ValueError('그래프 예산은 전체 문맥 예산보다 작아야 함')
    return cfg


class RunLock:
    """OS 잠금은 프로세스 종료 시 해제된다. 비정상 종료의 잔여 파일로 재개를 막지 않는다."""
    def __init__(self,run):
        self.path=Path(run)/'.running.lock'
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.fp=self.path.open('a+b')
        self.fp.seek(0)
        if not self.fp.read(1):self.fp.write(b'0');self.fp.flush()
        self.fp.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.fp.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.fp.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:
            self.fp.close()
            raise ValueError('같은 실행이 다른 프로세스에서 사용 중: '+str(self.path)) from exc
        return self
    def __exit__(self,*args):
        if os.name=='nt':
            import msvcrt
            self.fp.seek(0);msvcrt.locking(self.fp.fileno(),msvcrt.LK_UNLCK,1)
        else:
            import fcntl
            fcntl.flock(self.fp.fileno(),fcntl.LOCK_UN)
        self.fp.close()


def load_dataset(root):
    """QA 입력 일치와 모든 근거 좌표 및 진행도 경계를 검증한다."""
    root = Path(root)
    folder = root / 'data/qa/final'
    rows = read_jsonl(folder / 'qa.jsonl')
    inputs = keyed(folder / 'qa_inputs.jsonl', ('qa_id',))
    if len(rows) != len(inputs) or len({r['qa_id'] for r in rows}) != len(rows):
        raise ValueError('QA ID 중복 또는 입력 수 불일치')
    docs, manifests, pairs = {}, {}, defaultdict(list)
    for row in rows:
        qid, work, p = row['qa_id'], row['work_id'], row['p']
        if any(row[f] != inputs.get((qid,), {}).get(f) for f in ('qa_id', 'work_id', 'p', 'question')):
            raise ValueError(f'{qid}: QA 입력과 메타데이터 불일치')
        if row['qa_type'] not in TYPES or not row['question'].strip() or not row['ground_truth'].strip():
            raise ValueError(f'{qid}: 유형 또는 질문/정답 누락')
        if Path(work).name != work or any(c in work for c in '/\\:'):
            raise ValueError('작품 ID 경로 오류')
        if work not in docs:
            # 기존 코드의 텍스트 모드와 동일하게 개행을 정규화한다.
            docs[work] = (root / 'data/cleaned' / f'{work}.txt').read_text(encoding='utf-8-sig')
            manifests[work] = read_json(root / 'data/vector_index' / work / 'manifest.json')
        text, manifest = docs[work], manifests[work]
        if digest(text.encode()) != row['document_sha256']:
            raise ValueError(f'{qid}: 원문 해시 불일치')
        if digest((root / 'data/chunk' / f'{work}.jsonl').read_bytes()) != row['chunks_sha256']:
            raise ValueError(f'{qid}: chunk 해시 불일치')
        ends = manifest['read_end_char']
        if type(p) is not int or not 1 <= p <= len(ends) or row['read_end_char'] != ends[p-1]:
            raise ValueError(f'{qid}: p와 read_end_char 불일치')
        for field in ('gold_evidence', 'future_evidence'):
            for ev in row.get(field, []):
                a, b = ev['original_span']
                tau = next((i+1 for i, end in enumerate(ends) if b <= end), None)
                if not 0 <= a < b <= len(text) or text[a:b] != ev['text'] or ev['tau'] != tau:
                    raise ValueError(f'{qid}: {field} 원문/좌표/τ 불일치')
                if field == 'gold_evidence' and b > ends[p-1]:
                    raise ValueError(f'{qid}: 현재 정답 근거가 읽기 범위 밖')
                if field == 'future_evidence' and (b <= ends[p-1] or ev['tau'] <= p):
                    raise ValueError(f'{qid}: 미래 근거가 미래 범위가 아님')
        if row['qa_type'] == 'unanswerable_future':
            if not row.get('future_evidence'):
                raise ValueError(f'{qid}: 미래 평가 근거 없음')
        elif not row.get('gold_evidence'):
            raise ValueError(f'{qid}: 정답 근거 없음')
        if row['qa_type'] == 'relation_change':
            if not row.get('pair_group'):
                raise ValueError('관계 쌍 ID 없음')
            pairs[(work, row['pair_group'])].append(row)
    for key, pair in pairs.items():
        if len(pair) != 2 or pair[0]['p'] == pair[1]['p'] or pair[0]['question'] != pair[1]['question']:
            raise ValueError(f'관계 변화 쌍 오류: {key}')
    summary = read_json(folder / 'summary.json')
    if summary['items'] != len(rows):
        raise ValueError('summary 문항 수 불일치')
    actual = Counter(f"{r['work_id']}|{r['qa_type']}" for r in rows)
    if dict(actual) != summary['by_work_type']:
        raise ValueError('summary 유형별 수 불일치')
    return rows, docs


class API:
    """명시된 모델만 호출하고 JSON 원응답과 실패를 기록한다."""
    def __init__(self, root, run):
        from dotenv import load_dotenv
        from openai import OpenAI
        load_dotenv(Path(root) / '.env', override=False)
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError('프로젝트 루트 .env의 OPENAI_API_KEY가 필요합니다.')
        self.client = OpenAI(max_retries=0, timeout=180)
        self.log = Path(run) / 'calls.jsonl'
    def record(self, value):
        """키를 제외한 호출 메타데이터와 원응답을 기록한다."""
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open('a', encoding='utf-8') as f:
            f.write(dumps(value) + '\n')
    def chat(self, model, messages, label, max_tokens, json_mode=False, options=None, validate=None, schema=None):
        """형식/일시적 오류만 최대 3회 재시도한다. 모델/설정 자동 대체는 없다."""
        opts = dict(options or {})
        kwargs = {'model': model, 'messages': messages, **opts}
        if 'reasoning_effort' in opts:
            kwargs['max_completion_tokens'] = max_tokens
        else:
            kwargs['max_tokens'] = max_tokens
        if schema is not None:
            kwargs['response_format'] = {'type': 'json_schema', 'json_schema': {
                'name': 'evaluation_response', 'strict': True, 'schema': schema}}
        elif json_mode:
            kwargs['response_format'] = {'type': 'json_object'}
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(**kwargs)
            except Exception as exc:
                status = getattr(exc, 'status_code', None)
                self.record({'label': label, 'requested_model': model, 'attempt': attempt+1,
                             'request_sha256': digest(kwargs), 'status': 'api_error',
                             'error_type': type(exc).__name__, 'http_status': status})
                retry = status in (408, 409, 429) or (status is not None and status >= 500)
                retry |= type(exc).__name__ in ('APIConnectionError', 'APITimeoutError')
                if not retry or attempt == 2:
                    raise
                time.sleep(2**attempt)
                continue
            text = response.choices[0].message.content or ''
            usage = response.usage.model_dump() if response.usage else {}
            log = {'label': label, 'attempt': attempt+1, 'requested_model': model,
                   'response_model': response.model, 'response_id': response.id,
                   'request_sha256': digest(kwargs), 'raw': text, 'usage': usage,
                   'finish_reason': response.choices[0].finish_reason}
            try:
                if response.choices[0].finish_reason != 'stop' or not text.strip():
                    raise ValueError('빈 응답 또는 출력 토큰 한도 초과')
                value = json.loads(text) if json_mode else text.strip()
                if validate:
                    value = validate(value)
            except (ValueError, KeyError, TypeError) as exc:
                log.update(status='invalid', error=str(exc))
                self.record(log)
                print(f'[응답 검증 실패] {label} {attempt+1}/3: {exc}', flush=True)
                if attempt == 2:
                    raise ValueError(f'{label}: 3회 응답 검증 실패: {exc}') from exc
                kwargs['messages'] = list(messages) + [
                    {'role': 'assistant', 'content': text},
                    {'role': 'user', 'content': '응답 형식 검증 실패: '+str(exc)+
                     '. 원래 평가 기준을 유지해 JSON 전체를 다시 반환하라.'}]
                continue
            log['status'] = 'ok'
            self.record(log)
            return value, {'response_model': response.model, 'response_id': response.id, 'usage': usage}


class Embedder:
    """기존 data 캐시를 쓰지 않고 평가 폴더 안에 질의 벡터를 저장한다."""
    def __init__(self, api, run, cfg):
        self.api, self.cfg = api, cfg
        self.path = Path(run) / 'cache' / 'embeddings.json'
        self.cache = read_json(self.path) if self.path.exists() else {}
    def encode(self, texts):
        """중복 텍스트를 배치 처리하고 순서·차원·유한성을 검증한다."""
        import numpy as np
        texts = list(texts)
        keys = [digest([self.cfg['embedding_model'], self.cfg['dimensions'], t]) for t in texts]
        missing = {k:t for k,t in zip(keys,texts) if k not in self.cache}
        pairs = list(missing.items())
        batch_size = self.cfg.get('embedding_batch_size', 32)
        for offset in range(0,len(pairs),batch_size):
            batch = pairs[offset:offset+batch_size]
            if any(not t.strip() for _,t in batch):
                raise ValueError('빈 임베딩 텍스트')
            for attempt in range(3):
                try:
                    response = self.api.client.embeddings.create(model=self.cfg['embedding_model'],
                        dimensions=self.cfg['dimensions'], input=[t for _,t in batch])
                    ordered = sorted(response.data,key=lambda x:x.index)
                    if [x.index for x in ordered] != list(range(len(batch))):
                        raise ValueError('임베딩 응답 개수/순서 오류')
                    values = normalize([x.embedding for x in ordered],self.cfg['dimensions'])
                    break
                except Exception as exc:
                    status = getattr(exc,'status_code',None)
                    self.api.record({'label':'embedding_error','attempt':attempt+1,
                                     'error_type':type(exc).__name__,'http_status':status})
                    retry = status in (408,409,429) or (status is not None and status>=500)
                    retry |= type(exc).__name__ in ('APIConnectionError','APITimeoutError')
                    if not retry or attempt==2:
                        raise
                    time.sleep(2**attempt)
            for (key,_),vector in zip(batch,values):
                self.cache[key] = vector.tolist()
            save_json(self.path,self.cache)
            self.api.record({'label':'embedding_batch','items':len(batch),
                             'model':self.cfg['embedding_model'],'status':'ok'})
            print(f'[임베딩] {min(offset+batch_size,len(pairs))}/{len(pairs)}',flush=True)
        return normalize([self.cache[k] for k in keys],self.cfg['dimensions']) if keys else np.empty((0,self.cfg['dimensions']),dtype=np.float32)


def normalize(values, dimensions=1536):
    """검색용 float32 단위 벡터. NaN·Inf·영벡터는 거절한다."""
    import numpy as np
    a=np.asarray(values,dtype=np.float32)
    if a.ndim!=2 or a.shape[1]!=dimensions or not np.isfinite(a).all():
        raise ValueError('임베딩 차원/유한성 오류')
    norms=np.linalg.norm(a.astype(np.float64),axis=1,keepdims=True)
    if (norms==0).any():
        raise ValueError('영벡터')
    return (a/norms).astype(np.float32)


def failure_record(run, stage, actor, qa_id, exc=None):
    """미완료 문항별 최신 오류만 저장한다. 성공하면 해당 오류를 해제한다."""
    path = Path(run) / 'failures.jsonl'
    rows = keyed(path, ('stage', 'actor', 'qa_id'))
    key = (stage, actor, qa_id)
    if exc is None:
        if key in rows:
            del rows[key]
            save_rows(path, list(rows.values()))
        return
    rows[key] = {'stage': stage, 'actor': actor, 'qa_id': qa_id,
                 'error_type': type(exc).__name__, 'error': str(exc)}
    save_rows(path, list(rows.values()))


def permanent_error(exc):
    """인증/권한/모델 미지원은 반복 과금 없이 즉시 중단한다."""
    return getattr(exc, 'status_code', None) in (400, 401, 403, 404, 422)


def validate_answer_records(answers,dataset,fingerprint):
    """답변의 입력·실험·내용을 확인한다. QA 정답은 답변 생성기에 전달되지 않는다."""
    known={r['qa_id']:r for r in dataset}
    for (qid,),a in answers.items():
        if qid not in known or a.get('run_fingerprint')!=fingerprint:raise ValueError('답변 실험/QA 불일치: '+qid)
        if not isinstance(a.get('answer'),str) or not a['answer'].strip():raise ValueError('빈 답변: '+qid)
        if any(a.get(k)!=known[qid][k] for k in ('work_id','p','qa_type','question')):raise ValueError('답변 입력 불일치: '+qid)


def validate_saved(record,row,answers,fingerprint):
    """세 시스템 판정·답변 해시·익명 매핑을 매번 검사한다."""
    if record['run_fingerprint']!=fingerprint or record['qa_id']!=row['qa_id']:raise ValueError('채점 실험/QA 불일치')
    if record.get('answer_hashes')!={s:digest(answers[s]) for s in SYSTEMS}:raise ValueError('채점 대상 답변 변경: '+row['qa_id'])
    mapping=record.get('anonymous_mapping',{})
    if set(mapping)!={'A','B','C'} or set(mapping.values())!=set(SYSTEMS):raise ValueError('익명 매핑 오류')
    if set(record.get('verdicts',{}))!=set(SYSTEMS):raise ValueError('세 시스템 판정 누락')
    for label,system in mapping.items():
        if record['verdicts'][system].get('label')!=label:raise ValueError('익명 라벨과 시스템 판정 불일치')
    from judging import validate_judgment
    validate_judgment({'answers':[record['verdicts'][mapping[k]] for k in ('A','B','C')]},
                      row['qa_type']=='unanswerable_future',stored=True)


