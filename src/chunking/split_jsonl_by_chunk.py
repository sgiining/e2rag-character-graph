from pathlib import Path
import json
import tiktoken

# =========================
# 청킹 설정
# =========================
CHUNK_SIZE = 1200
OVERLAP = 100

if OVERLAP >= CHUNK_SIZE:
    raise ValueError("OVERLAP은 CHUNK_SIZE보다 작아야 합니다.")

# 현재 파일 위치:
# e2rag-character-graph/src/chunking/split_jsonl_by_chunk.py
#
# 프로젝트 최상위 폴더:
# e2rag-character-graph/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 원본 TXT 파일 폴더
INPUT_DIR = PROJECT_ROOT / "data" / "cleaned"

# JSONL 결과 파일 저장 폴더
OUTPUT_DIR = PROJECT_ROOT / "src" / "chunking"

# data/cleaned 안의 모든 TXT 파일 찾기
txt_files = sorted(INPUT_DIR.glob("*.txt"))

if not txt_files:
    raise FileNotFoundError(
        f"TXT 파일을 찾을 수 없습니다.\n확인한 폴더: {INPUT_DIR}"
    )

# GPT-4o mini와 같은 토큰 분할 규칙 사용
encoding = tiktoken.encoding_for_model("gpt-4o-mini")

# 1200 토큰 청크에서 100 토큰을 겹치게 하므로 1100 토큰씩 이동
stride = CHUNK_SIZE - OVERLAP

# TXT 파일을 하나씩 JSONL로 변환
for txt_path in txt_files:
    # 예: 1_눈의 여왕.txt → src/chunking/1_눈의 여왕.jsonl
    jsonl_path = OUTPUT_DIR / f"{txt_path.stem}.jsonl"

    # UTF-8 TXT 파일 읽기
    text = txt_path.read_text(encoding="utf-8")

    # GPT-4o mini 기준으로 토큰화
    tokens = encoding.encode(text)

    jsonl_lines = []

    # 토큰 기준으로 청크 생성
    for chunk_number, start in enumerate(range(0, len(tokens), stride), start=1):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]

        # 토큰이 없으면 종료
        if not chunk_tokens:
            break

        # 토큰을 다시 텍스트로 변환
        chunk_text = encoding.decode(chunk_tokens)

        # JSONL 한 줄에 들어갈 청크 데이터
        record = {
            "chunk_id": chunk_number,
            "text": chunk_text,
            "token_start": start + 1,
            "token_end": end,
            "token_count": len(chunk_tokens),
            "chunk_size_setting": CHUNK_SIZE,
            "overlap_setting": OVERLAP,
            "tokenizer_model": "gpt-4o-mini"
        }

        # 한 줄 = JSON 객체 하나
        jsonl_lines.append(json.dumps(record, ensure_ascii=False))

        # 마지막 청크까지 처리했으면 종료
        if end >= len(tokens):
            break

    # 각 TXT 파일별 JSONL 결과 저장
    jsonl_path.write_text(
        "\n".join(jsonl_lines),
        encoding="utf-8"
    )

    print(
        f"완료: {txt_path.name} → {jsonl_path.name} | "
        f"원문 {len(tokens):,} tokens | "
        f"{len(jsonl_lines)} chunks"
    )

print("\n모든 TXT 파일의 JSONL 청킹이 완료되었습니다.")