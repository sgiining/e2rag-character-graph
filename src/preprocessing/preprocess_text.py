from pathlib import Path
import re


# ============================================================
# 1. 경로 설정
# ============================================================

# 현재 파일:
# 26_capstone/
# └─ e2rag-character-graph/
#    └─ src/
#       └─ preprocessing/
#          └─ preprocess_text.py

# 26_capstone
WORKSPACE_ROOT = Path(__file__).resolve().parents[3]

# 26_capstone/e2rag-character-graph
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 원본 데이터 폴더
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# 전처리 결과 저장 폴더
CLEANED_DIR = PROJECT_ROOT / "data" / "cleaned"


# ============================================================
# 2. 작품 앞의 공통 정보 제거
# ============================================================

def remove_front_matter(text: str) -> str:
    """
    모든 원본 파일에 공통으로 존재하는 작품 외 정보를 제거한다.

    원본 구조:

    [Andersen 단편선]

    작품 제목

    김선희 역1)1) 역자 약력은 원고 말미에 기재하였습니다.

    실제 작품 내용...
    
    """

    lines = text.splitlines()

    # 내용이 존재하는 줄의 인덱스만 찾는다.
    non_empty_indices = [
        index
        for index, line in enumerate(lines)
        if line.strip()
    ]

    # 최소한
    # 1. [Andersen 단편선]
    # 2. 작품 제목
    # 3. 역자 정보
    # 4. 실제 작품 내용
    # 이 있어야 한다.
    if len(non_empty_indices) < 4:
        raise ValueError(
            "파일 앞부분의 구조가 예상과 다릅니다."
        )

    collection_index = non_empty_indices[0]
    title_index = non_empty_indices[1]
    translator_index = non_empty_indices[2]

    # 첫 번째 내용 줄 검증
    if lines[collection_index].strip() != "[Andersen 단편선]":
        raise ValueError(
            "'[Andersen 단편선]' 헤더를 찾을 수 없습니다."
        )

    # 세 번째 내용 줄이 실제 역자 정보인지 검증
    if "김선희 역" not in lines[translator_index]:
        raise ValueError(
            "'김선희 역' 정보를 찾을 수 없습니다."
        )

    # 역자 정보가 있는 줄 다음부터 전부 가져온다.
    body_lines = lines[translator_index + 1:]

    # 작품 본문 앞에 남아 있는 빈 줄 제거
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)

    return "\n".join(body_lines)


# ============================================================
# 3. 작품 뒤의 역자 정보 제거
# ============================================================

def remove_back_matter(text: str) -> str:
    """
    작품이 끝난 뒤 존재하는 정보를 제거한다.

    공통적으로 다음과 같은 정보가 들어 있다.

    옮긴이 약력 : ...
    블로그 : ...
    인스타 : ...

    따라서 '옮긴이 약력'부터 파일 끝까지 제거한다.
    """

    marker = "옮긴이 약력"

    if marker not in text:
        raise ValueError(
            "'옮긴이 약력'을 찾을 수 없습니다."
        )

    return text.split(marker, 1)[0]


# ============================================================
# 4. 엄지 공주 내부 번역자 주석 제거
# ============================================================

def remove_thumbelina_notes(text: str) -> str:
    """
    '엄지 공주' 본문에 포함된 번역자 주석을 제거한다.

    원본:

    12페니2)2) 옛 영국의 화폐단위
    12페니가 1실링(shilling)이었다. _옮긴이
    를 주고

    결과:

    12페니를 주고

    본문의 '12페니'는 유지하고
    번역자가 추가한 설명만 제거한다.
    """

    pattern = (
        r"12페니"
        r"\s*2\)2\)"
        r"\s*옛 영국의 화폐단위"
        r"\s*12페니가 1실링\(shilling\)이었다\."
        r"\s*_옮긴이"
        r"\s*를 주고"
    )

    text, count = re.subn(
        pattern,
        "12페니를 주고",
        text,
        count=1
    )

    if count != 1:
        raise ValueError(
            "'엄지 공주'의 12페니 번역자 주석을 "
            "정확히 찾지 못했습니다."
        )

    return text


# ============================================================
# 5. 인어 공주 내부 번역자 주석 제거
# ============================================================

def remove_little_mermaid_notes(text: str) -> str:
    """
    '인어 공주' 본문에 포함된 번역자 설명을 제거한다.

    제거 대상:
    1. 삭구 설명
    2. 태피스트리 설명

    '삭구', '태피스트리'라는 본문 단어 자체는 유지한다.
    """

    # --------------------------------------------------------
    # 삭구 주석
    # --------------------------------------------------------

    rigging_pattern = (
        r"삭구"
        r"\s*2\)2\)"
        r"\s*배의 돛대·활대·돛 따위를 다루기 위한"
        r"\s*밧줄·쇠사슬 등"
        r"\s*-?\s*옮긴이"
    )

    text, rigging_count = re.subn(
        rigging_pattern,
        "삭구",
        text,
        count=1
    )

    if rigging_count != 1:
        raise ValueError(
            "'인어 공주'의 삭구 번역자 주석을 "
            "정확히 찾지 못했습니다."
        )

    # --------------------------------------------------------
    # 태피스트리 주석
    # --------------------------------------------------------

    tapestry_pattern = (
        r"태피스트리"
        r"\s*3\)3\)"
        r"\s*색색의 실로 수놓은 벽걸이나 실내장식용 비단"
    )

    text, tapestry_count = re.subn(
        tapestry_pattern,
        "태피스트리",
        text,
        count=1
    )

    if tapestry_count != 1:
        raise ValueError(
            "'인어 공주'의 태피스트리 번역자 주석을 "
            "정확히 찾지 못했습니다."
        )

    return text


# ============================================================
# 6. 불필요한 공백 최소 정리
# ============================================================

def normalize_whitespace(text: str) -> str:
    """
    작품의 문장과 문단 구조는 유지하면서
    불필요한 공백만 정리한다.

    하지 않는 것:
    - 맞춤법 수정
    - 띄어쓰기 수정
    - 문장 합치기
    - 문장부호 제거
    - 등장인물 이름 변경
    - 형태소 분석
    - 불용어 제거
    """

    # 줄바꿈 형식을 \n으로 통일
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # 각 줄 끝의 불필요한 공백 제거
    lines = [
        line.rstrip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    # 빈 줄이 지나치게 많이 반복되는 경우
    # 빈 줄 하나만 남긴다.
    text = re.sub(
        r"\n[ \t]*\n(?:[ \t]*\n)+",
        "\n\n",
        text
    )

    # 파일 맨 앞과 맨 뒤의 빈 공간 제거
    return text.strip()


# ============================================================
# 7. 파일 하나 전처리
# ============================================================

def preprocess_file(input_path: Path) -> None:
    """
    data/raw의 TXT 파일 하나를 전처리하고
    같은 이름으로 data/cleaned에 저장한다.
    """

    file_name = input_path.name

    # 결과 파일 경로
    output_path = CLEANED_DIR / file_name

    # --------------------------------------------------------
    # 원본 읽기
    # --------------------------------------------------------

    # utf-8-sig:
    # 파일 앞에 UTF-8 BOM이 있을 경우 자동 제거한다.
    text = input_path.read_text(
        encoding="utf-8-sig"
    )

    original_length = len(text)

    # --------------------------------------------------------
    # 1단계: 앞부분 제거
    # --------------------------------------------------------

    text = remove_front_matter(text)

    # --------------------------------------------------------
    # 2단계: 뒷부분 제거
    # --------------------------------------------------------

    text = remove_back_matter(text)

    # --------------------------------------------------------
    # 3단계: 특정 작품의 본문 내부 번역자 주석 제거
    # --------------------------------------------------------

    if file_name == "2_엄지 공주.txt":
        text = remove_thumbelina_notes(text)

    elif file_name == "3_인어 공주.txt":
        text = remove_little_mermaid_notes(text)

    # --------------------------------------------------------
    # 4단계: 공백 최소 정리
    # --------------------------------------------------------

    text = normalize_whitespace(text)

    cleaned_length = len(text)

    # --------------------------------------------------------
    # 결과 저장
    # --------------------------------------------------------

    output_path.write_text(
        text,
        encoding="utf-8"
    )

    print(f"[완료] {file_name}")
    print(f"  원본 글자 수   : {original_length:,}")
    print(f"  전처리 글자 수 : {cleaned_length:,}")
    print(f"  저장 위치      : {output_path}")
    print()


# ============================================================
# 8. 전체 TXT 파일 전처리
# ============================================================

def main():
    """
    data/raw 안에 존재하는 모든 .txt 파일을 찾아
    순서대로 전처리한다.
    """

    # cleaned 폴더가 없으면 생성
    CLEANED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # raw 폴더가 존재하는지 확인
    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"raw 폴더를 찾을 수 없습니다:\n{RAW_DIR}"
        )

    # data/raw 안의 모든 txt 파일을 가져온다.
    raw_files = sorted(
        RAW_DIR.glob("*.txt")
    )

    if not raw_files:
        raise FileNotFoundError(
            f"전처리할 TXT 파일이 없습니다:\n{RAW_DIR}"
        )

    print("=" * 60)
    print("Andersen 데이터 전처리 시작")
    print(f"workspace : {WORKSPACE_ROOT}")
    print(f"project   : {PROJECT_ROOT}")
    print(f"파일 수    : {len(raw_files)}")
    print("=" * 60)
    print()

    for input_path in raw_files:
        preprocess_file(input_path)

    print("=" * 60)
    print("모든 작품 전처리 완료")
    print("=" * 60)


# ============================================================
# 9. 직접 실행했을 때만 main() 호출
# ============================================================

if __name__ == "__main__":
    main()