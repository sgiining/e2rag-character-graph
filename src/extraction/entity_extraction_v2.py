"""chunk의 서사 위치를 보존하는 Entity 추출 코드.

입력: split_jsonl_by_chunk.py로 생성한 src/chunking/<work>.jsonl
출력: data/entity/<work>.entities.jsonl

출력 파일의 각 줄은 개체가 특정 chunk에 등장한 기록이다.
각 기록은 observation_id로 구분하며 서로 다른 chunk의 기록은 따로 보존한다.

name_key는 이름을 기준으로 만든 참고용 키다.
name_key가 같다고 반드시 같은 인물이라는 뜻은 아니다.
이 파일에서는 서로 다른 chunk의 개체가 동일 인물인지 판별하지 않는다.

canonical_name에는 현재 chunk에 실제로 등장하는 이름이나 호칭을 저장한다.
검증에서 제외된 항목은 이유를 출력하며 나머지 유효한 항목은 계속 저장한다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI


ENTITY_TYPES = [
    "person",
    "organization",
    "location",
    "object",
    "creature",
    "other",
]

SYSTEM_PROMPT = """한국어 문학 본문에서 개체를 추출하라.
응답은 올바른 JSON 형식으로만 반환하라.

제공된 chunk에서 근거를 확인할 수 없는 개체를 추측해서 추가하지 마라.
서사 위치는 작품 속 사건이 실제로 일어난 시간이 아니라
원문을 나눈 chunk의 위치를 의미한다.

대명사가 가리키는 대상을 현재 chunk 안에서 명확하게 확인할 수 없다면
대명사만을 근거로 개체를 추출하지 마라."""

USER_TEMPLATE = """현재 chunk에서 이름이나 문맥상 특정할 수 있는 표현으로 등장하는 주요 개체를 모두 추출하라.

작품 ID: {work_id}
chunk ID: {chunk_id}
토큰 범위: {token_start}-{token_end}

각 개체에 대해 다음 정보를 반환하라.

- canonical_name:
  현재 chunk에 실제로 등장하는 이름이나 호칭을 그대로 사용하라.
  개체를 가장 구체적으로 가리키는 표현을 선택하라.
  원문에 없는 이름을 만들거나 외부 지식으로 이름을 바꾸지 마라.
  이 필드는 작품 전체에 걸친 동일 인물 판별 결과가 아니다.

- aliases:
  현재 chunk에서 같은 개체를 가리키는 것으로 확인되는 다른 이름이나 호칭.
  원문에 실제로 등장하는 표현만 문자열 목록으로 반환하라.
  다른 표현이 없으면 빈 목록을 반환하라.

- entity_type:
  다음 값 중 하나를 선택하라: {entity_types}
  person은 사람, organization은 조직, location은 장소,
  object는 사물, creature는 사람 이외의 생물이나 생명체,
  other는 그 밖의 개체를 의미한다.

- description:
  현재 chunk에서 확인할 수 있는 개체의 속성이나 행동을 간결하게 설명하라.
  작품의 뒷부분에서 알게 되는 정보나 외부 지식을 사용하지 마라.

- evidence:
  개체를 추출한 근거가 되는 짧은 구절을 현재 chunk에서 그대로 인용하라.
  해당 구절은 개체의 식별이나 설명에 적힌 속성 또는 행동을 뒷받침해야 한다.

다음 구조의 JSON 객체만 반환하라.
추출할 개체가 없으면 entities를 빈 목록으로 반환하라.

{{"entities":[{{"canonical_name":"...","aliases":["..."],"entity_type":"person","description":"...","evidence":"..."}}]}}

본문:
{chunk_text}"""


def stable_id(*parts: str, prefix: str) -> str:
    """같은 입력값에 대해 같은 해시 기반 ID를 생성한다."""
    digest = hashlib.sha256(
        "\x1f".join(parts).encode("utf-8")
    ).hexdigest()[:16]
    return f"{prefix}_{digest}"


def norm(value: str) -> str:
    """앞뒤 공백을 제거하고 연속된 공백을 하나로 정리한다."""
    return re.sub(r"\s+", " ", value).strip()


def slug(value: str) -> str:
    """이름을 참고용 키에 사용할 수 있는 문자열로 바꾼다."""
    value = re.sub(
        r"[^0-9A-Za-z가-힣]+",
        "_",
        norm(value),
    ).strip("_")
    return value[:80] or "unnamed"


def parse_json(text: str) -> dict[str, Any]:
    """모델 응답에서 JSON 객체를 읽고 entities 목록을 확인한다."""
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            text,
            flags=re.I,
        )

    start = text.find("{")
    end = text.rfind("}")

    if start < 0 or end < start:
        raise ValueError(
            f"모델 응답에서 JSON 객체를 찾을 수 없음: {text[:300]}"
        )

    data = json.loads(text[start : end + 1])

    if not isinstance(data.get("entities"), list):
        raise ValueError(
            "JSON 응답에 entities 목록이 있어야 함"
        )

    return data


def call_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    attempts: int = 3,
) -> dict[str, Any]:
    """모델을 호출하고 응답 처리에 실패하면 정해진 횟수만큼 시도한다."""
    last_error: Exception | None = None

    for attempt in range(attempts):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )

            return parse_json(
                response.choices[0].message.content or ""
            )

        except Exception as exc:
            last_error = exc
            time.sleep(2**attempt)

    raise RuntimeError(
        "모델 호출 또는 추출 응답 처리에 실패함"
    ) from last_error


def valid_entity(
    item: Any,
    chunk_text: str,
) -> tuple[bool, str]:
    """개체의 기본 형식과 이름·근거의 원문 포함 여부를 검사한다."""
    if not isinstance(item, dict):
        return False, "개체 항목이 JSON 객체가 아님"

    for field in ("canonical_name", "evidence"):
        value = item.get(field)

        if not isinstance(value, str) or not norm(value):
            return False, f"{field}가 비어 있거나 문자열이 아님"

    name = norm(item["canonical_name"])
    evidence = norm(item["evidence"])
    normalized_text = norm(chunk_text)

    if name not in normalized_text:
        return False, "canonical_name이 원문 chunk에 없음"

    if evidence not in normalized_text:
        return False, "evidence가 원문 chunk에 없음"

    if item.get("entity_type") not in ENTITY_TYPES:
        return False, "허용되지 않은 entity_type"

    # 이후 별칭을 정리할 때 잘못된 형식으로 오류가 나는 것을 방지한다.
    aliases = item.get("aliases", [])

    if not isinstance(aliases, list):
        return False, "aliases가 목록이 아님"

    if not all(isinstance(alias, str) for alias in aliases):
        return False, "aliases에 문자열이 아닌 항목이 있음"

    return True, ""


def extract_file(
    input_path: Path,
    output_path: Path,
    model: str,
) -> None:
    """chunk별로 개체를 추출하고 검증을 통과한 기록을 저장한다."""
    load_dotenv()
    client = OpenAI()

    work_id = input_path.stem
    seen_overlap: set[tuple[str, str]] = set()

    total_extracted = 0
    total_saved = 0
    total_rejected = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        input_path.open(encoding="utf-8") as src,
        output_path.open("w", encoding="utf-8") as dst,
    ):
        for line in src:
            chunk = json.loads(line)
            chunk_id = int(chunk["chunk_id"])
            text = chunk["text"]

            result = call_json(
                client,
                model,
                SYSTEM_PROMPT,
                USER_TEMPLATE.format(
                    work_id=work_id,
                    chunk_id=chunk_id,
                    token_start=chunk["token_start"],
                    token_end=chunk["token_end"],
                    entity_types=ENTITY_TYPES,
                    chunk_text=text,
                ),
            )

            chunk_extracted = len(result["entities"])
            chunk_saved = 0
            chunk_rejected = 0
            total_extracted += chunk_extracted

            for ordinal, raw in enumerate(
                result["entities"],
                start=1,
            ):
                is_valid, reason = valid_entity(raw, text)

                if not is_valid:
                    chunk_rejected += 1
                    total_rejected += 1

                    if isinstance(raw, dict):
                        label = (
                            raw.get("canonical_name")
                            or "(이름 없음)"
                        )
                    else:
                        label = "(잘못된 항목)"

                    print(
                        f"[제외] chunk={chunk_id} "
                        f"항목={ordinal} "
                        f"이름={label!r} "
                        f"사유={reason}",
                        flush=True,
                    )
                    continue

                name = norm(raw["canonical_name"])
                evidence = norm(raw["evidence"])

                # 이전에 같은 이름과 근거 구절이 저장되었는지 확인한다.
                # 반복된 기록도 삭제하지 않고 표시만 남긴다.
                overlap_key = (slug(name), evidence)
                is_overlap_repeat = overlap_key in seen_overlap
                seen_overlap.add(overlap_key)

                aliases = sorted(
                    {
                        norm(alias)
                        for alias in raw.get("aliases", [])
                        if norm(alias)
                    }
                )

                record = {
                    "observation_id": stable_id(
                        work_id,
                        str(chunk_id),
                        str(ordinal),
                        name,
                        evidence,
                        prefix="eobs",
                    ),

                    # 같은 키라고 동일 인물임을 보장하지 않는다.
                    "name_key": f"{work_id}:name:{slug(name)}",

                    "work_id": work_id,
                    "canonical_name": name,
                    "aliases": aliases,
                    "entity_type": raw["entity_type"],
                    "description": norm(
                        str(raw.get("description", ""))
                    ),
                    "evidence": evidence,

                    # 개체 정보가 추출된 chunk의 서사 위치를 보존한다.
                    "chunk_id": chunk_id,
                    "available_at_chunk": chunk_id,

                    # 아래 위치는 근거 구절이 아닌 chunk 전체의 범위다.
                    "token_start": chunk["token_start"],
                    "token_end": chunk["token_end"],

                    "is_overlap_repeat": is_overlap_repeat,
                    "extractor": {
                        "model": model,
                        "schema_version": "entity_observation.v2",
                    },
                }

                dst.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )

                chunk_saved += 1
                total_saved += 1

            print(
                f"[chunk {chunk_id}] "
                f"추출={chunk_extracted} "
                f"저장={chunk_saved} "
                f"제외={chunk_rejected}",
                flush=True,
            )

    print(
        f"[전체 완료] "
        f"추출={total_extracted} "
        f"저장={total_saved} "
        f"제외={total_rejected}",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="chunk별 Entity 추출 및 시점별 기록 저장"
    )
    p.add_argument(
        "input_jsonl",
        type=Path,
        help="chunk가 저장된 입력 JSONL 파일",
    )
    p.add_argument(
        "--output",
        type=Path,
        help="추출 결과를 저장할 JSONL 파일",
    )
    p.add_argument(
        "--model",
        default=os.getenv(
            "EXTRACTION_MODEL",
            "gpt-4o-mini",
        ),
        help="Entity 추출에 사용할 모델",
    )

    args = p.parse_args()

    output = (
        args.output
        or Path("data/entity")
        / f"{args.input_jsonl.stem}.entities.jsonl"
    )

    extract_file(
        args.input_jsonl,
        output,
        args.model,
    )

    print(f"저장 완료: {output}")