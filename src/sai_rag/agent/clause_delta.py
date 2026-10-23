"""
LLM이 JSON을 한 글자씩 써 내려가는 동안 "suggestedClause" 값만 골라낸다.

생성 모델은 아래처럼 JSON 문자열을 조각(delta)으로 보내 준다.

    '{"suggestedClause": "제1조 채무자는'   →  "제1조 채무자는"
    ' 2027년 1월'                          →  " 2027년 1월"
    ' 31일까지\\n변제한다.", "title"'       →  " 31일까지\n변제한다."  (여기서 끝)

JSON 전체가 완성되기 전이라 json.loads를 쓸 수 없으므로,
"suggestedClause" 키 뒤의 문자열만 직접 읽으면서 이스케이프(\\n, \\", \\uXXXX)를 풀어 준다.
조각 경계에서 이스케이프가 잘려도(예: "\\" 다음 조각에 "n") 다음 조각과 이어서 처리한다.
"""

from __future__ import annotations

import re

_SIMPLE_ESCAPES = {
    '"': '"',
    "\\": "\\",
    "/": "/",
    "b": "\b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
}


_INCOMPLETE = object()   # 조각이 잘려서 아직 판단할 수 없음

_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")


def _read_hex(text: str, start: int):
    """text[start:start+4]를 16진수로 읽는다. 모자라면 _INCOMPLETE, 잘못되면 None."""

    hex_part = text[start:start + 4]

    # int(..., 16)은 "+1_2" 같은 것도 받아 주므로 글자를 직접 확인한다.
    if not all(c in _HEX_DIGITS for c in hex_part):
        return None

    if len(hex_part) < 4:
        return _INCOMPLETE

    return int(hex_part, 16)


def _read_low_surrogate(text: str, start: int):
    """서로게이트 쌍의 뒤쪽 "\\uDC00~DFFF"를 읽는다. 모자라면 _INCOMPLETE, 없으면 None."""

    head = text[start:start + 2]

    if len(head) < 2:
        return _INCOMPLETE if head in ("", "\\") else None

    if head != "\\u":
        return None

    code = _read_hex(text, start + 2)

    if code is _INCOMPLETE or code is None:
        return code

    return code if 0xDC00 <= code <= 0xDFFF else None


class ClauseTextExtractor:
    """
    feed(조각)를 부를 때마다 이번에 새로 확정된 조항 글자를 돌려준다.
    """

    def __init__(self, key: str = "suggestedClause") -> None:
        self._start_pattern = re.compile(r'"' + re.escape(key) + r'"\s*:\s*"')
        self._buffer = ""      # 키를 찾기 전까지 모아 둔 원문
        self._pending = ""     # 값 안에서 아직 해석하지 못한 꼬리 (잘린 이스케이프)
        self._inside = False   # 값 문자열 안에 있는지
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    def feed(self, chunk: str) -> str:
        if self._finished or not chunk:
            return ""

        if not self._inside:
            self._buffer += chunk
            match = self._start_pattern.search(self._buffer)

            if not match:
                # 키가 조각 경계에 걸쳐 있을 수 있으니 끝부분만 남긴다.
                self._buffer = self._buffer[-64:]
                return ""

            self._inside = True
            chunk = self._buffer[match.end():]
            self._buffer = ""

        return self._decode(self._pending + chunk)

    def _decode(self, text: str) -> str:
        out: list[str] = []
        index = 0
        self._pending = ""

        while index < len(text):
            char = text[index]

            if char == '"':
                self._finished = True
                self._inside = False
                break

            if char != "\\":
                out.append(char)
                index += 1
                continue

            # 이스케이프: 뒤 글자가 아직 안 왔으면 다음 조각까지 기다린다.
            if index + 1 >= len(text):
                self._pending = text[index:]
                break

            marker = text[index + 1]

            if marker == "u":
                code = _read_hex(text, index + 2)

                if code is _INCOMPLETE:
                    self._pending = text[index:]
                    break

                if code is None:
                    # 잘못된 \u 이스케이프(예: "\uZZ"): 그 두 글자만 버리고 이어서 읽는다.
                    # (뒤에 오는 따옴표까지 삼켜 조항 끝을 놓치지 않도록 6글자를 건너뛰지 않는다)
                    index += 2
                    continue

                if 0xD800 <= code <= 0xDBFF:
                    # 이모지 같은 글자는 "\uD83D\uDE00"처럼 두 덩어리(서로게이트 쌍)로 온다.
                    low = _read_low_surrogate(text, index + 6)

                    if low is _INCOMPLETE:
                        self._pending = text[index:]
                        break

                    if low is not None:
                        out.append(
                            chr(0x10000 + ((code - 0xD800) << 10) + (low - 0xDC00))
                        )
                        index += 12
                        continue

                    # 짝이 없는 앞쪽 덩어리는 화면에 깨진 글자를 만들므로 버린다.
                    index += 6
                    continue

                if 0xDC00 <= code <= 0xDFFF:
                    # 짝이 없는 뒤쪽 덩어리도 버린다.
                    index += 6
                    continue

                out.append(chr(code))
                index += 6
                continue

            # JSON에 없는 이스케이프(예: "\x")는 표준대로라면 오류지만,
            # 미리보기용이므로 뒤 글자를 그대로 보여준다. (최종 문장은 draft/final로 다시 옴)

            out.append(_SIMPLE_ESCAPES.get(marker, marker))
            index += 2

        return "".join(out)