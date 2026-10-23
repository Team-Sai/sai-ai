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
                hex_part = text[index + 2:index + 6]

                if len(hex_part) < 4:
                    self._pending = text[index:]
                    break

                try:
                    code = int(hex_part, 16)
                except ValueError:
                    code = None

                # 이모지 같은 서로게이트 쌍은 화면 미리보기에서 생략한다.
                # (최종 문장은 draft/final 이벤트로 다시 오므로 빠지지 않는다)
                if code is not None and not 0xD800 <= code <= 0xDFFF:
                    out.append(chr(code))

                index += 6
                continue

            out.append(_SIMPLE_ESCAPES.get(marker, marker))
            index += 2

        return "".join(out)