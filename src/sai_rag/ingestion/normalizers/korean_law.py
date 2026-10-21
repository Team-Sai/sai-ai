import json
import re
from pathlib import Path
from typing import Any


class KoreanLawNormalizer:
    """
    법제처 국가법령정보 Open API의 Raw JSON을
    RAG에서 사용할 공통 Document 구조로 변환한다.

    역할:
    - dict / list 혼재 구조 통일
    - 조 / 항 / 호 / 목 텍스트 복원
    - 편 / 장 / 절 / 관 구조 추적
    - 날짜 형식 통일
    - 법령 Metadata 생성
    - 부칙 별도 Document 생성

    이 클래스에서는 Chunking을 하지 않는다.
    """

    PART_PATTERN = re.compile(
        r"^제\s*(\d+)\s*편\s*(.+)$"
    )
    CHAPTER_PATTERN = re.compile(
        r"^제\s*(\d+)\s*장\s*(.+)$"
    )
    SECTION_PATTERN = re.compile(
        r"^제\s*(\d+)\s*절\s*(.+)$"
    )
    SUBSECTION_PATTERN = re.compile(
        r"^제\s*(\d+)\s*관\s*(.+)$"
    )

    def normalize_file(
        self,
        raw_path: Path,
    ) -> dict[str, Any]:
        """
        Raw JSON 파일 하나를 읽고 정규화한다.
        """

        with raw_path.open(
            "r",
            encoding="utf-8",
        ) as f:
            raw_data = json.load(f)

        return self.normalize(
            raw_data=raw_data,
            source_file=raw_path.name,
        )

    def normalize(
        self,
        raw_data: dict[str, Any],
        source_file: str | None = None,
    ) -> dict[str, Any]:

        law = raw_data.get("법령", {})

        if not law:
            raise ValueError(
                "'법령' 데이터가 존재하지 않습니다."
            )

        basic_info = law.get(
            "기본정보",
            {},
        )

        law_id = self._clean_text(
            basic_info.get("법령ID")
        )

        law_name = self._clean_text(
            basic_info.get("법령명_한글")
        )

        if not law_id:
            raise ValueError(
                "법령ID가 존재하지 않습니다."
            )

        if not law_name:
            raise ValueError(
                "법령명이 존재하지 않습니다."
            )

        source_metadata = {
            "law_id": law_id,
            "law_key": self._clean_text(
                law.get("법령키")
            ),
            "law_name": law_name,
            "law_name_english": self._clean_text(
                basic_info.get("법령명_영어")
            ),
            "law_type": self._get_content(
                basic_info.get("법종구분")
            ),
            "ministry": self._get_content(
                basic_info.get("소관부처")
            ),
            "promulgation_number": self._clean_text(
                basic_info.get("공포번호")
            ),
            "promulgation_date": self._normalize_date(
                basic_info.get("공포일자")
            ),
            "effective_date": self._normalize_date(
                basic_info.get("시행일자")
            ),
            "revision_type": self._clean_text(
                basic_info.get("제개정구분")
            ),
            "source": "MOLEG",
            "source_file": source_file,
        }

        documents: list[dict[str, Any]] = []

        documents.extend(
            self._normalize_articles(
                law=law,
                source_metadata=source_metadata,
            )
        )

        documents.extend(
            self._normalize_supplementary(
                law=law,
                source_metadata=source_metadata,
            )
        )

        return {
            "schema_version": "1.0",
            "source": source_metadata,
            "documents": documents,
        }

    def _normalize_articles(
        self,
        law: dict[str, Any],
        source_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:

        articles_raw = (
            law.get("조문", {})
            .get("조문단위")
        )

        articles = self._ensure_list(
            articles_raw
        )

        documents = []

        section_context = {
            "part": None,
            "chapter": None,
            "section": None,
            "subsection": None,
        }

        for article in articles:

            if not isinstance(
                article,
                dict,
            ):
                continue

            article_content = self._clean_text(
                article.get("조문내용")
            )

            if not article_content:
                continue

            # 민법 등의 "제1편 총칙", "제1장 통칙" 등을
            # 별도 문서로 만들지 않고 현재 문맥으로 저장한다.
            if self._apply_heading_context(
                article_content,
                section_context,
            ):
                continue

            content = self._build_article_content(
                article
            )

            if not content:
                continue

            article_key = self._clean_text(
                article.get("조문키")
            )

            article_number = self._make_article_number(
                article
            )

            article_title = self._clean_text(
                article.get("조문제목")
            )

            article_type = self._clean_text(
                article.get("조문여부")
            )

            # 일반적인 조문은 ARTICLE.
            # 이자제한법 최고이자율 대통령령처럼
            # 조문여부="전문"인데 실제 법규 내용인 경우도 보존한다.
            document_kind = (
                "ARTICLE"
                if article_type == "조문"
                else "MAIN_TEXT"
            )

            document_id = (
                f"{source_metadata['law_id']}:"
                f"{article_key or article_number or 'main'}"
            )

            document = {
                "document_id": document_id,
                "document_kind": document_kind,
                "source_type": "LAW",

                "law_id": source_metadata["law_id"],
                "law_name": source_metadata["law_name"],
                "law_type": source_metadata["law_type"],
                "ministry": source_metadata["ministry"],

                "part": section_context["part"],
                "chapter": section_context["chapter"],
                "section": section_context["section"],
                "subsection": section_context["subsection"],

                "article_key": article_key,
                "article_number": article_number,
                "article_title": (
                    article_title or None
                ),

                "content": content,

                "effective_date": (
                    source_metadata[
                        "effective_date"
                    ]
                ),
                "article_effective_date":
                    self._normalize_date(
                        article.get(
                            "조문시행일자"
                        )
                    ),

                "promulgation_date":
                    source_metadata[
                        "promulgation_date"
                    ],

                "revision_type":
                    self._clean_text(
                        article.get(
                            "조문제개정유형"
                        )
                    ),

                "source": "MOLEG",
            }

            documents.append(
                document
            )

        return documents

    def _normalize_supplementary(
        self,
        law: dict[str, Any],
        source_metadata: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        부칙은 본문 조문과 섞지 않고
        SUPPLEMENTARY Document로 별도 보존한다.

        당장 Vector DB에 넣을지는 다음 단계에서 결정한다.
        """

        supplementary_raw = (
            law.get("부칙", {})
            .get("부칙단위")
        )

        supplementary_list = (
            self._ensure_list(
                supplementary_raw
            )
        )

        documents = []

        for supplementary in supplementary_list:

            if not isinstance(
                supplementary,
                dict,
            ):
                continue

            supplementary_key = (
                self._clean_text(
                    supplementary.get(
                        "부칙키"
                    )
                )
            )

            lines = self._flatten_strings(
                supplementary.get(
                    "부칙내용"
                )
            )

            content = "\n".join(
                line
                for line in lines
                if line
            )

            if not content:
                continue

            documents.append({
                "document_id": (
                    f"{source_metadata['law_id']}:"
                    f"supplementary:"
                    f"{supplementary_key}"
                ),

                "document_kind":
                    "SUPPLEMENTARY",

                "source_type": "LAW",

                "law_id":
                    source_metadata[
                        "law_id"
                    ],

                "law_name":
                    source_metadata[
                        "law_name"
                    ],

                "law_type":
                    source_metadata[
                        "law_type"
                    ],

                "ministry":
                    source_metadata[
                        "ministry"
                    ],

                "part": None,
                "chapter": None,
                "section": None,
                "subsection": None,

                "article_key": None,
                "article_number": None,
                "article_title": "부칙",

                "content": content,

                "effective_date":
                    source_metadata[
                        "effective_date"
                    ],

                "article_effective_date":
                    None,

                "promulgation_date":
                    self._normalize_date(
                        supplementary.get(
                            "부칙공포일자"
                        )
                    ),

                "revision_type": None,

                "source": "MOLEG",
            })

        return documents

    def _build_article_content(
        self,
        article: dict[str, Any],
    ) -> str:
        """
        조문내용 + 항 + 호 + 목을
        사람이 읽을 수 있는 텍스트 하나로 복원한다.
        """

        lines = []

        article_content = self._clean_text(
            article.get("조문내용")
        )

        if article_content:
            lines.append(
                article_content
            )

        paragraphs = self._ensure_list(
            article.get("항")
        )

        for paragraph in paragraphs:
            lines.extend(
                self._collect_hierarchy_text(
                    paragraph
                )
            )

        return "\n".join(
            line
            for line in lines
            if line
        )

    def _collect_hierarchy_text(
        self,
        node: Any,
    ) -> list[str]:
        """
        항 → 호 → 목 구조를 재귀적으로 읽는다.

        법제처 JSON에서는 항/호/목이
        list일 수도 있고 dict 하나일 수도 있기 때문에
        _ensure_list를 사용한다.
        """

        if node is None:
            return []

        if isinstance(
            node,
            list,
        ):
            result = []

            for item in node:
                result.extend(
                    self._collect_hierarchy_text(
                        item
                    )
                )

            return result

        if not isinstance(
            node,
            dict,
        ):
            return []

        result = []

        for content_key in (
            "항내용",
            "호내용",
            "목내용",
        ):
            content = self._clean_text(
                node.get(content_key)
            )

            if content:
                result.append(
                    content
                )
                break

        # 법령에 따라 항 없이 바로 호가 오거나,
        # 호 안에 목이 존재할 수 있다.
        for child_key in (
            "항",
            "호",
            "목",
        ):
            children = self._ensure_list(
                node.get(child_key)
            )

            for child in children:
                result.extend(
                    self._collect_hierarchy_text(
                        child
                    )
                )

        return result

    def _apply_heading_context(
        self,
        content: str,
        context: dict[str, str | None],
    ) -> bool:
        """
        민법 등에 섞여있는:

        제1편 총칙
        제1장 통칙
        제1절 능력

        같은 구조 정보를 감지한다.

        구조정보라면 True를 반환하고,
        실제 법조문이면 False를 반환한다.
        """

        match = self.PART_PATTERN.match(
            content
        )

        if match:
            context["part"] = content
            context["chapter"] = None
            context["section"] = None
            context["subsection"] = None
            return True

        match = self.CHAPTER_PATTERN.match(
            content
        )

        if match:
            context["chapter"] = content
            context["section"] = None
            context["subsection"] = None
            return True

        match = self.SECTION_PATTERN.match(
            content
        )

        if match:
            context["section"] = content
            context["subsection"] = None
            return True

        match = self.SUBSECTION_PATTERN.match(
            content
        )

        if match:
            context["subsection"] = content
            return True

        return False

    @staticmethod
    def _make_article_number(
        article: dict[str, Any],
    ) -> str | None:

        number = str(
            article.get(
                "조문번호",
                "",
            )
        ).strip()

        branch = str(
            article.get(
                "조문가지번호",
                "",
            )
        ).strip()

        # 최고이자율 대통령령처럼
        # 조문번호가 0인 전문 형태
        if not number or number == "0":
            return None

        result = f"제{number}조"

        if branch:
            result += f"의{branch}"

        return result

    @staticmethod
    def _ensure_list(
        value: Any,
    ) -> list[Any]:
        """
        None -> []
        dict -> [dict]
        list -> 그대로

        법제처 API의 가장 흔한 구조 변동을
        이 함수 하나로 통일한다.
        """

        if value is None:
            return []

        if isinstance(
            value,
            list,
        ):
            return value

        return [value]

    @staticmethod
    def _get_content(
        value: Any,
    ) -> str | None:

        if isinstance(
            value,
            dict,
        ):
            value = value.get(
                "content"
            )

        value = (
            KoreanLawNormalizer
            ._clean_text(value)
        )

        return value or None

    @staticmethod
    def _normalize_date(
        value: Any,
    ) -> str | None:

        value = (
            KoreanLawNormalizer
            ._clean_text(value)
        )

        if not value:
            return None

        if (
            len(value) == 8
            and value.isdigit()
        ):
            return (
                f"{value[0:4]}-"
                f"{value[4:6]}-"
                f"{value[6:8]}"
            )

        return value

    @staticmethod
    def _clean_text(
        value: Any,
    ) -> str:

        if value is None:
            return ""

        text = str(value)

        # 탭과 연속 공백 정리.
        # 법률 문장의 실제 내용은 변경하지 않는다.
        text = re.sub(
            r"[ \t]+",
            " ",
            text,
        )

        return text.strip()

    def _flatten_strings(
        self,
        value: Any,
    ) -> list[str]:
        """
        부칙내용처럼
        list 안에 list가 여러 번 중첩된 데이터를
        문자열 배열 하나로 만든다.
        """

        if value is None:
            return []

        if isinstance(
            value,
            str,
        ):
            cleaned = self._clean_text(
                value
            )

            return (
                [cleaned]
                if cleaned
                else []
            )

        if isinstance(
            value,
            list,
        ):
            result = []

            for item in value:
                result.extend(
                    self._flatten_strings(
                        item
                    )
                )

            return result

        return []