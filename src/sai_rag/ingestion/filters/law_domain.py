import re
from typing import Any


class LawDomainFilter:
    """
    정규화된 법령 Document 중
    서비스 도메인에 필요한 조문만 선택한다.

    조문 선택 범위는 코드가 아니라
    외부 설정 파일에서 관리한다.
    """

    ARTICLE_NUMBER_PATTERN = re.compile(
        r"제(\d+)조(?:의(\d+))?"
    )

    def filter_documents(
        self,
        normalized_data: dict[str, Any],
        scope: dict[str, Any],
    ) -> dict[str, Any]:

        source = normalized_data["source"]
        documents = normalized_data["documents"]

        filtered_documents = []

        for document in documents:

            if self._should_include(
                document=document,
                scope=scope,
            ):
                filtered_documents.append(
                    document
                )

        return {
            "schema_version":
                normalized_data.get(
                    "schema_version",
                    "1.0",
                ),

            "source": source,

            "domain_filter": {
                "include_all":
                    scope.get(
                        "include_all",
                        False,
                    ),

                "article_ranges":
                    scope.get(
                        "article_ranges",
                        [],
                    ),

                "article_numbers":
                    scope.get(
                        "article_numbers",
                        [],
                    ),

                "include_supplementary":
                    scope.get(
                        "include_supplementary",
                        False,
                    ),
            },

            "documents": filtered_documents,
        }

    def _should_include(
        self,
        document: dict[str, Any],
        scope: dict[str, Any],
    ) -> bool:

        document_kind = document.get(
            "document_kind"
        )

        if (
            document_kind
            == "SUPPLEMENTARY"
        ):
            return scope.get(
                "include_supplementary",
                False,
            )

        if scope.get(
            "include_all",
            False,
        ):
            return True

        article_number = (
            document.get(
                "article_number"
            )
        )

        if not article_number:
            return False

        parsed = (
            self._parse_article_number(
                article_number
            )
        )

        if parsed is None:
            return False

        main_number, branch_number = parsed

        if self._matches_exact_article(
            main_number=main_number,
            branch_number=branch_number,
            scope=scope,
        ):
            return True

        if self._matches_article_range(
            main_number=main_number,
            scope=scope,
        ):
            return True

        return False

    def _matches_exact_article(
        self,
        main_number: int,
        branch_number: int | None,
        scope: dict[str, Any],
    ) -> bool:

        configured_numbers = (
            scope.get(
                "article_numbers",
                []
            )
        )

        for configured in configured_numbers:

            parsed = (
                self._parse_article_number(
                    f"제{configured}조"
                )
            )

            if parsed is None:
                continue

            configured_main, configured_branch = (
                parsed
            )

            if (
                main_number
                == configured_main
                and branch_number
                == configured_branch
            ):
                return True

        return False

    @staticmethod
    def _matches_article_range(
        main_number: int,
        scope: dict[str, Any],
    ) -> bool:

        ranges = scope.get(
            "article_ranges",
            []
        )

        for start, end in ranges:

            start_number = int(start)
            end_number = int(end)

            if (
                start_number
                <= main_number
                <= end_number
            ):
                return True

        return False

    def _parse_article_number(
        self,
        article_number: str,
    ) -> tuple[int, int | None] | None:

        match = (
            self.ARTICLE_NUMBER_PATTERN
            .fullmatch(
                article_number.strip()
            )
        )

        if not match:
            return None

        main_number = int(
            match.group(1)
        )

        branch = match.group(2)

        branch_number = (
            int(branch)
            if branch
            else None
        )

        return (
            main_number,
            branch_number,
        )