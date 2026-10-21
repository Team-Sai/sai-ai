from typing import Any


class LawChunker:
    """
    정규화 + 도메인 필터링이 끝난 법령 Document를
    RAG 검색용 Chunk 구조로 변환한다.

    V1 정책:
    - Document 1개 = Chunk 1개
    - 본문 앞에 법령명/조문 정보를 붙여 임베딩 문맥을 강화한다.
    - 원본 Metadata는 그대로 유지한다.
    """

    def chunk_document(
        self,
        document: dict[str, Any],
    ) -> dict[str, Any]:

        document_id = document["document_id"]

        chunk_id = f"{document_id}:0"

        text = self._build_chunk_text(
            document
        )

        metadata = self._build_metadata(
            document
        )

        return {
            "chunk_id": chunk_id,
            "document_id": document_id,
            "text": text,
            "metadata": metadata,
        }

    def chunk_documents(
        self,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:

        chunks = []

        for document in documents:
            chunk = self.chunk_document(
                document
            )

            chunks.append(
                chunk
            )

        return chunks

    def _build_chunk_text(
        self,
        document: dict[str, Any],
    ) -> str:

        law_name = (
            document.get("law_name")
            or ""
        )

        article_number = (
            document.get("article_number")
            or ""
        )

        article_title = (
            document.get("article_title")
            or ""
        )

        content = (
            document.get("content")
            or ""
        )

        header_parts = []

        if law_name:
            header_parts.append(
                law_name
            )

        if article_number or article_title:
            article_header = " ".join(
                part
                for part in (
                    article_number,
                    article_title,
                )
                if part
            )

            header_parts.append(
                article_header
            )

        header = "\n".join(
            header_parts
        )

        if header and content:
            return (
                f"{header}\n\n"
                f"{content}"
            )

        if content:
            return content

        return header

    @staticmethod
    def _build_metadata(
        document: dict[str, Any],
    ) -> dict[str, Any]:

        return {
            "source_type":
                document.get(
                    "source_type"
                ),

            "document_kind":
                document.get(
                    "document_kind"
                ),

            "law_id":
                document.get(
                    "law_id"
                ),

            "law_name":
                document.get(
                    "law_name"
                ),

            "law_type":
                document.get(
                    "law_type"
                ),

            "ministry":
                document.get(
                    "ministry"
                ),

            "part":
                document.get(
                    "part"
                ),

            "chapter":
                document.get(
                    "chapter"
                ),

            "section":
                document.get(
                    "section"
                ),

            "subsection":
                document.get(
                    "subsection"
                ),

            "article_key":
                document.get(
                    "article_key"
                ),

            "article_number":
                document.get(
                    "article_number"
                ),

            "article_title":
                document.get(
                    "article_title"
                ),

            "effective_date":
                document.get(
                    "effective_date"
                ),

            "article_effective_date":
                document.get(
                    "article_effective_date"
                ),

            "promulgation_date":
                document.get(
                    "promulgation_date"
                ),

            "revision_type":
                document.get(
                    "revision_type"
                ),

            "source":
                document.get(
                    "source"
                ),
        }