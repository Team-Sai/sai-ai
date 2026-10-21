import os
from typing import Any

import requests
from dotenv import load_dotenv


load_dotenv()


class KoreanLawConnector:
    SEARCH_URL = "https://www.law.go.kr/DRF/lawSearch.do"
    SERVICE_URL = "https://www.law.go.kr/DRF/lawService.do"

    def __init__(self):
        self.oc = os.getenv("MOLEG_API_KEY")

        if not self.oc:
            raise ValueError(
                "MOLEG_API_KEY가 없습니다. .env 파일을 확인해주세요."
            )

    def search_law(self, law_name: str) -> dict[str, Any]:
        params = {
            "OC": self.oc,
            "target": "eflaw",
            "type": "JSON",
            "search": 1,
            "query": law_name,
            "nw": 3,
            "display": 100,
            "page": 1,
        }

        response = requests.get(
            self.SEARCH_URL,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        return response.json()

    def find_exact_law(self, law_name: str) -> dict[str, Any]:
        result = self.search_law(law_name)

        law_search = result.get("LawSearch", {})
        laws = law_search.get("law", [])

        if isinstance(laws, dict):
            laws = [laws]

        target_name = law_name.strip()

        # 1. 우선 정확히 일치하는 법령 찾기
        for law in laws:
            name = law.get("법령명한글", "").strip()

            if name == target_name:
                return law

        # 2. 정확히 일치하지 않으면 실제 검색 결과를 보여준다.
        searched_names = [
            law.get("법령명한글", "").strip()
            for law in laws
        ]

        raise LookupError(
            f"정확히 일치하는 법령을 찾지 못했습니다: {law_name}\n"
            f"검색된 법령: {searched_names}"
        )

    def fetch_law(self, law_id: str) -> dict[str, Any]:
        params = {
            "OC": self.oc,
            "target": "eflaw",
            "type": "JSON",
            "ID": law_id,
        }

        response = requests.get(
            self.SERVICE_URL,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        return response.json()