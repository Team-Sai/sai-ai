"""
발표 시연 시나리오에 근거 법령을 채워 넣는다. (발표 준비 때 한 번만 실행)

AI 서버를 켠 상태에서 실행하면, 퀵 질문마다 /prepare로 실제 검색한 결과를
config/demo_scenarios.json의 references에 저장한다.
이후 발표 시연에서는 검색하지 않고 이 법령을 정해 둔 시간에 맞춰 보여 준다.

사용: python3 scripts/fill_demo_references.py [AI 서버 주소]
"""

import json
import sys
import urllib.request
from pathlib import Path

CONFIG = Path(__file__).resolve().parents[1] / "config" / "demo_scenarios.json"

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8001").rstrip("/")

# 검색 질의를 만드는 데만 쓰는 예시 계약 (개인 정보 아님)
CONTRACT = {
    "principal": 10000000,
    "interestRate": 5,
    "startDate": "2026-11-01",
    "repaymentDate": "2027-10-31",
    "repaymentDay": 20,
    "repaymentType": "EQUAL_PRINCIPAL",
}


def main() -> None:
    data = json.loads(CONFIG.read_text(encoding="utf-8"))

    for scenario in data["scenarios"]:
        body = json.dumps(
            {"contract": CONTRACT, "message": scenario["question"], "issues": scenario["issues"]},
            ensure_ascii=False,
        ).encode("utf-8")

        request = urllib.request.Request(
            f"{BASE_URL}/rag/contracts/clause-chat/prepare",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(request, timeout=300) as response:
            scenario["references"] = json.loads(response.read().decode("utf-8"))["references"]

        print(f"{scenario['question']} → 근거 법령 {len(scenario['references'])}건")

    CONFIG.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"저장했어요: {CONFIG}")


if __name__ == "__main__":
    main()