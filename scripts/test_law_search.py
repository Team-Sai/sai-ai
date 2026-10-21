from pprint import pprint

from src.sai_rag.ingestion.connectors.korean_law import KoreanLawConnector


connector = KoreanLawConnector()

result = connector.search_law("이자제한법")

pprint(result)