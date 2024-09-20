# 감사기준서 https://db.kasb.or.kr/api/standard-indexes/530
# https://db.kasb.or.kr/api/paragraphs/1001/mA9o3F?searchWord=
# 한국채택국제회계기준 (K-IFRS)

import os
import math
from typing import Any, Dict
import requests
import json
import re


class Crawler:
    def __init__(self):
        self.url = "https://db.kasb.or.kr/api"
        self.index_dict = {
            1001: "재무제표 표시",
            1002: "재고자산",
            1007: "현금흐름표",
            1008: "회계정책, 회계추정치 변경과 오류",
            1010: "보고기간후사건",
            1012: "법인세",
            1016: "유형자산",
            1019: "종업원급여",
            1020: "정부보조금의 회계처리와 정부지원의 공시",
            1021: "환율변동효과",
            1023: "차입원가",
            1024: "특수관계자 공시",
            1026: "퇴직급여제도에 의한 회계처리와 보고",
            1027: "별도재무제표",
            1028: "관계기업과 공동기업에 대한 투자",
            1029: "초인플레이션 경제에서의 재무보고",
            1032: "금융상품: 표시",
            1033: "주당이익",
            1034: "중간재무보고",
            1036: "자산손상",
            1037: "충당부채, 우발부채, 우발자산",
            1038: "무형자산",
            1039: "금융상품: 인식과 측정",
            1040: "투자부동산",
            1041: "농림어업",
            1101: "한국채택국제회계기준의 최초채택",
            1102: "주식기준보상",
            1103: "사업결합",
            1105: "매각예정비유동자산과 중단영업",
            1106: "광물자원의 탐사와 평가",
            1107: "금융상품: 공시",
            1108: "영업부문",
            1109: "금융상품",
            1110: "연결재무제표",
            1111: "공동약정",
            1112: "타 기업에 대한 지분의 공시",
            1113: "공정가치 측정",
            1114: "규제연계정",
            1115: "고객과의 계약에서 생기는 수익",
            1116: "리스",
            1117: "보험계약",
            200: "독립된 감사인의 전반적인 목적 및 감사기준에 따른 감사의 수행",
            210: "감사업무 조건의 합의",
            220: "재무제표감사의 품질관리",
            230: "감사문서",
            240: "재무제표감사에서의 부정에 관한 감사인의 책임",
            250: "재무제표감사에서의 법률과 규정의 고려",
            265: "지배기구와의 커뮤니케이션",
            260: "내부통제 미비점에 대한 지배기구와 경영진과의 커뮤니케이션",
            300: "재무제표감사의 계획수립",
            315: "기업과 기업환경 이해를 통한 중요왜곡표시위험의 식별과 평가",
            320: "감사의 계획수립과 수행에 있어서의 중요성",
            330: "평가된 위험에 대한 감사인의 대응",
            402: "서비스조직을 이용하는 기업에 관한 감사 고려사항",
            450: "감사 중 식별된 왜곡표시의 평가",
            500: "감사증거",
            501: "감사증거-특정항목에 대한 구체적인 고려사항",
            505: "외부조회",
            510: "초도감사-기초잔액",
            520: "분석적절차",
            530: "표본감사",
            540: "공정가치 등 회계추정치와 관련 공시에 대한 감사",
            550: "특수관계자",
            560: "후속사건",
            570: "계속기업",
            580: "서면진술",
            600: "그룹재무제표 감사 - 부문감사인이 수행한 업무 등 특별 고려사항",
            610: "내부감사인이 수행한 업무의 활용",
            620: "감사인측 전문가가 수행한 업무의 활용",
            700: "재무제표에 대한 의견형성과 보고",
            701: "감사보고서 핵심감사사항 커뮤니케이션",
            705: "감사견해 변형",
            706: "감사보고서의 강조사항문단과 기타사항문단",
            710: "비교정보 - 대응수치 및 비교재무제표",
            720: "감사받은 재무제표를 포함하고 있는 문서 내의 기타정보와 관련된 감사인의 책임",
            800: "특정목적 재무보고체계에 따라 작성된 재무제표의 감사 - 특별 고려사항",
            805: "단위재무제표와 재무제표 특정 요소, 계정 또는 항목에 대한 감사 - 특별 고려사항",
            810: "요약재무제표에 대한 보고업무",
            1100: "내부회계관리제도의 감사",
            1200: "소규모기업 재무제표에 대한 감사",
            3002: "내부회계관리제도 설계 및 운영 개념체계",
            3003: "내부회계관리제도 평가 및 보고 모범규준",
            99: "재무회계개념체계",
            1: "목적, 구성 및 적용",
            2: "재무제표의 작성과 표시 I",
            3: "재무제표의 작성과 표시 II",
            4: "연결재무제표",
            5: "회계정책, 회계추정의 변경 및 오류",
            6: "금융자산·금융부채",
            7: "재고자산",
            8: "지분법",
            9: "조인트벤처 투자",
            10: "유형자산",
            11: "무형자산",
            12: "사업결합",
            13: "리스",
            14: "충당부채, 우발부채 및 우발자산",
            15: "자본",
            16: "수익",
            17: "정부보조금의 회계처리",
            18: "차입원가자본화",
            19: "주식기준보상",
            20: "자산손상",
            21: "종업원급여",
            22: "법인세회계",
            23: "환율변동효과",
            24: "보고기간후사건",
            25: "특수관계자 공시",
            26: "기본주당이익",
            27: "특수활동",
            28: "중단사업",
            29: "중간재무제표",
            30: "일반기업회계기준의 최초채택",
            31: "중소기업 회계처리 특례",
            32: "동일지배거래",
            33: "온실가스 배출권과 배출부채",
            60: "시행일 및 경과규정",
            91: "보험협회계제규준",
            3004: "내부회계관리제도 설계 및 운영 적용기법",
            3005: "내부회계관리제도 평가 및 보고 적용기법",
            3006: "중소기업 내부회계관리제도 설계 및 운영 적용기법",
            3007: "중소기업 내부회계관리제도 평가 및 보고 적용기법",
            3101: "내부회계관리제도 검토기준",
            3102: "내부회계관리제도 검토기준 적용지침",
            # 회계감사실무지침
            8507: "회계감사실무지침 2018-1",
            8508: "회계감사실무지침 2018-2",
            8509: "회계감사실무지침 2018-3",
            8506: "회계감사실무지침 2017-1",
            8505: "회계감사실무지침 2016-1",
            8504: "회계감사실무지침 2015-1",
            8501: "회계감사실무지침 2014-1",
            8502: "회계감사실무지침 2014-2",
            8503: "회계감사실무지침 2014-3",
            # 기타 기준서
            5001: "결합재무제표",
            5002: "기업구조조정투자회사",
            5003: "집합투자기구",
            5004: "신탁업자의 신탁계정",
            8100: "중소기업회계기준",
            8200: "비영리조직회계기준",
            8300: "기업회계기준전문",
        }
        self.index_list = list(self.index_dict.keys())
        self.index_documents_list = list()
        self._MAX_LEVEL = 5
        self._title_lookup_table: dict[int, dict[str, str]] = {}
        self._parent_document_ids_lookup_table: dict[int, dict[str, list[str]]] = {}

    def _get_index_url(self, index: int):
        return f"{self.url}/standard-indexes/{index}"

    def _get_document_url(self, index: int, document_id: str):
        return f"{self.url}/paragraphs/{index}/{document_id}?searchWord="
        # json 저장

    def get_document_response(self, index) -> Dict[str, Any]:
        response = requests.get(self._get_document_url(index, index.get("documentId")))
        return response.json()

    def save_all_docs(self, root_dir: str = "data/raw"):
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        self.save_all_docs_by_index_id(1001)
        # for index in self.index_list:
        #     self.save_all_docs_by_index_id(index)

    def save_all_docs_by_index_id(self, index_id: int):
        (lv0_ids, root_level) = self.fetch_all_lv0_ids_by_index_id(index_id)
        print(lv0_ids, root_level)
        for lv0_id in lv0_ids:
            url = self._get_document_url(index_id, lv0_id)
            response = requests.get(url).json()
            print(response)
            for document_id, value in self.parse_documents(
                index_id, response["clauses"], root_level
            ).items():
                file_path = f"./data/raw/{document_id}.json"
                with open(file_path, "w", encoding="UTF-8-sig") as f:
                    json.dump(value, f, ensure_ascii=False)

    def parse_documents(
        self,
        index_id: int,
        documents: list[dict],
        init_prev_level,
    ) -> dict[str, dict[str, str | list[dict]]]:
        result: dict[str, dict[str, str | list[Any]]] = (
            {}
        )  # dict[파일 이름, 실제 json 값]
        result_json_data: list[dict] = []
        prev_level: int = init_prev_level

        for document in documents:
            try:
                doc_type: str = document["type"]
            except Exception:
                continue

            curr_level: int | None = document.get("level", None)

            if doc_type == "title":
                assert curr_level != None
                curr_level = math.floor(curr_level)

                if self.check_whether_to_split_file(
                    prev_level=prev_level,
                    curr_level=curr_level,
                    root_level=init_prev_level,
                ):
                    result[result_json_data[0]["documentId"]] = {
                        "invectorType": "kifrs",
                        "title": [self.index_dict[index_id]]
                        + self.convert_doc_ids_to_titles(
                            index_id=index_id,
                            doc_ids=self._parent_document_ids_lookup_table[index_id][
                                result_json_data[0]["documentId"]
                            ],
                        ),
                        "data": result_json_data.copy(),
                    }
                    result_json_data = []

                result_json_data.append(document)
                prev_level = curr_level
                continue
            else:
                document["fullContent"] = document["fullContent"].replace("\n", "")
                result_json_data.append(document)

        return result

    def convert_doc_ids_to_titles(self, index_id: int, doc_ids: list[str]) -> list[str]:
        result = []
        for doc_id in doc_ids:
            try:
                result.append(self._title_lookup_table[index_id][doc_id])
            except Exception:
                result.append("")

        return result

    
    def check_whether_to_split_file(
        self, prev_level: int, curr_level: int, root_level
    ) -> bool:
        """
        파일 분리하지 않는 경우:
        이전 level이 MAX_LEVEL에 도달하고, 현재 level이 max 이전 level보다 클 때, 또는 현재 level과 이전 level이 같을 때, 또는 현재 level과 초기 level이 같을 때

        파일 분리하는 경우:

        """
        if (
            (prev_level >= self._MAX_LEVEL and prev_level < curr_level)
            or prev_level == curr_level
            or prev_level == root_level
        ):
            return False
        else:
            return True

    def fetch_all_lv0_ids_by_index_id(self, index_id: int) -> tuple[list[str], int]:
        root_level = 0
        response = requests.get(self._get_index_url(index_id)).json()
        docs_metadata = response["standardIndexes"]

        lv0s_metadata = list(
            filter(lambda doc_metadata: doc_metadata["level"] == 0, docs_metadata)
        )

        if len(lv0s_metadata) == 0:
            lv0s_metadata = list(
                filter(lambda doc_metadata: doc_metadata["level"] == 1, docs_metadata)
            )

            root_level = 1

        self._title_lookup_table[index_id] = {}
        self._parent_document_ids_lookup_table[index_id] = {}

        for doc_metadata in docs_metadata:
            self._title_lookup_table[index_id][doc_metadata["documentId"]] = (
                doc_metadata["title"]
            )
            self._title_lookup_table[index_id]["monster-sunjung"] = ""
            doc_metadata["parentDocumentIds"].reverse()
            self._parent_document_ids_lookup_table[index_id][
                doc_metadata["documentId"]
            ] = list(
                map(
                    lambda x: x if x != None else "monster-sunjung",
                    doc_metadata["parentDocumentIds"],
                )
            )

        return (
            list(map(lambda lv0_metadata: lv0_metadata["documentId"], lv0s_metadata)),
            root_level,
        )

    def remove_html_tags(self, text: str) -> str:
        clean = re.compile("<.*?>")
        return re.sub(clean, "", text)

    def count_level(self):
        count = 0
        for index in self.index_list:
            url = self._get_index_url(index)
            response = requests.get(url).json()["standardIndexes"]

            for res in response:
                if res["level"] in [3, 3.5, 4, 4.5, 5, 5.5, 6]:
                    count += 1
            count += 2
        print(count)


crawler = Crawler()
crawler.save_all_docs()