import json
import os
import csv
import networkx as nx
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
import math
import requests
import re
import matplotlib.pyplot as plt
import random 
import matplotlib.font_manager as fm
from bs4 import BeautifulSoup
import matplotlib.patches as mpatches
# 나눔글꼴 경로 설정
fe = fm.FontEntry(fname='./NanumGothic.ttf', name='NanumGothic')
fm.fontManager.ttflist.insert(0, fe)
plt.rcParams.update({'font.size': 10, 'font.family': 'NanumGothic'})
plt.rcParams['axes.unicode_minus'] = False
available_fonts = fm.findSystemFonts(fontpaths=None, fontext='ttf')
print(available_fonts)
class ElementType(str):
    # PAGE = "Page"
    TITLE = "Title"
    # FIGURE = "Figure"
    # FOOTER = "Footer"
    CLAUSE_HEADER = "Clause Header"
    LIST_ITEM = "List Item"
    HEADER = "Header"
    DOCUMENT = "Document"
    SECTION_HEADER = "Section Header"
    TEXT = "Text"
    TABLE = "Table"


ELEMENTS = [ElementType.TITLE,
            ElementType.LIST_ITEM, ElementType.HEADER, ElementType.DOCUMENT, 
            ElementType.SECTION_HEADER, ElementType.TEXT, ElementType.TABLE]

# 감사기준서 https://db.kasb.or.kr/api/standard-indexes/530
# https://db.kasb.or.kr/api/paragraphs/1001/mA9o3F?searchWord=
# 한국채택국제회계기준 (K-IFRS)

graph = nx.DiGraph()



class Preprocessor:
    def __init__(self, file_name: str):
        self.file_name = file_name
        self.graph = graph  # Using networkx directed graph
        self.triples = []  # Store relations
        self.json_data = None
        self.elements_by_page_number = None
        self.elements = None
        self.explicit_links = None
        self.footers_by_page_number = defaultdict(list)
        self.hash_lookup_node = {}

        # # Load JSON data for elements
        # print("Loading JSON data...")
        # try:
        #     with open(os.path.join("./docs_structure", "reducto", f"{self.file_name}.json"), "r") as f:
        #         self.json_data = json.load(f)
        #     print("JSON data loaded.")
        # except FileNotFoundError:
        #     raise FileNotFoundError("JSON file not found.")

        self.elements_by_page_number = self.get_elements_sort_by_pages(self.file_name)
        self.elements = self.get_elements(self.file_name)

        self.url = "https://db.kasb.or.kr/api"
        self.index_dict = {
            1000: "재무보고를 위한 개념체계",
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
        self.qa_index_list = [2, 4, 5, 6, 7, 8, 11, 13, 24, 28, 30, 32, 99, 1000, 1001, 1002, 1008, 1012, 1016, 1019, 1021, 1023, 1027, 1028, 1032, 1034, 1036, 1037, 1040, 1102, 1103, 1109, 1110, 1111, 1113, 1115, 1116, 1117, 5003]
        self.index_documents_list = list()
        self._MAX_LEVEL = 5
        self._title_lookup_table: dict[int, dict[str, str]] = {}
        self._parent_document_ids_lookup_table: dict[int, dict[str, list[str]]] = {}


        self.explicit_links = []

        self.documents: dict = {}

    def _get_index_url(self, index: int):
        return f"{self.url}/standard-indexes/{index}"

    def _get_document_url(self, index: int, document_id: str):
        return f"{self.url}/paragraphs/{index}/{document_id}?searchWord="
        # json 저장

    def get_document_response(self, index) -> Dict[str, Any]:
        response = requests.get(self._get_document_url(index, index.get("documentId")))
        return response.json()

    def convert_doc_ids_to_titles(self, index_id: int, doc_ids: list[str]) -> list[str]:
        result = []
        for doc_id in doc_ids:
            try:
                result.append(self._title_lookup_table[index_id][doc_id])
            except Exception:
                result.append("")

        return result

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
            unique_id = f"{index_id}-{doc_metadata['documentId']}"

            doc_metadata["type"] = "title"
            doc_metadata["uniqueKey"] = unique_id
            doc_metadata["children"] = []
            # self._title_lookup_table[index_id][doc_metadata["documentId"]] = (
            #     doc_metadata["title"]
            # )
            # self._title_lookup_table[index_id]["monster-sunjung"] = ""

            parent_ids = list(map(lambda x: f"{index_id}-{x}", filter(
                    lambda x: x != None,
                    doc_metadata["parentDocumentIds"],
                )))
            parent_ids.append(f"{index_id}")
            parent_ids.reverse()
            del doc_metadata["parentDocumentIds"]

            if len(parent_ids) == 1:
                self.documents[index_id]["children"].append(unique_id)
            else:
                parent_id = parent_ids[-1]
                
                if parent_id not in self.documents:
                    self.documents[parent_id] = {
                        "children": []
                    }
                self.documents[parent_id]["children"].append(unique_id)
            doc_metadata["parents"] = parent_ids
            self.documents[unique_id] = doc_metadata

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

    def preprocess_by_index_id(self, index_id: int):
        self.documents[index_id] = {"children": []}
        (lv0_ids, root_level) = self.fetch_all_lv0_ids_by_index_id(index_id)
        for lv0_id in lv0_ids:
            url = self._get_document_url(index_id, lv0_id)
            response = requests.get(url).json()
            documents = response["clauses"]

            for document in documents:
                try:
                    doc_type: str = document["type"]
                except Exception:
                    continue
                
                
                unique_id = f"{index_id}-{document['documentId']}" if doc_type == "title" else f"{index_id}-{document['paraNum']}"

                if "uniqueKey" in document and document["uniqueKey"] != unique_id:
                    print("Error: uniqueKey is not equal to unique_id", document["uniqueKey"], unique_id)

                if doc_type == "title":
                    last_section_id = unique_id
        
                elif doc_type == "paragraph":
                    parent_id = f"{index_id}-{document['documentId']}"
                    document["uniqueKey"] = unique_id
                    document["children"] = []
                    document["parent"] = parent_id
                    self.documents[parent_id]["children"].append(unique_id)
                    self.documents[unique_id] = document


                else:
                    print("Invalid document type")
                    # result_json_data.append(document)
            
            # print or visualize graph
            # return result


    def start(self):
        print("Building graph with nodes and edges...")


        for index_id in self.index_list:
            self.preprocess_by_index_id(index_id)

        with open(f"./result.json", "w") as f:
            json.dump(self.documents, f, indent=4, ensure_ascii=False)
        # self.analyse_links()

    def hash_element(self, element):
        # Create a unique hash ID for an element (simplified for example)
        return hash(json.dumps(element))

    def whyhow_unique_node_name(self, content, element_hash_id):
        # Create a unique name for a node (simplified for example)
        return f"{content[:30]}_{element_hash_id}"

    def link_nodes(self, head_hash, tail_hash, relation):
        self.graph.add_edge(head_hash, tail_hash, relation=relation)

    def get_elements_sort_by_pages(self, file_name):
        # Placeholder for actual function logic
        return {}

    def get_elements(self, file_name):
        # Placeholder for actual function logic
        return {}

    def analyse_links(self):
        for (head_node_id, tail_node_id) in self.explicit_links:
            self.link_nodes(head_node_id, tail_node_id, "links_to")


# Change the code to find title node and retrieve all of its related nodes
# Just find select one title and retrieve all of its related nodes
def get_related_nodes(graph, relation_types):
    related_nodes = set()
    document_id = "8vdODD"
    
    def dfs(node):
        if node not in related_nodes:
            related_nodes.add(node)
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data and 'relation' in edge_data:
                    if edge_data['relation'] in relation_types:
                        dfs(neighbor)
    
    dfs(document_id)
    return related_nodes

central_bank_doc = Preprocessor(file_name="Central_Bank_Document")
lexical_graph = central_bank_doc.start()
