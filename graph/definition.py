import json
import os
import fitz  # PyMuPDF
import csv
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from typing import List, Any
import networkx as nx
# Use NetworkX to represent the graph

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Initialize OpenAI client
openai_client = OpenAI()
openai_client.api_key = openai_api_key


class TermDefinition(BaseModel):
    """
    Pydantic schema used to describe the structured output of legal term-definition pairs for the GPT-4o model.
    """

    term: str
    definition: str


class TermDefinitionCSV(BaseModel):
    """
    Pydantic schema used to describe the structured output of legal term-definition pairs CSv for GPT-4o
    """

    content: List[TermDefinition]


class DefinitionGraphBuilder(BaseModel):
    """
    Given a range of pages, the builder extracts legal terms and their definitions from a PDF.
    This PDF is found using the global FILENAME.
    The builder can save extracted terms as a CSV file, and upload to WhyHow.

    If build_csv is False, no new CSV is created, only the existing one is uploaded.
    """

    # page range starts count from 1
    definition_key: str
    file_path: str
    definitions_csv_path: str = "./definitions.csv"
    definitions_csv_filename: str = os.path.basename(definitions_csv_path)


    # if build_csv is False, the builder will not create a new local CSV, it will just upload the existing one
    build_csv: bool = True
    
    graph: Any = None
    with open(file_path, "r") as file:
        documents = json.load(file)
    def extract_definition_keys(self):
        definition_keys = []
        for key, document in self.documents.items():
            if "-" not in key:
                continue
            if "용어의 정의" in document["title"]:
                definition_keys.append(key)
        return

    def extract_and_save_to_csv(self):
        text = self.documents[self.definition_key]['fullContent']

        all_terms_and_definitions = []

        prompt = f"""
You will be shown a text from a page of a document containing legal terms and their definitions. Your task is to extract the legal terms and their corresponding definitions (verbatim) from the text and return them as the schema JSON.

Text:
{text}

Provide the extracted terms and definitions in the specified CSV format.
"""
        completion = openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract the terms and definitions."},
                {"role": "user", "content": prompt},
            ],
            response_format=TermDefinitionCSV,
        )

        terms_and_definitions = completion.choices[0].message.parsed.content
        print(f"{terms_and_definitions}")

        # Ensure that terms_and_definitions is a list of TermDefinition objects
        if isinstance(terms_and_definitions, list):
            for item in terms_and_definitions:
                if isinstance(item, TermDefinition):
                    all_terms_and_definitions.append((item.term, item.definition))
                else:
                    print("Unexpected item format:", item)

        with open(self.definitions_csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Legal Term", "Definition"])
            for term, definition in all_terms_and_definitions:
                writer.writerow([term, definition])

        print(f"Created new CSV file, saved to {self.definitions_csv_path}")

    def build_definitions_graph(self):
        if self.build_csv:
            self.extract_and_save_to_csv()

        self.graph = nx.DiGraph()

        with open(self.definitions_csv_path, "r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                term = row["Legal Term"]
                definition = row["Definition"]
                self.graph.add_node(term, type="Legal Term")
                self.graph.add_node(definition, type="Definition")
                self.graph.add_edge(term, definition, type="HAS_DEFINITION")
        # print(f"Created graph with ID: {self.graph.graph_id}")

        return self.graph


definitions_graph = DefinitionGraphBuilder(
    index_id="1109",
    definition_key="1109-웩13",
    file_path=f"./kifrs_result.json",
    definitions_csv_path=f"./definitions/definitions.csv",
    build_csv=False,
).build_definitions_graph()