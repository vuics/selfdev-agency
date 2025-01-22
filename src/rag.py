#!/usr/bin/env python
'''
RAG Agent (Pitcher)

He is a pitching Assistant capable of answering questions about our documents
using a large language model (LLM) involves integrating a document-processing
pipeline with an LLM API.

Here's an outline of the implementation and a sample Python code using a
framework like `langchain` and OpenAI's GPT models.

The assistant is based on the document:
[Build a Retrieval Augmented Generation (RAG) App: Part 1]
(https://python.langchain.com/docs/tutorials/rag/)
'''
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict


load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "rag")
PORT = int(os.getenv("PORT", "6604"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
EMBEDDINGS_NAME = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

VECTOR_STORE = os.getenv("VECTOR_STORE", "in-memory")

DIRECTORY_LOADER_PATH = os.getenv("DIRECTORY_LOADER_PATH", "./input")
DIRECOTRY_LOADER_GLOB = os.getenv("DIRECTORY_LOADER_GLOB", "**/*.*")

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_TOKEN = os.getenv("GOOGLE_TOKEN", "./google_token.json")


# Load LLM and embeddings
#

llm = None
embeddings = None
if LLM_PROVIDER == "openai":
    llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model=EMBEDDINGS_NAME)
elif LLM_PROVIDER == "ollama":
    embeddings = OllamaEmbeddings(model=MODEL_NAME)
else:
    raise Exception(f"Unknown LLM provider: {LLM_PROVIDER}")

# Load vector store
#
vector_store = None
if VECTOR_STORE == "in-memory":
    vector_store = InMemoryVectorStore(embeddings)
else:
    raise Exception(f"Unknown vector store: {VECTOR_STORE}")


# Document Loaders
#

# WebBaseLoader
"""
from langchain_community.document_loaders import WebBaseLoader
# Load and chunk contents of the blog
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
print("documents loaded:", len(docs))
"""

# DirectoryLoader
# import nltk
# nltk.download('punkt_tab')
# nltk.download('averaged_perceptron_tagger_eng')
#
loader = DirectoryLoader(
  DIRECTORY_LOADER_PATH,  # "./input",
  DIRECOTRY_LOADER_GLOB,  # glob="**/*.*",
  # show_progress=True,
  show_progress=False,
  use_multithreading=True
)
docs = loader.load()
print("documents loaded:", len(docs))

# GoogleDriveLoader
"""
# from langchain_google_community import GoogleDriveLoader # Use the basic version
from langchain_googledrive.document_loaders import GoogleDriveLoader # Use the advanced version.
from langchain_community.document_loaders import UnstructuredFileIOLoader
import os


# Enable Google Docs API:
# https://console.cloud.google.com/apis/api/docs.googleapis.com/metrics?project=crafty-shelter-447210-v3&inv=1&invt=AbmX6A

# To fix "Could not locate runnable browser" error, see:
# https://stackoverflow.com/questions/48056052/webbrowser-get-could-not-locate-runnable-browser
# import webbrowser
# webbrowser.register('chromium', None) # , None,webbrowser.BackgroundBrowser('chrome_path'),1)

loader = GoogleDriveLoader(
    # folder_id = "root",
    # folder_id = "0AJ-wGlqk4qe7Uk9PVA", # Home
    # folder_id="1WQnHtR2juxHJboHlyCPX12PwoQp_CFR1", # My Drive > Meeting Notes

    # file_ids=['1MUuBvIvwWUhQ3b_VdTPv9cMVl3RiZss8HXRC5UJ1FNg'], # MeetingNotes-Hal

    # document_ids=['1MUuBvIvwWUhQ3b_VdTPv9cMVl3RiZss8HXRC5UJ1FNg'], # MeetingNotes-Hal
    # document_ids=['1YR3htOQb-HCa5vlUIlgoFrJa0CZvh3k3rSrfPqIPQio'], # Questions for Odoo

    document_ids=[
        '1pi95Wc03l8poJoIJpRXniILIPNGIbDn9VMBfmZPdgZY', # az1.ai Company Charter, https://docs.google.com/document/d/1pi95Wc03l8poJoIJpRXniILIPNGIbDn9VMBfmZPdgZY/edit?tab=t.0
        '1PdeQWPP1EZMXCnNNeMAdUhuRQffTbigfKU3bYC3hGjA', # az1.ai Venture Ecosystem Database, https://docs.google.com/document/d/1PdeQWPP1EZMXCnNNeMAdUhuRQffTbigfKU3bYC3hGjA/edit?tab=t.0
        '12adeT8_7-9ZP7mO205zFlLxU1PjrvtMviV7uwhRunAY', # Designing an Agentic System with ArangoDB, https://docs.google.com/document/d/12adeT8_7-9ZP7mO205zFlLxU1PjrvtMviV7uwhRunAY/edit?tab=t.0
        '17U3QGlmaKxY_DoXkSZCC5EhSQ7iVehBJSWVifxRpLPo', # az1.ai INVESTMENT to EXIT: FOUR STRATEGIES, https://docs.google.com/document/d/17U3QGlmaKxY_DoXkSZCC5EhSQ7iVehBJSWVifxRpLPo/edit?tab=t.0
        '114agEJugBBjhOoY8Tj0o0tXdntLg94kyGLPmNBemq1A', # Co-CEO GPT Agent Prompt: az1.ai Co-CEO Strategic Advisor: GPT AGENT KNOWLEDGEBASE, https://docs.google.com/document/d/114agEJugBBjhOoY8Tj0o0tXdntLg94kyGLPmNBemq1A/edit?tab=t.0
        '1DOwKaugogQy-yR9H-rAd-gqPDIfcDV7B3s6orvytKso', # WORKING WITH Co-CEO GPT Agent Prompt: az1.ai Co-CEO Strategic Advisor: GPT AGENT KNOWLEDGEBASE, https://docs.google.com/document/d/1DOwKaugogQy-yR9H-rAd-gqPDIfcDV7B3s6orvytKso/edit?tab=t.0
        '1H9OjmYsSJ8Bq2HE4X3bMidranqvkjqP-kLjkcVxQIGA', # Develop KPIs by Categories, https://docs.google.com/document/d/1H9OjmYsSJ8Bq2HE4X3bMidranqvkjqP-kLjkcVxQIGA/edit?tab=t.0
        '1zKuMfvQx0Lq_cJgJmzssOZnIHi7hLEpcILsDq7IPOAY', # 02-FIN Financial KPIs, https://docs.google.com/document/d/1zKuMfvQx0Lq_cJgJmzssOZnIHi7hLEpcILsDq7IPOAY/edit?tab=t.0
        '11FokUQgLmaVsO7fQQi01nEXvpcgY1DhHF5dyRGLZeOE', # 2025 CRITICAL DATES, https://docs.google.com/spreadsheets/d/11FokUQgLmaVsO7fQQi01nEXvpcgY1DhHF5dyRGLZeOE/edit?gid=0#gid=0
        '1iJQQ-__EGdsApjFAPJu2c0-raDnCebcXq33UgWL-2CM', # Executive Sales and Revenue Reporting Integrity Multi-Agent Agentic System: Research for az1.zi Business Plan, https://docs.google.com/document/d/1iJQQ-__EGdsApjFAPJu2c0-raDnCebcXq33UgWL-2CM/edit?tab=t.0
        '1j1_cTw01NUO7tiVWfRADFV2WddZV-ttORupmkMd66vs', # Enable Cross-Domain Project Access for az1.ai Users in asafer.ai: GCP - HOW-TO, https://docs.google.com/document/d/1j1_cTw01NUO7tiVWfRADFV2WddZV-ttORupmkMd66vs/edit?tab=t.0
        '16PrhlaVbOqWL-J6N2zKBzKxROICbbf_R7FCoNEmpXac', # az1.ai - Accelerating SaaS Growth: Winning with Your Minimum Viable Market: Features, MVP, MVS, MVM, https://docs.google.com/document/d/16PrhlaVbOqWL-J6N2zKBzKxROICbbf_R7FCoNEmpXac/edit?tab=t.0
        '1RBULCW0TXrYjTL8i9rFcZXu6cvMIkYmJr4cMqf1B9eI', # a1z.ai VISION, MISSION and VALUES, https://docs.google.com/document/d/1RBULCW0TXrYjTL8i9rFcZXu6cvMIkYmJr4cMqf1B9eI/edit?tab=t.0

        # MARKET_VALIDATION: https://drive.google.com/drive/folders/1Rf64NC04L6x_35iL7aRyP-ddHBvgEAwz
        '1ozAo6OGcJRj96pk6OXNLCo-cBHT-vZaJ0PEckEAJzUc', # README: az1.ai Investment Strategy Documentation, https://docs.google.com/document/d/1ozAo6OGcJRj96pk6OXNLCo-cBHT-vZaJ0PEckEAJzUc/edit?tab=t.0#heading=h.ivdv4l1nu3sj
        '1Oq1T9H6EM-XKmQ1FjGTC7SvZRXoz6k1Z1QLBnx6osDY', # az1.ai_MVS_CHOICE_and_STRATEGY_for_MVS_DOMINATION, https://docs.google.com/document/d/1Oq1T9H6EM-XKmQ1FjGTC7SvZRXoz6k1Z1QLBnx6osDY/edit?tab=t.0#heading=h.jig1w2a1pgie
        '1lgvjB6RKYviPHC9sgEaCpZc-lbiBTZ1XWEop3Vbq_iQ', # az1.ai_MVP_FEATURE_LIST, https://docs.google.com/document/d/1lgvjB6RKYviPHC9sgEaCpZc-lbiBTZ1XWEop3Vbq_iQ/edit?tab=t.0#heading=h.nzftv0n6bxcz
        '1c1cJSqJKJDYj-w8nSWXoc43uFlyNshqDTscMkk-mFuk', # az1.ai_MVP_FEATURE_LIST_v2, https://docs.google.com/document/d/1c1cJSqJKJDYj-w8nSWXoc43uFlyNshqDTscMkk-mFuk/edit?tab=t.0#heading=h.12b7ytss1rtr
        '1EfDV6cVE4ipe4ZiAFYhFd4jPsCOrYJ-3ENT0wYf1IDk', # az1.ai_MVM_MINIMUM_VIABLE_MARKET, https://docs.google.com/document/d/1EfDV6cVE4ipe4ZiAFYhFd4jPsCOrYJ-3ENT0wYf1IDk/edit?tab=t.0#heading=h.mn26lotaaa25
        '1882BF98pW90cb5tS-nCyEuOB2eXO7EOTrNJdyykdC3Q', # az1.ai_MARKETING_VALIDATION_REVIEW_ADDENDUM, https://docs.google.com/document/d/1882BF98pW90cb5tS-nCyEuOB2eXO7EOTrNJdyykdC3Q/edit?tab=t.0
        '162yIECys1DdLF88jqfMm9kTvt7HoYs47ixPVqxTir94', # az1.ai_Market_Validation_Project_Plan, https://docs.google.com/document/d/162yIECys1DdLF88jqfMm9kTvt7HoYs47ixPVqxTir94/edit?tab=t.0#heading=h.y2mp5qiy54n6
        '15JwiNM-28Z9L-ZMvnLaqXd80yxOO9hAmZ6mU87Kk5zA', # az1.ai_MARKET_STRATEGY, https://docs.google.com/document/d/15JwiNM-28Z9L-ZMvnLaqXd80yxOO9hAmZ6mU87Kk5zA/edit?tab=t.0#heading=h.sl0l8aods28i
        '1MCPlsbmsyTU_h2ehDiqcLSaDOAGoIIJc4KVz7Nh_J9M', # az1.ai_GTM_GO_to_MARKET_STRATEGY_for_AGRESSIVE_GROWTH, https://docs.google.com/document/d/1MCPlsbmsyTU_h2ehDiqcLSaDOAGoIIJc4KVz7Nh_J9M/edit?tab=t.0#heading=h.59f735ipfa49
    ],

    recursive=True,

    # file_loader_cls=UnstructuredFileIOLoader,
    # file_loader_kwargs={"mode": "elements"},

    credentials_path=GOOGLE_APPLICATION_CREDENTIALS,
    token_path=GOOGLE_TOKEN,
)

docs = loader.load()
print("documents loaded:", len(docs))
"""


# Retrieval Augmented Generation (RAG)
#

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

# Index chunks
_ = vector_store.add_documents(documents=all_splits)

# Define prompt for question-answering
prompt = hub.pull("rlm/rag-prompt")


# Define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Define application steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# API

app = FastAPI()


class ChatRequest(BaseModel):
    prompt: str


@app.post("/v1/chat")
async def chat(request: ChatRequest):
    try:
        prompt = request.prompt
        print('prompt:', prompt)
        response = graph.invoke({"question": prompt})
        print('Response:', response)
        return JSONResponse(
            content={
                "result": "ok",
                "agent": AGENT_NAME,
                "content": response["answer"],
            },
            status_code=200
        )
    except Exception as err:
        print('Chat error:', err)
        return JSONResponse(
            content={
                "result": "error",
                'error': str(err),
            },
            status_code=500
        )


@app.on_event("shutdown")
async def shutdown_event():
    pass


if __name__ == "__main__":
    import uvicorn
    module = Path(__file__).stem
    print('Start module:', module)
    uvicorn.run(f"{module}:app", host="0.0.0.0", port=PORT, reload=True)
