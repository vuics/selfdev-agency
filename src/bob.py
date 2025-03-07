#!/usr/bin/env python
'''
Bob Agent

Bob is a RAG Agent (Pitcher)

He is a pitching Assistant capable of answering questions about our documents
using a large language model (LLM) involves integrating a document-processing
pipeline with an LLM API.

Here's an outline of the implementation and a sample Python code using a
framework like `langchain` and OpenAI's GPT models.

The assistant is based on the document:
[Build a Retrieval Augmented Generation (RAG) App: Part 1]
(https://python.langchain.com/docs/tutorials/rag/)
'''

import logging
import os
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
import chromadb
import json
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_googledrive.document_loaders import GoogleDriveLoader
from langchain_community.document_loaders import UnstructuredFileIOLoader
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore
import asyncio

from base_model import init_model, init_embeddings
from xmpp_agent import XmppAgent
from helpers import str_to_bool


load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME", "bob")
SYSTEM_MESSAGE = os.getenv("SYSTEM_MESSAGE", "")

XMPP_HOST = os.getenv("XMPP_HOST", "selfdev-prosody.dev.local")
XMPP_USER = os.getenv("XMPP_USER", AGENT_NAME)
XMPP_PASSWORD = os.getenv("XMPP_PASSWORD", "123")
XMPP_MUC_HOST = os.getenv("XMPP_MUC_HOST", f"conference.{XMPP_HOST}")
XMPP_JOIN_ROOMS = json.loads(os.getenv('XMPP_JOIN_ROOMS',
                                       '[ "team", "agents" ]'))
XMPP_NICK = os.getenv("XMPP_NICK", AGENT_NAME)

MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "openai")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
# openai: text-embedding-3-large
# ollama: nomic-embed-text
EMBEDDINGS_NAME = os.getenv("EMBEDDINGS_NAME", "text-embedding-3-large")

VECTOR_STORE = os.getenv("VECTOR_STORE", "memory")  # memory, chroma, weaviate

CHROMA_TYPE = os.getenv("CHROMA_TYPE", "host")  # host, directory
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")  # host only
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))  # host only
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", f"{AGENT_NAME}_collection")
CHROMA_DIRECTORY = os.getenv("CHROMA_DIRECTORY", "./chroma_db")  # directory only

# Weaviate settings
# WEAVIATE_TENANT = os.getenv("WEAVIATE_TENANT", f"{AGENT_NAME}_tenant")
WEAVIATE_HTTP_HOST = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_HTTP_SECURE = str_to_bool(os.getenv("WEAVIATE_HTTP_SECURE", "false"))
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "localhost")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_GRPC_SECURE = str_to_bool(os.getenv("WEAVIATE_GRPC_SECURE", "false"))

# TODO: Should I move loaders to a separate script?
TEXT_LOADER = str_to_bool(os.getenv("TEXT_LOADER", "false"))
TEXT_LOADER_FILES = json.loads(os.getenv("TEXT_LOADER_FILES", "[]"))  # JSON array

DIRECTORY_LOADER = str_to_bool(os.getenv("DIRECTORY_LOADER", "false"))
DIRECTORY_LOADER_PATH = os.getenv("DIRECTORY_LOADER_PATH", "./input")
DIRECOTRY_LOADER_GLOB = os.getenv("DIRECTORY_LOADER_GLOB", "**/*.*")

WEB_BASE_LOADER = str_to_bool(os.getenv("WEB_BASE_LOADER", "false"))
WEB_BASE_PATHS = json.loads(os.getenv("WEB_BASE_PATHS", "[]"))

GOOGLE_DRIVE_LOADER = str_to_bool(os.getenv("GOOGLE_DRIVE_LOADER", "false"))
GOOGLE_DRIVE_DOCUMENT_IDS = json.loads(os.getenv("GOOGLE_DRIVE_DOCUMENT_IDS", "[]"))
GOOGLE_DRIVE_FILE_IDS = json.loads(os.getenv("GOOGLE_DRIVE_FILE_IDS", "[]"))
GOOGLE_DRIVE_FOLDER_ID = os.getenv("GOOGLE_DRIVE_FOLDER_ID", "")
GOOGLE_DRIVE_RECURSIVE = str_to_bool(os.getenv("GOOGLE_DRIVE_RECURSIVE", "true"))
GOOGLE_DRIVE_UNSTRUCTURED = str_to_bool(os.getenv("GOOGLE_DRIVE_UNSTRUCTURED", "true"))

GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_TOKEN = os.getenv("GOOGLE_TOKEN", "./google_token.json")


# Load LLM and embeddings
try:
  model = init_model(model_provider=MODEL_PROVIDER,
                     model_name=MODEL_NAME)
except Exception as e:
  print("Error initializing model:", e)

try:
  embeddings = init_embeddings(model_provider=MODEL_PROVIDER,
                               embeddings_name=EMBEDDINGS_NAME)
except Exception as e:
  print("Error initializing embeddings model:", e)


# Load vector store
vector_store = None
if VECTOR_STORE == "memory":
    vector_store = InMemoryVectorStore(embeddings)
elif VECTOR_STORE == "chroma":
    try:
        if CHROMA_TYPE == "host":
            chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
            vector_store = Chroma(client=chroma_client, collection_name=CHROMA_COLLECTION, embedding_function=embeddings)
        elif CHROMA_TYPE == "directory":
            vector_store = Chroma(persist_directory=CHROMA_DIRECTORY, collection_name=CHROMA_COLLECTION, embedding_function=embeddings)
        else:
            raise Exception(f"Unknown chroma type: {CHROMA_TYPE}")
    except Exception as e:
        print(f"Error creating Chroma client: {e}")
        raise
elif VECTOR_STORE == "weaviate":
    try:
        weaviate_client = weaviate.connect_to_custom(
            http_host=WEAVIATE_HTTP_HOST,
            http_port=WEAVIATE_HTTP_PORT,
            http_secure=WEAVIATE_HTTP_SECURE,
            grpc_host=WEAVIATE_GRPC_HOST,
            grpc_port=WEAVIATE_GRPC_PORT,
            grpc_secure=WEAVIATE_GRPC_SECURE,
            # headers={ },
        )
        vector_store = WeaviateVectorStore.from_documents([], embeddings, client=weaviate_client)  #, tenant=WEAVIATE_TENANT)
    except Exception as e:
        print(f"Error creating Weaviate client: {e}")
        raise
else:
    raise Exception(f"Unknown vector store: {VECTOR_STORE}")
print('Vector store:', VECTOR_STORE)


# Document Loaders
docs = []

if TEXT_LOADER:
    try:
        for file_path in TEXT_LOADER_FILES:
            loader = TextLoader(file_path)
            docs_loaded = loader.load()
            print(f"TextLoader> documents loaded from {file_path}:", len(docs_loaded))
            docs += docs_loaded
    except json.JSONDecodeError as e:
        print(f"Error parsing TEXT_LOADER_FILES JSON: {e}")
    except Exception as e:
        print(f"Error loading text files: {e}")

if DIRECTORY_LOADER:
    loader = DirectoryLoader(
      DIRECTORY_LOADER_PATH,
      DIRECOTRY_LOADER_GLOB,
      show_progress=True,
      use_multithreading=True
    )
    docs_loaded = loader.load()
    print("DirectoryLoader> documents loaded:", len(docs_loaded))
    docs += docs_loaded

if WEB_BASE_LOADER:
    loader = WebBaseLoader(
        web_paths=tuple(WEB_BASE_PATHS),
        # bs_kwargs=dict(
        #     parse_only=SoupStrainer(
        #         class_=("post-content", "post-title", "post-header")
        #     )
        # ),
    )
    docs_loaded = loader.load()
    print("WebBaseLoader> documents loaded:", len(docs_loaded))
    docs += docs_loaded

if GOOGLE_DRIVE_LOADER:
    # Enable Google Docs API:
    # https://console.cloud.google.com/apis/api/docs.googleapis.com/metrics
    loader = GoogleDriveLoader(
        folder_id=GOOGLE_DRIVE_FOLDER_ID if GOOGLE_DRIVE_FOLDER_ID else None,
        file_ids=GOOGLE_DRIVE_FILE_IDS if GOOGLE_DRIVE_FILE_IDS else None,
        document_ids=GOOGLE_DRIVE_DOCUMENT_IDS if GOOGLE_DRIVE_DOCUMENT_IDS else None,
        recursive=GOOGLE_DRIVE_RECURSIVE,

        file_loader_cls=UnstructuredFileIOLoader if GOOGLE_DRIVE_UNSTRUCTURED else None,
        file_loader_kwargs={"mode": "elements"} if GOOGLE_DRIVE_UNSTRUCTURED else None,

        credentials_path=GOOGLE_APPLICATION_CREDENTIALS,
        token_path=GOOGLE_TOKEN,
    )
    docs_loaded = loader.load()
    print("GoogleDriveLoader> documents loaded:", len(docs_loaded))
    docs += docs_loaded

print("Total documents loaded:", len(docs))


# Retrieval Augmented Generation (RAG)
#

if docs:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    _ = vector_store.add_documents(documents=all_splits)  # Index chunks

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
    response = model.invoke(messages)
    return {"answer": response.content}


# Compile application and test
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
print('Graph compiled')


class BobAgent(XmppAgent):
  # def __init__(self, *, host, user, password, muc_host, join_rooms, nick, options):
  #   XmppAgent.__init__(self, host, user, password, muc_host, join_rooms, nick, options)

  def chat(self, prompt):
    try:
      print('prompt:', prompt)
      # TODO: can we add a SystemMessage(SYSTEM_MESSAGE)?
      response = graph.invoke({"question": prompt})
      print("answer:", response["answer"])
      return response["answer"],
    except Exception as err:
      print('chat error:', err)
      return 'Error: {str(err)}'


if __name__ == '__main__':
  logging.basicConfig(
    level=logging.INFO,  # DEBUG, ERROR, INFO,
    format='%(levelname)-8s %(message)s'
  )

  agent = BobAgent(
    host=XMPP_HOST,
    user=XMPP_USER,
    password=XMPP_PASSWORD,
    muc_host=XMPP_MUC_HOST,
    join_rooms=XMPP_JOIN_ROOMS,
    nick=XMPP_NICK,
  )
  asyncio.get_event_loop().run_forever()
