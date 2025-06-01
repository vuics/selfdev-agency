'''
RagV1 Agent Archetype

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

from base_model import init_model, init_embeddings
from xmpp_agent import XmppAgent
from helpers import str_to_bool

logger = logging.getLogger("RagV1")

load_dotenv()

# Chroma settings
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")  # host only
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))  # host only
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "agent_{}_collection")  # {} will be replaced by agent name
CHROMA_DIRECTORY = os.getenv("CHROMA_DIRECTORY", "./chroma_db")  # directory only

# Weaviate settings
WEAVIATE_HTTP_HOST = os.getenv("WEAVIATE_HTTP_HOST", "localhost")
WEAVIATE_HTTP_PORT = int(os.getenv("WEAVIATE_HTTP_PORT", "8080"))
WEAVIATE_HTTP_SECURE = str_to_bool(os.getenv("WEAVIATE_HTTP_SECURE", "false"))
WEAVIATE_GRPC_HOST = os.getenv("WEAVIATE_GRPC_HOST", "localhost")
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))
WEAVIATE_GRPC_SECURE = str_to_bool(os.getenv("WEAVIATE_GRPC_SECURE", "false"))

# Google Settings
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
GOOGLE_TOKEN = os.getenv("GOOGLE_TOKEN", "./token.json")


class RagV1(XmppAgent):
  '''
  RagV1 provides chat based on Retrieval-Augmented Generation
  '''

  async def start(self):
    await super().start()
    # Load LLM and embeddings
    try:
      self.model = init_model(
        model_provider=self.config.options.model.provider,
        model_name=self.config.options.model.name,
        api_key=self.config.options.model.apiKey if self.config.options.model.apiKey else None,
      )
      logger.debug(f"self.model: {self.model}")
    except Exception as e:
      logger.error(f"Error initializing model: {e}")

    try:
      self.embeddings = init_embeddings(
        model_provider=self.config.options.embeddings.provider,
        embeddings_name=self.config.options.embeddings.name,
        api_key=self.config.options.embeddings.apiKey if self.config.options.embeddings.apiKey else None,
      )
    except Exception as e:
      logger.error(f"Error initializing embeddings model: {e}")

    # Load vector store
    vector_store = None
    logger.info(f"Vector store: {self.config.options.vectorStore}")
    if self.config.options.vectorStore == "memory":
      vector_store = InMemoryVectorStore(self.embeddings)
    elif self.config.options.vectorStore == "chroma":
      try:
        collection_name = CHROMA_COLLECTION.format(self.config.options.name)
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST,
                                            port=CHROMA_PORT)
        vector_store = Chroma(client=chroma_client,
                              collection_name=collection_name,
                              embedding_function=self.embeddings)
      except Exception as e:
        logger.error(f"Error creating Chroma client: {e}")
        raise
    elif self.config.options.vectorStore == "chroma-dir":
      try:
        collection_name = CHROMA_COLLECTION.format(self.config.options.name)
        vector_store = Chroma(persist_directory=CHROMA_DIRECTORY,
                              collection_name=CHROMA_COLLECTION,
                              embedding_function=self.embeddings)
      except Exception as e:
        logger.error(f"Error creating Chroma client: {e}")
        raise
    elif self.config.options.vectorStore == "weaviate":
      try:
        weaviate_client = weaviate.connect_to_custom(
          http_host=WEAVIATE_HTTP_HOST,
          http_port=WEAVIATE_HTTP_PORT,
          http_secure=WEAVIATE_HTTP_SECURE,
          grpc_host=WEAVIATE_GRPC_HOST,
          grpc_port=WEAVIATE_GRPC_PORT,
          grpc_secure=WEAVIATE_GRPC_SECURE,
        )
        vector_store = WeaviateVectorStore.from_documents([], self.embeddings, client=weaviate_client)  # , tenant=WEAVIATE_TENANT)
      except Exception as e:
        logger.error(f"Error creating Weaviate client: {e}")
        raise
    else:
      raise Exception(f"Unknown vector store: {self.config.options.vectorStore}")

    # Document Loaders
    #
    # Example of loaders:
    # loaders: [
    #   { enable: true, kind: "text", files: [ "/opt/app/README.md", "/opt/app/input/tech-docs/raw-features-list.md", "/opt/app/input/tech-docs/tech-development-leading_draft.md" ] },
    #   { enable: true, kind: "directory", path: "/opt/app/input", glob: "**/*.*" },
    #   { enable: true, kind: "web", urls: [ "https://selfdev.vuics.com", "https://en.wikipedia.org/wiki/Agent-based_model" ] },
    #   { enable: true, kind: "google-drive", folderId: "", recursive: true, filesIds: [ "" ], documentIds: [ "1pi95Wc03l8poJoIJpRXniILIPNGIbDn9VMBfmZPdgZY","1PdeQWPP1EZMXCnNNeMAdUhuRQffTbigfKU3bYC3hGjA","12adeT8_7-9ZP7mO205zFlLxU1PjrvtMviV7uwhRunAY","17U3QGlmaKxY_DoXkSZCC5EhSQ7iVehBJSWVifxRpLPo","114agEJugBBjhOoY8Tj0o0tXdntLg94kyGLPmNBemq1A","1DOwKaugogQy-yR9H-rAd-gqPDIfcDV7B3s6orvytKso","1H9OjmYsSJ8Bq2HE4X3bMidranqvkjqP-kLjkcVxQIGA","1zKuMfvQx0Lq_cJgJmzssOZnIHi7hLEpcILsDq7IPOAY","1iJQQ-__EGdsApjFAPJu2c0-raDnCebcXq33UgWL-2CM","1j1_cTw01NUO7tiVWfRADFV2WddZV-ttORupmkMd66vs","16PrhlaVbOqWL-J6N2zKBzKxROICbbf_R7FCoNEmpXac","1RBULCW0TXrYjTL8i9rFcZXu6cvMIkYmJr4cMqf1B9eI","1ozAo6OGcJRj96pk6OXNLCo-cBHT-vZaJ0PEckEAJzUc","1Oq1T9H6EM-XKmQ1FjGTC7SvZRXoz6k1Z1QLBnx6osDY","1lgvjB6RKYviPHC9sgEaCpZc-lbiBTZ1XWEop3Vbq_iQ","1c1cJSqJKJDYj-w8nSWXoc43uFlyNshqDTscMkk-mFuk","1EfDV6cVE4ipe4ZiAFYhFd4jPsCOrYJ-3ENT0wYf1IDk","1882BF98pW90cb5tS-nCyEuOB2eXO7EOTrNJdyykdC3Q","162yIECys1DdLF88jqfMm9kTvt7HoYs47ixPVqxTir94","15JwiNM-28Z9L-ZMvnLaqXd80yxOO9hAmZ6mU87Kk5zA","1MCPlsbmsyTU_h2ehDiqcLSaDOAGoIIJc4KVz7Nh_J9M" ] },
    # ]

    self.docs = []

    for loader in self.config.options.loaders:
      logger.debug(f"loader: {loader}")
      if not loader.enable:
        continue

      if loader.kind == "text":
        try:
          for file_path in loader.files:
            text_loader = TextLoader(file_path)
            docs_loaded = text_loader.load()
            logger.info(f"TextLoader> documents loaded from {file_path}: {len(docs_loaded)}")
            self.docs += docs_loaded
        except json.JSONDecodeError as e:
          logger.error(f"Error parsing TEXT_LOADER_FILES JSON: {e}")
        except Exception as e:
          logger.error(f"Error loading text files: {e}")

      if loader.kind == "directory":
        directory_loader = DirectoryLoader(
          loader.path,
          loader.glob,
          show_progress=True,
          use_multithreading=True
        )
        docs_loaded = directory_loader.load()
        logger.info(f"DirectoryLoader> documents loaded: {len(docs_loaded)}")
        self.docs += docs_loaded

      if loader.kind == "web":
        web_loader = WebBaseLoader(web_paths=tuple(loader.urls))
        docs_loaded = web_loader.load()
        logger.info(f"WebBaseLoader> documents loaded: {len(docs_loaded)}")
        self.docs += docs_loaded

      if loader.kind == 'google-drive':
        # Enable Google Docs API:
        # https://console.cloud.google.com/apis/api/docs.googleapis.com/metrics
        gdrive_loader = GoogleDriveLoader(
          folder_id=loader.folderId if loader.folderId else None,
          recursive=loader.recursive,
          file_ids=loader.filesIds if loader.filesIds else None,
          document_ids=loader.documentIds if loader.documentIds else None,

          file_loader_cls=UnstructuredFileIOLoader if loader.unstructured else None,
          file_loader_kwargs={"mode": "elements"} if loader.unstructured else None,

          credentials_path=GOOGLE_APPLICATION_CREDENTIALS,
          token_path=GOOGLE_TOKEN,
        )
        docs_loaded = gdrive_loader.load()
        logger.info(f"GoogleDriveLoader> documents loaded: {len(docs_loaded)}")
        self.docs += docs_loaded

    logger.info(f"Total documents loaded: {len(self.docs)}")

    # Retrieval Augmented Generation (RAG)

    if self.docs:
      text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
      all_splits = text_splitter.split_documents(self.docs)
      _ = vector_store.add_documents(documents=all_splits)  # Index chunks

    # TODO: Bring here the whole prompt from hub
    # Define prompt for question-answering
    self.prompt = hub.pull("rlm/rag-prompt")

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
      messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
      response = self.model.invoke(messages)
      return {"answer": response.content}

    # Compile application and test
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    self.graph = graph_builder.compile()
    logger.info('Graph compiled')

  async def chat(self, *, prompt, reply_func=None):
    if not hasattr(self, 'graph'):
      return "I am not ready while loading documents."

    try:
      logger.debug(f"prompt: {prompt}")
      # TODO: can we add a SystemMessage(SYSTEM_MESSAGE)?
      response = self.graph.invoke({"question": prompt})
      logger.debug(f"answer: {response['answer']}")
      return response["answer"]
    except Exception as e:
      logger.error(f"chat error: {e}")
      return f"Error: {str(e)}"
