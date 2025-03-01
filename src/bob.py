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

# Based on
# https://slixmpp.readthedocs.io/en/latest/getting_started/muc.html
# Slixmpp is an MIT licensed XMPP library for Python 3.7+,

import logging
import os
import ssl
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_chroma import Chroma
from langchain_community.vectorstores import Weaviate
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
from bs4 import SoupStrainer
from langchain_googledrive.document_loaders import GoogleDriveLoader  # Use the advanced version.
from langchain_community.document_loaders import UnstructuredFileIOLoader
import weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore

import asyncio
import slixmpp

from base_model import init_model, init_embeddings
from helpers import str_to_bool


load_dotenv()


AGENT_NAME = os.getenv("AGENT_NAME", "bob")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

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

XMPP_JID = os.getenv("XMPP_JID", f"{AGENT_NAME}@selfdev-prosody.dev.local")
XMPP_PASSWORD = os.getenv("XMPP_PASSWORD", "123")
XMPP_ROOM = os.getenv("XMPP_ROOM", "team@conference.selfdev-prosody.dev.local")
XMPP_NICK = os.getenv("XMPP_NICK", AGENT_NAME)


# Set env vars
# os.environ['BROWSER'] = '/usr/bin/chromium'


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


class BobAgent(slixmpp.ClientXMPP):

    """
    A simple Slixmpp bot that will greets those
    who enter the room, and acknowledge any messages
    that mentions the bot's nickname.
    """

    def __init__(self, jid, password, room, nick):
        slixmpp.ClientXMPP.__init__(self, jid, password)

        # Allow insecure certificates
        #
        # Configure SSL context
        self.ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.ssl_context.check_hostname = False
        self.ssl_context.verify_mode = ssl.CERT_NONE
        # Enable all available protocols
        self.ssl_context.minimum_version = ssl.TLSVersion.MINIMUM_SUPPORTED
        self.ssl_context.maximum_version = ssl.TLSVersion.MAXIMUM_SUPPORTED
        # Register event handlers
        self.add_event_handler('ssl_invalid_cert', self.ssl_invalid_cert)

        self.room = room
        self.nick = nick

        # The session_start event will be triggered when
        # the bot establishes its connection with the server
        # and the XML streams are ready for use. We want to
        # listen for this event so that we we can initialize
        # our roster.
        self.add_event_handler("session_start", self.start)

        # The groupchat_message event is triggered whenever a message
        # stanza is received from any chat room. If you also also
        # register a handler for the 'message' event, MUC messages
        # will be processed by both handlers.
        self.add_event_handler("groupchat_message", self.muc_message)

        # The groupchat_presence event is triggered whenever a
        # presence stanza is received from any chat room, including
        # any presences you send yourself. To limit event handling
        # to a single room, use the events muc::room@server::presence,
        # muc::room@server::got_online, or muc::room@server::got_offline.
        self.add_event_handler("muc::%s::got_online" % self.room,
                               self.muc_online)


    def ssl_invalid_cert(self, pem_cert):
        print("Warning: Invalid SSL certificate received")
        return True


    async def start(self, event):
        """
        Process the session_start event.

        Typical actions for the session_start event are
        requesting the roster and broadcasting an initial
        presence stanza.

        Arguments:
            event -- An empty dictionary. The session_start
                     event does not provide any additional
                     data.
        """
        await self.get_roster()
        self.send_presence()
        self.plugin['xep_0045'].join_muc(self.room,
                                         self.nick,
                                         # If a room password is needed, use:
                                         # password=the_room_password,
                                         )

    def muc_message(self, msg):
        """
        Process incoming message stanzas from any chat room. Be aware
        that if you also have any handlers for the 'message' event,
        message stanzas may be processed by both handlers, so check
        the 'type' attribute when using a 'message' event handler.

        Whenever the bot's nickname is mentioned, respond to
        the message.

        IMPORTANT: Always check that a message is not from yourself,
                   otherwise you will create an infinite loop responding
                   to your own messages.

        This handler will reply to messages that mention
        the bot's nickname.

        Arguments:
            msg -- The received message stanza. See the documentation
                   for stanza objects and the Message stanza to see
                   how it may be used.
        """
        # print('msg:', msg)
        if msg['mucnick'] != self.nick and self.nick in msg['body']:
            try:
              prompt = msg['body']
              print('prompt:', prompt)
              # TODO: can we add a SystemMessage(SYSTEM_MESSAGE)?
              response = graph.invoke({"question": prompt})
              self.send_message(mto=msg['from'].bare,
                                mbody=response["answer"],
                                mtype='groupchat')
            except Exception as err:
              print('Chat error:', err)
              self.send_message(mto=msg['from'].bare,
                                mbody=f'Error: {str(err)}',
                                mtype='groupchat')
        # elif msg['mucnick'] != self.nick:
        #     self.send_message(mto=msg['from'].bare,
        #                       mbody=f"Echo: {msg['body']}",
        #                       mtype='groupchat')

    def muc_online(self, presence):
        """
        Process a presence stanza from a chat room. In this case,
        presences from users that have just come online are
        handled by sending a welcome message that includes
        the user's nickname and role in the room.

        Arguments:
            presence -- The received presence stanza. See the
                        documentation for the Presence stanza
                        to see how else it may be used.
        """
        # print('presense:', presence)
        if presence['muc']['nick'] != self.nick:
            self.send_message(mto=presence['from'].bare,
                              mbody="Hello, %s %s" % (presence['muc']['role'],
                                                      presence['muc']['nick']),
                              mtype='groupchat')


if __name__ == '__main__':

    logging.basicConfig(
      level=logging.DEBUG,
      # level=logging.ERROR,
      # level=logging.INFO,
      format='%(levelname)-8s %(message)s'
    )

    xmpp = BobAgent(XMPP_JID, XMPP_PASSWORD, XMPP_ROOM, XMPP_NICK)
    xmpp.register_plugin('xep_0030')  # Service Discovery
    xmpp.register_plugin('xep_0045')  # Multi-User Chat
    xmpp.register_plugin('xep_0199')  # XMPP Ping

    xmpp.connect()
    asyncio.get_event_loop().run_forever()
