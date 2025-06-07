'''
ChatV1 Agent Archetype (with message history)
'''
import os
import logging
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage, SystemMessage
# from langchain.memory.chat_message_histories import MongoDBChatMessageHistory
from langchain_community.chat_message_histories import MongoDBChatMessageHistory

from dotenv import load_dotenv

from base_model import init_model
from xmpp_agent import XmppAgent
from file_manager import FileManager

logger = logging.getLogger("ChatV1")

# Load environment variables
load_dotenv()

# MongoDB connection settings
DB_URL = os.getenv("DB_URL", "mongodb://mongo.dev.local:27017/selfdev")


class ChatV1(XmppAgent):
  '''
  ChatV1 provides chats with LLMs using persistent memory via MongoDB
  '''
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.file_manager = FileManager()
    self.chat_history = None  # Will initialize per session

  async def start(self):
    await super().start()
    try:
      # logger.debug(f"self.config.options: {self.config.options}")
      # logger.debug(f"self.config.options.chat: {self.config.options.chat}")
      self.model = init_model(
        model_provider=self.config.options.chat.model.provider,
        model_name=self.config.options.chat.model.name,
        api_key=self.config.options.chat.model.apiKey or None,
      )
      logger.debug(f"Model initialized: {self.model}")

      if self.config.options.chat.session:
        logger.info(f"Message history enabled with session: {self.config.options.chat.session}")
        session_id = f"user_{self.config.userId}_{self.config.options.chat.session}"
        logger.info(f"session_id: {session_id}")
        parsed_db_url = urlparse(DB_URL)
        database_name = parsed_db_url.path.lstrip('/')
        logger.info(f"database_name: {database_name}")
        self.chat_history = MongoDBChatMessageHistory(
          connection_string=DB_URL,
          database_name=database_name,
          collection_name="conversations",
          session_id=session_id,
        )

    except Exception as e:
      logger.error(f"Error initializing model: {e}")

  async def chat(self, *, prompt, reply_func=None):
    try:
      logger.debug(f"Received prompt: {prompt}")

      if self.file_manager.is_shared_file_url(prompt):
        return self.file_manager.add_file_url(prompt)

      files_info = self.file_manager.get_files_info()
      human_content = [{"type": "text", "text": prompt}]

      if files_info:
        human_content.append({"type": "text", "text": "Attached files:"})
        human_content.extend(files_info)
        self.file_manager.clear()
        logger.debug(f"Sending prompt with {len(files_info)} files to model.")

      # Build full message history
      messages = [
        SystemMessage(self.config.options.chat.systemMessage),
        *(self.chat_history.messages if self.chat_history else []),
        HumanMessage(content=human_content)
      ]

      # Call model with full context
      ai_msg = await self.model.ainvoke(messages)

      if self.chat_history:
        # Save messages to history
        self.chat_history.add_user_message(human_content)
        self.chat_history.add_ai_message(ai_msg.content)

      logger.debug(f"model response: {ai_msg.content}")
      return ai_msg.content

    except Exception as e:
      logger.error(f"Chat error: {e}")
      return f"Error: {str(e)}"
