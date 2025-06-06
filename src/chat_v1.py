'''
ChatV1 Agent Archetype
'''
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from base_model import init_model
from xmpp_agent import XmppAgent
from file_manager import FileManager

logger = logging.getLogger("ChatV1")


class ChatV1(XmppAgent):
  '''
  ChatV1 provides chats with LLMs
  '''
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.file_manager = FileManager()

  async def start(self):
    await super().start()
    try:
      # logger.debug(f"self.config.options: {self.config.options}")
      self.model = init_model(
        model_provider=self.config.options.model.provider,
        model_name=self.config.options.model.name,
        api_key=self.config.options.model.apiKey if self.config.options.model.apiKey else None,
      )
      logger.debug(f"self.model: {self.model}")
    except Exception as e:
      logger.error(f"Error initializing model: {e}")

  async def chat(self, *, prompt, reply_func=None):
    try:
      logger.debug(f"Received prompt: {prompt}")

      # Check if prompt is exactly a URL matching the share URL prefix
      if self.file_manager.is_file_url(prompt):
        logger.debug("Prompt is a file URL; downloading and storing.")
        self.file_manager.add_file_from_url(prompt)
        return "File downloaded and stored."

      # Otherwise, treat prompt as a message with possible previous files
      files = self.file_manager.get_all_files()
      human_content = [{"type": "text", "text": prompt}]

      # Append all stored files to the message if any
      if files:
        human_content.append({"type": "text", "text": "Attached files:"})
        human_content.extend(files)
        self.file_manager.clear()
        logger.debug(f"Sending prompt with {len(files)} files to model.")

      ai_msg = await self.model.ainvoke([
        SystemMessage(self.config.options.systemMessage),
        HumanMessage(content=human_content),
      ])

      logger.debug(f"AI response: {ai_msg.content}")
      return ai_msg.content
    except Exception as e:
      logger.error(f"Chat error: {e}")
      return f"Error: {str(e)}"
