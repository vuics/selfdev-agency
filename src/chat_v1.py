'''
ChatV1 Agent Archetype
'''
import logging
from langchain_core.messages import HumanMessage, SystemMessage

from base_model import init_model
from xmpp_agent import XmppAgent

logger = logging.getLogger("ChatV1")


class ChatV1(XmppAgent):
  '''
  ChatV1 provides chats with LLMs
  '''
  async def start(self):
    await super().start()
    try:
      self.model = init_model(model_provider=self.config.options.model.provider,
                              model_name=self.config.options.model.name)
      logger.debug(f"self.model: {self.model}")
    except Exception as e:
      logger.error(f"Error initializing model: {e}")

  async def chat(self, *, prompt, reply_func=None):
    try:
      logger.debug(f"prompt: {prompt}")
      logger.debug(f'self.config.options: {self.config.options}')
      logger.debug(f'self.config.options.systemMessage: {self.config.options.systemMessage}')
      ai_msg = await self.model.ainvoke([
        SystemMessage(self.config.options.systemMessage),
        HumanMessage(prompt)
      ])
      print("ai_msg:", ai_msg.content)
      return ai_msg.content
    except Exception as e:
      logger.error(f"Chat error: {e}")
      return f'Error: {str(e)}'
