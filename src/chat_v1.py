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
      self.model = init_model(model_provider=self.options.model.provider,
                              model_name=self.options.model.name)
      logger.debug(f"self.model: {self.model}")
    except Exception as e:
      logger.error(f"Error initializing model: {e}")

  async def chat(self, *, prompt, reply_func=None):
    try:
      logger.debug(f"prompt: {prompt}")
      logger.debug(f'self.options: {self.options}')
      logger.debug(f'self.options.systemMessage: {self.options.systemMessage}')
      ai_msg = await self.model.ainvoke([
        SystemMessage(self.options.systemMessage),
        HumanMessage(prompt)
      ])
      print("ai_msg:", ai_msg.content)
      return ai_msg.content
    except Exception as e:
      logger.error(f"Chat error: {e}")
      return f'Error: {str(e)}'
