'''
Alice Agent
'''
import logging
from langchain_core.messages import HumanMessage, SystemMessage

from base_model import init_model
from xmpp_agent import XmppAgent

logger = logging.getLogger("alice")


class AliceAgent(XmppAgent):
  '''
  AliceAgent provides chats with LLMs
  '''
  # def __init__(self, *, host, user, password, muc_host, join_rooms, nick, options):
  #   XmppAgent.__init__(self, host, user, password, muc_host, join_rooms, nick, options)

  def start(self):
    try:
      self.model = init_model(model_provider=self.options.model.provider,
                              model_name=self.options.model.name)
      logger.debug(f"self.model: {self.model}")
    except Exception as e:
      print("Error initializing model:", e)

  async def chat(self, *, prompt):
    try:
      print('prompt:', prompt)
      logger.debug(f'self.options: {self.options}')
      logger.debug(f'config.systemMessage: {self.options.systemMessage}')
      ai_msg = await self.model.ainvoke([
        SystemMessage(self.options.systemMessage),
        HumanMessage(prompt)
      ])
      print("ai_msg:", ai_msg.content)
      return ai_msg.content
    except Exception as e:
      print('chat error:', e)
      return f'Error: {str(e)}'
