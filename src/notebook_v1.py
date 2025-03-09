'''
NotebookV1 Agent Archetype
'''
import logging
# from langchain_core.messages import HumanMessage, SystemMessage

# from base_model import init_model
from xmpp_agent import XmppAgent
logger = logging.getLogger("NotebookV1")


class NotebookV1(XmppAgent):
  '''
  NotebookV1 provides exectution of Jupyter Notebooks and their outputs
  '''
  async def start(self):
    pass

  async def chat(self, *, prompt):
    try:
      logger.debug(f"prompt: {prompt}")
      return f'Running notebook with prompt: {prompt}'
    except Exception as e:
      logger.error(f"Chat error: {e}")
      return f'Error: {str(e)}'
