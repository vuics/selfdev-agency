'''
CodeV1 Agent Archetype
'''
import os
import logging
import re
import time

# from dotenv import load_dotenv
from jupyter_client import KernelManager

from xmpp_agent import XmppAgent

logger = logging.getLogger("CodeV1")

# Load environment variables
# load_dotenv()

# MongoDB connection settings
# DB_URL = os.getenv("DB_URL", "mongodb://mongo.dev.local:27017/selfdev")


class CodeV1(XmppAgent):
  '''
  CodeV1 provides code execution with kernels similar to a Jupyter Notebook.
  '''
  async def start(self):
    await super().start()
    try:
      # logger.debug(f'self.config: {self.config}')
      self.code = self.config.options.code
      logger.debug(f'self.code: {self.code}')

      self.regex_start = re.compile(self.code.commands["start"])
      self.regex_restart = re.compile(self.code.commands["restart"])
      self.regex_reconnect = re.compile(self.code.commands["reconnect"])
      self.regex_shutdown = re.compile(self.code.commands["shutdown"])
      logger.debug(f'regex_start: {self.regex_start}')
      logger.debug(f'regex_restart: {self.regex_restart}')
      logger.debug(f'regex_reconnect: {self.regex_reconnect}')
      logger.debug(f'regex_shutdown: {self.regex_shutdown}')

      if self.code.kernel == "python3":
        pass
      elif self.code.kernel == "javascript":
        pass
      else:
        raise Exception('Unknown kernel')

      self.km = KernelManager(kernel_name=self.code.kernel)
      self.km.start_kernel()
      logger.debug(f'km: {self.km}')
      self.kc = self.km.client()
      self.kc.start_channels()
      logger.debug(f'kc: {self.kc}')

    except Exception as e:
      logger.error(f"Error initializing model: {e}")

  async def chat(self, *, prompt, reply_func=None):
    try:
      logger.debug(f"prompt: {prompt}")
      # logger.debug(f'self.config.options: {self.config.options}')
      content = ''

      if re.match(self.regex_start, prompt):
        logger.debug("START command")
        self.km.start_kernel()
        self.kc = self.km.client()
        self.kc.start_channels()
        logger.debug(f'kc: {self.kc}')
        content = "Started the kernel."

      if re.match(self.regex_restart, prompt):
        logger.debug("RESTART command")
        self.km.restart_kernel()
        time.sleep(3)
        self.kc.stop_channels()
        self.kc = self.km.client()
        self.kc.start_channels()
        content = "Restarted the kernel."

      if re.match(self.regex_reconnect, prompt):
        logger.debug("RECONNECT command")
        self.kc.stop_channels()
        self.kc = self.km.client()
        self.kc.start_channels()
        content = "Reconnected to the kernel."

      if re.match(self.regex_shutdown, prompt):
        self.kc.stop_channels()
        self.km.shutdown_kernel()
        content = "Shut down the kernel."

      else:
        self.kc.execute(prompt)
        while True:
          msg = self.kc.get_iopub_msg()
          if msg['msg_type'] == 'stream':
            text = msg['content']['text']
            content += text
            logger.debug(f"STDOUT: {text}")
          elif msg['msg_type'] == 'execute_result':
            data = msg['content']['data']['text/plain']
            text += data
            logger.debug(f"RESULT: {text}")
          elif msg['msg_type'] == 'error':
            error = msg['content']['evalue']
            content += error
            logger.debug(f"ERROR: {error}")
            break
          elif msg['msg_type'] == 'status' and msg['content']['execution_state'] == 'idle':
            break

      return content
    except Exception as e:
      logger.error(f"Storage error: {e}")
      return f'Error: {str(e)}'

  async def disconnect(self):
    """
    Release resources
    """
    try:
      await super().disconnect()
      self.kc.stop_channels()
      self.km.shutdown_kernel()
    except Exception as e:
      logger.error(f"Disconnect error: {e}")
