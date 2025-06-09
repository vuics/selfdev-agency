'''
TtsV1 Agent Archetype
'''
import logging

from openai import OpenAI

from xmpp_agent import XmppAgent

logger = logging.getLogger("TtsV1")


class TtsV1(XmppAgent):
  '''
  TtsV1 provides speech synthesis from text to audio
  '''
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # self.file_manager = FileManager()

  async def start(self):
    await super().start()
    try:
      # logger.debug(f"self.config.options.tts: {self.config.options.tts}")
      # logger.debug(f"self.config.options.tts.model: {self.config.options.tts.model}")
      if self.config.options.tts.model.provider == 'openai':
        self.client = OpenAI(
          api_key=self.config.options.tts.model.apiKey or None,
        )
      else:
        raise Exception(f"Unknown model provider: {self.config.options.tts.model.provider}")
      logger.debug(f"Client initialized: {self.client}")
    except Exception as e:
      logger.error(f"Error initializing model: {e}")

  async def chat(self, *, prompt, reply_func=None):
    try:
      logger.debug(f"Received prompt: {prompt}")

      if not self.client:
        raise Exception("Client was not initialized")

      if self.config.options.tts.model.provider == 'openai':
        result = self.client.audio.speech.create(
          model=self.config.options.tts.model.name,
          voice=self.config.options.tts.model.voice,
          input=prompt,
        )
        # logger.debug(f"openai result: {result}")
        file_bytes = result.content
        filename = "speech.mp3"
        content_type = "audio/mpeg"
      else:
        raise Exception(f"Unknown model provider: {self.config.options.tts.model.provider}")

      get_url = await self.upload_file(
        file_bytes=file_bytes,
        filename=filename,
        content_type=content_type,
      )
      logger.info(f"get_url: {get_url}")

      if reply_func:
        reply_func(get_url)

      content = f'<audio controls><source src="{get_url}" type="{content_type}">Your browser does not support the audio element.</audio>'
      logger.debug(f"content: {content}")

      return content
    except Exception as e:
      logger.error(f"Tts error: {e}")
      return f"Error: {str(e)}"
